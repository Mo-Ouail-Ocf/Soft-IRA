from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .common import GaussianActor, TwinCritic, soft_update, tensor_to_action


class SAC:
    is_stochastic = True

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: str = "cpu",
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
        alpha_init: float = 1.0,
        auto_alpha: bool = True,
        use_double_q: bool = True,
        **_: dict,
    ) -> None:
        self.device = device
        self.discount = discount
        self.tau = tau
        self.use_double_q = use_double_q
        self.auto_alpha = auto_alpha

        self.actor = GaussianActor(state_dim, action_dim, max_action, hidden_dim=hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TwinCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_entropy = float(-action_dim if target_entropy is None else target_entropy)
        self.log_alpha = torch.tensor(
            [float(np.log(alpha_init))],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            return tensor_to_action(self.actor.deterministic(tensor_state))

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            action, _, _ = self.actor.sample(tensor_state)
            return tensor_to_action(action)

    def estimate_q(self, state: np.ndarray, action: np.ndarray) -> float:
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            tensor_action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(1, -1)
            q1, q2 = self.critic(tensor_state, tensor_action)
            if self.use_double_q:
                return float(torch.min(q1, q2).item())
            return float(q1.item())

    def train(self, replay_buffer, batch_size: int = 256) -> dict[str, float]:
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        sampled_action, log_prob, _ = self.actor.sample(state)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            next_q = torch.min(target_q1, target_q2) if self.use_double_q else target_q1
            target_q = reward + not_done * self.discount * (next_q - self.alpha.detach() * next_log_prob)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q)
        if self.use_double_q:
            critic_loss = critic_loss + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        q1_pi, q2_pi = self.critic(state, sampled_action)
        min_q_pi = torch.min(q1_pi, q2_pi) if self.use_double_q else q1_pi
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss_value = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = alpha_loss.item()

        soft_update(self.critic, self.critic_target, self.tau)

        return {
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/alpha": self.alpha.item(),
            "train/alpha_loss": alpha_loss_value,
            "train/entropy": (-log_prob.mean()).item(),
            "train/q1_mean": current_q1.mean().item(),
            "train/q2_mean": current_q2.mean().item(),
            "train/target_q_mean": target_q.mean().item(),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "use_double_q": self.use_double_q,
            "auto_alpha": self.auto_alpha,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha.data = state["log_alpha"].to(self.device)
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
