from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .common import DeterministicActor, TwinCritic, soft_update, tensor_to_action


class TD3:
    is_stochastic = False

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: str = "cpu",
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        **_: dict,
    ) -> None:
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        self.actor = DeterministicActor(state_dim, action_dim, max_action, hidden_dim=hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TwinCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            return tensor_to_action(self.actor(tensor_state))

    sample_action = select_action

    def estimate_q(self, state: np.ndarray, action: np.ndarray) -> float:
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            tensor_action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(1, -1)
            q1, q2 = self.critic(tensor_state, tensor_action)
            return float(torch.min(q1, q2).item())

    def train(self, replay_buffer, batch_size: int = 256) -> dict[str, float]:
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = None
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic, self.critic_target, self.tau)
            soft_update(self.actor, self.actor_target, self.tau)
            actor_loss_value = actor_loss.item()

        metrics = {
            "train/critic_loss": critic_loss.item(),
            "train/q1_mean": current_q1.mean().item(),
            "train/q2_mean": current_q2.mean().item(),
            "train/target_q_mean": target_q.mean().item(),
        }
        if actor_loss_value is not None:
            metrics["train/actor_loss"] = actor_loss_value
        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.total_it = state.get("total_it", 0)
