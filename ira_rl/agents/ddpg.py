from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .common import DeterministicActor, SingleCritic, soft_update, tensor_to_action


class DDPG:
    is_stochastic = False

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: str = "cpu",
        hidden_dim: int = 256,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        critic_weight_decay: float = 1e-2,
        discount: float = 0.99,
        tau: float = 0.001,
        **_: dict,
    ) -> None:
        self.device = device
        self.discount = discount
        self.tau = tau

        self.actor = DeterministicActor(state_dim, action_dim, max_action, hidden_dim=hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = SingleCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=critic_weight_decay,
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            return tensor_to_action(self.actor(tensor_state))

    sample_action = select_action

    def estimate_q(self, state: np.ndarray, action: np.ndarray) -> float:
        with torch.no_grad():
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
            tensor_action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(1, -1)
            return float(self.critic(tensor_state, tensor_action).item())

    def train(self, replay_buffer, batch_size: int = 64) -> dict[str, float]:
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + not_done * self.discount * self.critic_target(next_state, next_action)

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_action = self.actor(state)
        actor_loss = -self.critic(state, actor_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.critic, self.critic_target, self.tau)
        soft_update(self.actor, self.actor_target, self.tau)

        return {
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/q_mean": current_q.mean().item(),
            "train/target_q_mean": target_q.mean().item(),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discount": self.discount,
            "tau": self.tau,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
