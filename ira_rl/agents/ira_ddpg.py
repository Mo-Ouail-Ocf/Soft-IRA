from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .common import ActionMemory, DeterministicActor, SingleCritic, tensor_to_action

try:
    from faiss import METRIC_L2, StandardGpuResources
    from faiss.contrib.torch_utils import torch_replacement_knn_gpu
except ImportError:  # pragma: no cover
    METRIC_L2 = None
    StandardGpuResources = None
    torch_replacement_knn_gpu = None


def _chunked_l2_knn(query: torch.Tensor, memory: torch.Tensor, k: int, chunk_size: int = 4096) -> torch.Tensor:
    all_indices = []
    for start in range(0, query.shape[0], chunk_size):
        batch = query[start : start + chunk_size]
        distances = torch.cdist(batch, memory)
        indices = torch.topk(distances, k=min(k, memory.shape[0]), largest=False, dim=1).indices
        all_indices.append(indices)
    return torch.cat(all_indices, dim=0)


class IRA_DDPG:
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
        alpha: float = 5e-4,
        mu: float = 1.0,
        k: int = 10,
        warmup_timesteps: int = 4000,
        action_buffer_size: int = 200_000,
        decay_mu: bool = True,
        **_: dict,
    ) -> None:
        self.device = device
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.mu = mu
        self.k = k
        self.warmup_timesteps = warmup_timesteps
        self.decay_mu = decay_mu
        self.total_it = 0

        self.actor = DeterministicActor(state_dim, action_dim, max_action, hidden_dim=hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = SingleCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_memory = ActionMemory(action_dim, size=action_buffer_size, device=device)
        self.faiss_resource = None
        if (
            torch_replacement_knn_gpu is not None
            and StandardGpuResources is not None
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.set_device(int(str(device).replace("cuda:", "")))
            self.faiss_resource = StandardGpuResources()

    def _adjust_mu(self) -> None:
        self.mu = self.mu - (self.mu - 0.1) / 100
        self.mu = max(self.mu, 0.1)

    def _knn(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if self.faiss_resource is not None:
            _, indices = torch_replacement_knn_gpu(
                self.faiss_resource,
                query.detach(),
                memory.detach(),
                min(self.k, memory.shape[0]),
                metric=METRIC_L2,
            )
            return indices
        return _chunked_l2_knn(query.detach(), memory.detach(), self.k)

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

    def train(self, replay_buffer, batch_size: int = 256) -> dict[str, float]:
        if self.decay_mu and self.total_it > 0 and self.total_it % 10_000 == 0:
            self._adjust_mu()
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        self.action_memory.add_batch(action.detach())

        with torch.no_grad():
            target_q = reward + not_done * self.discount * self.critic_target(next_state, self.actor_target(next_state))

        predicted_action = self.actor(state)
        post_warmup = self.total_it > self.warmup_timesteps and self.action_memory.count >= max(2, self.k)

        optimal_action = None
        peer_loss_value = 0.0
        current_q, current_features = self.critic(state, action, return_features=True)
        critic_loss = F.mse_loss(current_q, target_q)

        if post_warmup:
            with torch.no_grad():
                memory = self.action_memory.valid()
                indices = self._knn(predicted_action, memory)
                neighbor_actions = memory[indices]
                batch_indices = torch.arange(state.shape[0], device=self.device)
                flat_state = state.unsqueeze(1).expand(-1, neighbor_actions.shape[1], -1).reshape(
                    state.shape[0] * neighbor_actions.shape[1],
                    state.shape[1],
                )
                flat_actions = neighbor_actions.reshape(state.shape[0] * neighbor_actions.shape[1], predicted_action.shape[1])
                ranking = self.critic_target(flat_state, flat_actions).reshape(state.shape[0], neighbor_actions.shape[1])
                sorted_indices = torch.argsort(ranking, dim=-1, descending=True)
                optimal_action = neighbor_actions[batch_indices, sorted_indices[:, 0]]
                suboptimal_action = neighbor_actions[batch_indices, sorted_indices[:, 1] if neighbor_actions.shape[1] > 1 else sorted_indices[:, 0]]
                _, sub_features = self.critic_target(state, suboptimal_action, return_features=True)
            peer_loss = torch.einsum("ij,ij->i", current_features, sub_features).mean() * self.alpha
            critic_loss = critic_loss + peer_loss
            peer_loss_value = peer_loss.item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, predicted_action).mean()
        guidance_loss_value = 0.0
        if post_warmup and optimal_action is not None:
            guidance_loss = self.mu * (predicted_action - optimal_action).pow(2).mean()
            actor_loss = actor_loss + guidance_loss
            guidance_loss_value = guidance_loss.item()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return {
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/peer_loss": peer_loss_value,
            "train/guidance_loss": guidance_loss_value,
            "train/mu": self.mu,
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
            "memory": self.action_memory.valid().detach().cpu(),
            "memory_index": self.action_memory.index,
            "memory_full": self.action_memory.is_full,
            "total_it": self.total_it,
            "mu": self.mu,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        memory = state["memory"].to(self.device)
        self.action_memory.storage.zero_()
        self.action_memory.storage[: memory.shape[0]] = memory
        self.action_memory.index = state.get("memory_index", memory.shape[0])
        self.action_memory.is_full = state.get("memory_full", False)
        self.total_it = state.get("total_it", 0)
        self.mu = state.get("mu", self.mu)
