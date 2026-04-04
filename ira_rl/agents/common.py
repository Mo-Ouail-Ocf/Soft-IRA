from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    return optimizer.state_dict()


def module_state(module: nn.Module) -> dict:
    return module.state_dict()


def load_module_state(module: nn.Module, state: dict) -> None:
    module.load_state_dict(state)


def tensor_to_action(action: torch.Tensor) -> np.ndarray:
    return action.detach().cpu().numpy().flatten()


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.l1(state))
        hidden = F.relu(self.l2(hidden))
        return self.max_action * torch.tanh(self.l3(hidden))


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def _trunk(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.l1(state))
        hidden = F.relu(self.l2(hidden))
        mean = self.mean_layer(hidden)
        log_std = self.log_std_layer(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self._trunk(state)
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = dist.rsample()
        normalized_action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - normalized_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic_action = torch.tanh(mean) * self.max_action
        return normalized_action * self.max_action, log_prob, deterministic_action

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self._trunk(state)
        return torch.tanh(mean) * self.max_action

    def log_prob_from_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, std = self._trunk(state)
        normalized_action = (action / self.max_action).clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(normalized_action)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - normalized_action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_features: bool = False,
    ):
        state_action = torch.cat([state, action], dim=1)
        features_1 = F.relu(self.l2(F.relu(self.l1(state_action))))
        features_2 = F.relu(self.l5(F.relu(self.l4(state_action))))
        q1 = self.l3(features_1)
        q2 = self.l6(features_2)
        if return_features:
            return q1, q2, features_1, features_2
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor, return_features: bool = False):
        state_action = torch.cat([state, action], dim=1)
        features = F.relu(self.l2(F.relu(self.l1(state_action))))
        value = self.l3(features)
        if return_features:
            return value, features
        return value

    def q2(self, state: torch.Tensor, action: torch.Tensor, return_features: bool = False):
        state_action = torch.cat([state, action], dim=1)
        features = F.relu(self.l5(F.relu(self.l4(state_action))))
        value = self.l6(features)
        if return_features:
            return value, features
        return value


class SingleCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_features: bool = False,
    ):
        state_action = torch.cat([state, action], dim=1)
        features = F.relu(self.l2(F.relu(self.l1(state_action))))
        value = self.l3(features)
        if return_features:
            return value, features
        return value


class ActionMemory:
    def __init__(self, action_dim: int, size: int, device: str) -> None:
        self.storage = torch.zeros((int(size), action_dim), device=device)
        self.size = int(size)
        self.index = 0
        self.is_full = False

    @property
    def count(self) -> int:
        return self.size if self.is_full else self.index

    def add_batch(self, actions: torch.Tensor) -> None:
        batch_size = actions.shape[0]
        end = self.index + batch_size
        if end > self.size:
            overflow = end - self.size
            self.storage[self.index :] = actions[: batch_size - overflow]
            self.storage[:overflow] = actions[batch_size - overflow :]
            self.index = overflow
            self.is_full = True
            return
        self.storage[self.index : end] = actions
        self.index = end % self.size
        if end >= self.size:
            self.is_full = True

    def valid(self) -> torch.Tensor:
        if self.is_full:
            return self.storage
        return self.storage[: self.index]


def chunked_linf_knn(query: torch.Tensor, memory: torch.Tensor, k: int, chunk_size: int = 4096) -> torch.Tensor:
    """Fallback kNN for CPU or environments without GPU FAISS."""
    all_indices = []
    for start in range(0, query.shape[0], chunk_size):
        batch = query[start : start + chunk_size]
        distances = torch.max(torch.abs(batch.unsqueeze(1) - memory.unsqueeze(0)), dim=-1).values
        indices = torch.topk(distances, k=min(k, memory.shape[0]), largest=False, dim=1).indices
        all_indices.append(indices)
    return torch.cat(all_indices, dim=0)
