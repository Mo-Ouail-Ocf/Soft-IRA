from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Simple numpy-backed replay buffer with torch sampling."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = int(1e6),
        device: str = "cpu",
    ) -> None:
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = device

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: float,
    ) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[indices], device=self.device),
            torch.as_tensor(self.action[indices], device=self.device),
            torch.as_tensor(self.next_state[indices], device=self.device),
            torch.as_tensor(self.reward[indices], device=self.device),
            torch.as_tensor(self.not_done[indices], device=self.device),
        )


class ActionBuffer:
    """Legacy compatibility buffer for old imports."""

    def __init__(self, action_dim: int, max_size: int = int(2e5), device: str = "cpu") -> None:
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.device = device

    def add(self, action, size=None) -> None:
        self.action[self.ptr] = action
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
