from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

try:  # noqa: F401
    import gymnasium_robotics  # type: ignore
except ImportError:  # pragma: no cover
    gymnasium_robotics = None


def make_env(env_id: str):
    return gym.make(env_id)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_env(env: Any, seed: int | None = None):
    if seed is None:
        result = env.reset()
    else:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env: Any, action):
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return next_state, reward, terminated or truncated, info
    return result


def seed_action_space(env: Any, seed: int) -> None:
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)


def compute_discounted_returns(rewards: list[float], discount: float) -> list[float]:
    returns = [0.0 for _ in rewards]
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + discount * running
        returns[index] = running
    return returns
