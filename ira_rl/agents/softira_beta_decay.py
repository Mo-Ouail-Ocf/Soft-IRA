import math

import torch

from .softira import SoftIRA


class SoftIRABetaDecay(SoftIRA):
    """SoftIRA variant with a linear beta schedule from max_beta to min_beta."""

    def __init__(
        self,
        *args,
        min_beta=0.1,
        max_beta=1.0,
        total_timesteps=None,
        beta_decay_start=None,
        **kwargs,
    ):
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.schedule_total_timesteps = None if total_timesteps is None else int(total_timesteps)
        super().__init__(*args, **kwargs)

        if self.min_beta < 0.0:
            raise ValueError("min_beta must be non-negative.")
        if self.max_beta < self.min_beta:
            raise ValueError("max_beta must be greater than or equal to min_beta.")

        self.beta_decay_start = self.warmup_timestamps if beta_decay_start is None else int(beta_decay_start)
        schedule_total = self.schedule_total_timesteps or (self.beta_decay_start + 1)
        self.beta_decay_steps = max(schedule_total - self.beta_decay_start, 1)

        init_beta = max(self.max_beta, 1e-8)
        self.log_beta.data.fill_(math.log(init_beta))

    def _beta_value(self) -> float:
        progress_steps = max(self.total_it - self.beta_decay_start, 0)
        progress = min(progress_steps / self.beta_decay_steps, 1.0)
        return self.max_beta + (self.min_beta - self.max_beta) * progress

    @property
    def beta(self):
        return torch.tensor([self._beta_value()], device=self.device)

    def _update_beta(self, deviation: torch.Tensor) -> float:
        return 0.0
