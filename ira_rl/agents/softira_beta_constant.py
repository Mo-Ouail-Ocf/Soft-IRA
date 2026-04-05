import math

import torch

from .softira import SoftIRA


class SoftIRABetaConstant(SoftIRA):
    """SoftIRA variant with a frozen beta weight for the GAG term."""

    def __init__(self, *args, fixed_beta=1.0, **kwargs):
        self.fixed_beta = float(fixed_beta)
        super().__init__(*args, **kwargs)

        init_beta = max(self.fixed_beta, 1e-8)
        self.log_beta.data.fill_(math.log(init_beta))
        self._fixed_beta_tensor = torch.tensor([self.fixed_beta], device=self.device)

    @property
    def beta(self):
        return self._fixed_beta_tensor

    def _update_beta(self, deviation: torch.Tensor) -> float:
        return 0.0
