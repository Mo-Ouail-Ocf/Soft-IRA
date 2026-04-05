from __future__ import annotations

from omegaconf import OmegaConf

from .alh import memTD3
from .ddpg import DDPG
from .ira import IRA
from .ira_ddpg import IRA_DDPG
from .peer import PEER
from .sac import SAC
from .softira import SoftIRA
from .softira_beta_constant import SoftIRABetaConstant
from .softira_beta_decay import SoftIRABetaDecay
from .td3 import TD3


def build_agent(
    algorithm_cfg,
    state_dim: int,
    action_dim: int,
    max_action: float,
    device: str,
    total_timesteps: int | None = None,
):
    params = OmegaConf.to_container(algorithm_cfg, resolve=True)
    assert isinstance(params, dict)
    name = str(params.pop("name")).lower()
    params["device"] = device
    params["state_dim"] = state_dim
    params["action_dim"] = action_dim
    params["max_action"] = max_action
    if total_timesteps is not None and name.startswith("softira"):
        params["total_timesteps"] = int(total_timesteps)

    registry = {
        "ddpg": DDPG,
        "td3": TD3,
        "ira": IRA,
        "ira_ddpg": IRA_DDPG,
        "peer": PEER,
        "alh": memTD3,
        "sac": SAC,
        "softira": SoftIRA,
        "softira_beta_constant": SoftIRABetaConstant,
        "softira_beta_decay": SoftIRABetaDecay,
    }
    if name not in registry:
        raise ValueError(f"Unsupported algorithm '{algorithm_cfg.name}'.")
    return registry[name](**params)
