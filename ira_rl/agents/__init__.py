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

__all__ = [
    "DDPG",
    "IRA",
    "IRA_DDPG",
    "PEER",
    "SAC",
    "SoftIRA",
    "SoftIRABetaConstant",
    "SoftIRABetaDecay",
    "TD3",
    "memTD3",
]
