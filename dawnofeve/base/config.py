from dataclasses import dataclass
from typing import Tuple

class DoEConfig():
    pass

@dataclass
class SGDConfig(DoEConfig):
    lr : float = 1e-3
    momentum: float = 0
    nesterov: bool = False