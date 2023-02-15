from typing import Dict, Tuple, Any
from torch.optim.optimizer import Optimizer

import torch





class BaseOptimizer(Optimizer):
    def __init__(
        self, 
        params, 
        defaults: Dict[str, Any], 
        lr: float, 
        momentum: float, 
        nestrov:bool, 
        eps: float,
        betas: Tuple[float, float]

    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")



        defaults.update(dict(lr=lr, momentum=momentum, nestrov=nestrov, eps=eps, betas=betas))
        super().__init__(params, defaults)



