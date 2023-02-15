from typing import Dict, Tuple, Any, Optional

import torch
from torch.optim.optimizer import Optimizer


class DoEConfig():
    pass



class BaseOptimizer(Optimizer):
    def __init__(
        self, 
        params, 
        config : DoEConfig, 
        defaults: Dict[str, Any] 


    ):
        defaults = config.__dict__

        super().__init__(params, defaults)
    
    



