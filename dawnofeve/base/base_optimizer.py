
from abc import ABC, abstractmethod

import torch


class BaseOptimizer(ABC):
    
    @torch.no_grad()
    def reset(self):
        raise NotImplementedError