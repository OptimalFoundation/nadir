from typing import Dict, List, Type
from torch.optim.optimizer import Optimizer
from .optimizer import BaseOptimizer, DoEConfig

from .sgd import SGD, SGDConfig
from .adam import Adam, AdamConfig

__all__ = ('SGD', 'SGDConfig', 'Adam', 'AdamConfig')