from typing import Dict, List, Type
from torch.optim.optimizer import Optimizer
from .optimizer import BaseOptimizer, DoEConfig

from .sgd import SGD, SGDConfig

__all__ = ('SGD', 'sgd', 'SGDConfig')