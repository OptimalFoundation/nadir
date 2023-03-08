### Copyright 2023 [Dawn Of Eve]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Type
from torch.optim.optimizer import Optimizer


from .adadelta import Adadelta, AdadeltaConfig
from .adagrad import Adagrad, AdagradConfig
from .adam import Adam, AdamConfig
from .adamax import Adamax, AdamaxConfig
from .base import BaseOptimizer, BaseConfig
from .lion import Lion, LionConfig
from .momentum import Momentum, MomentumConfig
from .rmsprop import RMSProp, RMSPropConfig
from .radam import Radam, RadamConfig
from .sgd import SGD, SGDConfig


__version__ = "0.0.3"

__all__ = ('Adadelta',
           'AdadeltaConfig',
           'Adagrad',
           'AdagradConfig',
           'Adam',
           'AdamConfig',
           'Adamax', 
           'AdamaxConfig',
           'Adam' 
           'BaseOptimizer',
           'BaseConfig',
           'Lion',
           'LionConfig',
           'Momentum',
           'MomentumConfig',
           'RMSProp',
           'RMSPropConfig',
           'Radam',
           'RadamConfig',
           'SGD',
           'SGDConfig')
