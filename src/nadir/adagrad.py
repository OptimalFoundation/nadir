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
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch

from .base import BaseOptimizer
from .base import BaseConfig


__all__ = ['AdagradConfig', 'Adagrad']

@dataclass
class AdagradConfig(BaseConfig):
  lr : float = 1E-3
  adaptive : bool = True
  eps : float = 1E-8

class Adagrad(BaseOptimizer):

  def __init__ (self, params, config : AdagradConfig = AdagradConfig()):
    if not config.adaptive:
      raise ValueError(f"Invalid value for adaptive in config: {config.adaptive} ", 
                       "Value must be True")
    super().__init__(params, config)
    self.config = config

  def adaptivity(self, 
                 state,
                 grad):
    
    v = state['adaptivity']
    v.add_(torch.pow(grad, 2))
    state['adaptivity'] = v

    return torch.sqrt(v + self.config.eps)