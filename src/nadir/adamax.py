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

from .adam import Adam, AdamConfig

__all__ = ['AdamaxConfig', 'Adamax']

@dataclass
class AdamaxConfig(AdamConfig):
  lr : float = 2E-3
  momentum : bool = True
  adaptive : bool = True
  beta_1 : float = 0.9
  beta_2 : float = 0.999
  eps : float = 1E-8

class Adamax(Adam):
  def __init__ (self, params, config : AdamaxConfig = AdamaxConfig()):
    if not config.momentum:
      raise ValueError(f"Invalid value for momentum in config: {config.momentum} ", 
                       "Value must be True")
    if not config.adaptive:
      raise ValueError(f"Invalid value for adaptive in config: {config.adaptive} ", 
                       "Value must be True")
    
    super().__init__(params, config)
    self.config = config
  
  def adaptivity(self, state, grad):
    u = state['adaptivity']
    beta_2 = self.config.beta_2

    u = torch.max(torch.mul(u, beta_2), torch.abs(grad) + self.config.eps)
    
    state['adaptivity'] = u
    return u