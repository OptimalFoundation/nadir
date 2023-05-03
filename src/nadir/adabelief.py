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

__all__ = ['Adabelief', 'AdabeliefConfig']

@dataclass
class AdabeliefConfig(AdamConfig):
  lr : float = 3E-4
  nesterov : bool = True

class Adabelief(Adam):
  def __init__ (self, params, config : AdabeliefConfig = AdabeliefConfig()):
    super().__init__(params, config)
    self.config = config
  
  @Adam.amsgrad
  def adaptivity(self, 
                 state, 
                 grad):
    
    step = state['step']
    v = state['adaptivity']
    m = state['momentum']
    beta_2 = self.config.beta_2
    bias_correction = self.config.bias_correction

    v.mul_(beta_2).addcmul_(grad - m, grad - m, value = (1 - beta_2))

    if bias_correction:
      v_hat = v.div(1 - beta_2**(step + 1))
    else:
      v_hat = v
    
    state['adaptivity'] = v
    return torch.sqrt(v_hat + self.config.eps)