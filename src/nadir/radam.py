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
import math

from .adam import Adam, AdamConfig
from .lion import LionConfig



__all__ = ['RadamConfig', 'Radam']

@dataclass
class RadamConfig(AdamConfig):
  lr : float = 3E-4
  beta_1 : float = 0.9
  beta_2 : float = 0.99
  eps : float = 1E-8
  weight_decay : float = 0.

class Radam(Adam):
  def __init__ (self, params, config : LionConfig = LionConfig()):
    super().__init__(params, config)
    self.config = config

  def update(self, state, group, grad, param):
    lr = group['lr']
    beta_2 = self.config.beta_2
    step = state['step']
    p_inf = 2/(1-beta_2) - 1

    m = self.momentum(state, grad)
    v = self.adaptivity(state, grad)

    p_t = p_inf - 2 * step * (beta_2 ** (step+1))/(1 - beta_2**(step+1))

    if p_t > 4:
      numrt = ((p_t - 4) * (p_t - 2) * p_inf)
      denomrt = ((p_inf - 4) * (p_inf - 2) * p_t)
      r_t  = math.sqrt(numrt/denomrt)

      param.data.addcdiv_(m, v, value= -1 * r_t * lr)
    else:
      param.data.add_(m, alpha= -lr)

    if self.config.weight_decay > 0:
      param.data.add_(param.data,
                      alpha = -1 * lr * self.config.weight_decay)
    
    state['step'] += 1