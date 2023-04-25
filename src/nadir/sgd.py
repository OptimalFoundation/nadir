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


__all__ = ['SGDConfig', 'SGD']

@dataclass
class SGDConfig(BaseConfig):
  lr : float = 1E-3
  momentum : float = 0.0
  dampening : float = 0.0
  weight_decay : float = 0.0

class SGD(BaseOptimizer):
  def __init__(self, params, config: SGDConfig = SGDConfig()):
    super().__init__(params, config)

  def init_state(self,
                 state,
                 group,
                 param):
    state['step'] = 0

    if 1 >= self.config.momentum > 0:
      state['momentum'] = torch.zeros_like(param, memory_format=torch.preserve_format)


  def momentum(self,
               state, 
               grad):
    step = state['step']
    m = state['momentum']
    beta = self.config.momentum
    gamma = self.config.dampening

    m.mul_(beta).add_(grad, alpha=(1 - gamma))

    state['momentum'] = m
    return m

  def update(self,
             state: Dict[str, any],
             group: Dict[str, any],
             grad:  torch.Tensor,
             param: torch.Tensor):
    
    lr = group['lr']
    beta = self.config.momentum

    if beta != 0.0:
      m = self.momentum(state, grad)
    else:
      m = grad

    param.data.add_(m, alpha = -1 * lr)

    if self.config.weight_decay > 0:
      param.data.add_(param.data,
                      alpha = -1 * lr * self.config.weight_decay)
    state['step'] += 1