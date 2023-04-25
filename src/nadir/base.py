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

from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass

import torch
from torch.optim.optimizer import Optimizer

__all__ = ['BaseConfig', 'BaseOptimizer']

@dataclass
class BaseConfig:
  lr : float = 1E-3
  weight_decay : float = 0.0

  def dict(self):
    return self.__dict__


class BaseOptimizer (optim.Optimizer):

  def __init__  (self, params, config: BaseConfig = BaseConfig()):
    if not config.lr > 0.0:
      raise ValueError(f"Invalid value for lr in config: {config.lr} ", 
                       "Value must be > 0")
    if not 1.0 > config.weight_decay >= 0.0:
      raise ValueError(f"Invalid value for weight decay in config: {config.weight_decay} ", 
                       "Value must be between 1 and 0")
    super().__init__(params, config.dict())

    self.config = config

  def init_state(self,
                 state,
                 group,
                 param):
    state['step'] = 0

  def update(self,
             state: Dict[str, any],
             group: Dict[str, any],
             grad:  torch.Tensor,
             param: torch.Tensor):
    
    lr = group['lr']
    param.data.add_(grad, alpha = -1 * lr)

    if self.config.weight_decay > 0:
      param.data.add_(param.data,
                      alpha = -1 * lr * self.config.weight_decay)
    state['step'] += 1

  @torch.no_grad()
  def step(self, closure = None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for param in group['params']:
        if param.grad is None:
         continue
        grad = param.grad.data
        if grad.is_sparse:
          raise RuntimeError('This Optimizer does not support sparse gradients,'
                                  ' please consider SparseAdam instead')
        state = self.state[param]
        if len(state) == 0:
          self.init_state(state, group, param)

        self.update(state, group, grad, param)
    
    return loss