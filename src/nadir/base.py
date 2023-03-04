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
  momentum : bool = False
  adaptive : bool = False
  beta_1 : float = 0.0
  beta_2 : float = 0.0
  eps : float = 1E-8

  def dict(self):
    return self.__dict__


class BaseOptimizer(Optimizer):

  def __init__  (self, params, config: BaseConfig = BaseConfig()):
    if not config.lr > 0.0:
      raise ValueError(f"Invalid value for lr in config: {config.lr} ", 
                       "Value must be > 0")
    if not 1.0 > config.beta_1 >= 0.0:
      raise ValueError(f"Invalid value for beta_1 in config: {config.beta_1} ", 
                       "Value must be 1 > x >= 0")
    if not 1.0 > config.beta_2 >= 0.0:
      raise ValueError(f"Invalid value for beta_2 in config: {config.beta_2} ", 
                       "Value must be 1 > x >= 0")
    super().__init__(params, config.dict())

    self.config = config

  def init_state(self,
                 state,
                 group,
                 param):
    
    if self.config.momentum:
      state['momentum_step'] = 0
      state['momentum'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    if self.config.adaptive:
      state['adaptive_step'] = 0
      state['adaptivity'] = torch.zeros_like(param, memory_format=torch.preserve_format)
  
  def momentum(self,
               state, 
               grad):
    step = state['momentum_step']
    m = state['momentum']
    beta_1 = self.config.beta_1

    m.mul_(beta_1).add_(grad, alpha= (1 - beta_1))
    m.div_(1 - beta_1**(step + 1))

    state['momentum'] = m
    state['momentum_step'] = step + 1
    return m
  
  def adaptivity(self, 
                 state, 
                 grad):
    
    step = state['adaptive_step']
    v = state['adaptivity']
    beta_2 = self.config.beta_2

    v.mul_(beta_2).addcmul_(grad, grad, value = (1 - beta_2))
    v.div_(1 - beta_2**(step + 1))

    state['adaptivity'] = v
    state['step'] = step + 1
    return torch.sqrt(v + self.config.eps)

  def update(self,
             state: Dict[str, any],
             group: Dict[str, any],
             grad:  torch.Tensor,
             param: torch.Tensor):
    
    lr = group['lr']

    if self.config.momentum:
      m = self.momentum(state, grad)
      upd = m
    else:
      upd = grad
    
    if self.config.adaptive:
      v = self.adaptivity(state, grad)
      param.data.addcdiv_(upd, v, value = -1 * lr)
    else:
      param.data.add_(upd, alpha = -1 * lr)

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