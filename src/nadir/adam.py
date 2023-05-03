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


__all__ = ['AdamConfig', 'Adam']

@dataclass
class AdamConfig(BaseConfig):
  lr : float = 3E-4
  beta_1 : float = 0.9
  beta_2 : float = 0.999
  eps : float = 1E-8
  weight_decay : float = 0.0
  amsgrad : bool = False
  nesterov : bool = False
  bias_correction: bool = True

class Adam(BaseOptimizer):

  def __init__(self, params, config : AdamConfig = AdamConfig()):
    if not 1 >= config.beta_1 >= 0:
      raise ValueError(f"Invalid value for beta_1 in config: {config.beta_1} ", 
                       "Value must be between 0 and 1")
    if not 1 >= config.beta_2 >= 0:
      raise ValueError(f"Invalid value for beta_2 in config: {config.beta_2} ", 
                       "Value must be between 0 and 1")
    super().__init__(params, config)

    self.config = config

  def init_state(self,
                 state,
                 group,
                 param):
    state['step'] = 0

    if 1 >= self.config.beta_1 > 0:
      state['momentum'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    
    if 1 >= self.config.beta_2 > 0:
      state['adaptivity'] = torch.zeros_like(param, memory_format=torch.preserve_format)
      
      if self.config.amsgrad:
        state['amsgrad'] = torch.zeros_like(param, memory_format=torch.preserve_format)

  def nesterov(momentum):

    def __momentum__(self, state, grad):
      m = momentum(self, state, grad)
      
      if self.config.nesterov:
        beta = self.config.beta_1
        step = state['step']

        if step > 0:
          if self.config.bias_correction:
            grad_hat = grad.mul(1-beta).div(1 - beta ** (step))
          else:
            grad_hat = grad
          n = m.mul(beta).add_(grad_hat)
        else:
          n = grad
    
        return n

      return m

    return __momentum__
  
  @nesterov
  def momentum(self,
               state, 
               grad):
    step = state['step']
    m = state['momentum']
    beta_1 = self.config.beta_1
    bias_correction = self.config.bias_correction

    m.mul_(beta_1).add_(grad, alpha= (1 - beta_1))
    
    if bias_correction:
      m_hat = m.div(1 - beta_1**(step + 1))
    else:
      m_hat = m

    state['momentum'] = m
    return m_hat

  def amsgrad(adaptivity):

    def __adaptivity__(self, state, grad):
      u = adaptivity(self, state, grad)

      if self.config.amsgrad:
        v = state['amsgrad']
        v = torch.max(v, u)
        state['amsgrad'] = v
        return v
      
      return u

    return __adaptivity__

  @amsgrad
  def adaptivity(self, 
                 state, 
                 grad):
    
    step = state['step']
    v = state['adaptivity']
    beta_2 = self.config.beta_2
    bias_correction = self.config.bias_correction

    v.mul_(beta_2).addcmul_(grad, grad, value = (1 - beta_2))

    if bias_correction:
      v_hat = v.div(1 - beta_2**(step + 1))
    else:
      v_hat = v
    
    state['adaptivity'] = v
    return torch.sqrt(v_hat + self.config.eps)

  def update(self,
             state: Dict[str, any],
             group: Dict[str, any],
             grad:  torch.Tensor,
             param: torch.Tensor):
    
    lr = group['lr']
    beta_1 = self.config.beta_1
    beta_2 = self.config.beta_2

    if 1 >= beta_1 > 0:
      m = self.momentum(state, grad)
      upd = m
    else:
      upd = grad
    
    if 1>= beta_2 > 0:
      v = self.adaptivity(state, grad)
      param.data.addcdiv_(upd, v, value = -1 * lr)
    else:
      param.data.add_(upd, alpha = -1 * lr)

    if self.config.weight_decay > 0:
      param.data.add_(param.data,
                      alpha = -1 * lr * self.config.weight_decay)
      
    state['step'] += 1