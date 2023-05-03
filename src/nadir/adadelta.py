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


__all__ = ['AdadeltaConfig', 'Adadelta']

@dataclass
class AdadeltaConfig(AdamConfig):
  lr : float = 1
  rho : float = 0.90
  beta_1 : float = 0.0
  beta_2 : float = 0.90
  eps : float = 1E-6
  bias_correction : bool = False


class Adadelta(Adam):
  def __init__ (self, params, config : AdadeltaConfig = AdadeltaConfig()):
    super().__init__(params, config)

    self.config = config

    if self.config.rho != self.config.beta_2:
      self.config.beta_2 = self.config.rho
  
  def init_state(self, state, group, param):
    state['step'] = 0
    state['adaptivity'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    state['acc_delta'] = torch.zeros_like(param, memory_format=torch.preserve_format)

  def update(self, state, group, grad, param):
    eps = self.config.eps
    rho = self.config.rho
    lr = group['lr']
    m = state['acc_delta']

    denom = self.adaptivity(state, grad)

    delta = m.add(eps).sqrt_().div_(denom).mul_(grad)
    
    param.data.add_(delta, alpha = -1 * lr)

    m.mul_(rho).addcmul_(delta, delta, value=(1 - rho))
    state['acc_delta'] = m    