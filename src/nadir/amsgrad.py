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

from .adam import Adam, AdamConfig

__all__ = ['AMSGradConfig', 'AMSGrad']

@dataclass
class AMSGradConfig(AdamConfig):
  lr : float = 3E-4
  amsgrad : bool = True

class AMSGrad(Adam):
  def __init__ (self, params, config : AMSGradConfig = AMSGradConfig()):
    
    if not config.amsgrad:
      raise ValueError(f"Invalid value for amsgrad in config: {config.amsgrad} ", 
                       "Value must be True")
    super().__init__(params, config)
    
    self.config = config