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


__all__ = ['AdamWConfig', 'AdamW']

@dataclass
class AdamWConfig(AdamConfig):
  lr : float = 3E-4
  weight_decay : float = 0.01

class AdamW(Adam):
  def __init__ (self, params, config : AdamWConfig = AdamWConfig()):

    if not 1.0 > config.weight_decay > 0.0:
      raise ValueError(f"Invalid value for weight_decay in config: {config.weight_decay} ", 
                       "Value must be float between 0 and 1")  
      
    super().__init__(params, config)
    
    self.config = config