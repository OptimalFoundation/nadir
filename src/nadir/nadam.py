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

__all__ = ['Nadam', 'NadamConfig']

@dataclass
class NadamConfig(AdamConfig):
  lr : float = 3E-4
  nesterov : bool = True

class Nadam(Adam):
  def __init__ (self, params, config : NadamConfig = NadamConfig()):
    super().__init__(params, config)
    self.config = config
  