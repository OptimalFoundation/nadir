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

import torch
from dataclasses import dataclass

from .BaseOptimiser import BaseOptimizer
from .BaseOptimiser import BaseConfig
from typing import Dict, Any, Optional


__all__ = ['SGD', 'sgd']

@dataclass
class SGDConfig(BaseConfig):
    lr : float = 1e-3
    momentum: float = 0
    nesterov: bool = False

class SGD(BaseOptimizer):
    def __init__(self, params, config: SGDConfig, defaults: Optional[Dict[str, Any]] = None):
        if not 0.0 <= config.lr:
            raise ValueError(f"Invalid learning rate: {config.lr}")
        if not 0.0 <= config.momentum < 1.0:
            raise ValueError(f"Invalid momentum: {config.momentum}")
        defaults = {} if defaults is None else defaults
        super().__init__(params, config, defaults)


    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()
                
        for group in self.param_groups:
            # weight_decay = group['weight_decay']
            momentum = group['momentum']
            # dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # if weight_decay != 0:
                #     d_p.add_(weight_decay, p.data)
                # Apply learning rate  
                d_p.mul_(group['lr'])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    # else:
                    #     buf = param_state['momentum_buffer']
                    #     buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-1)

        return loss