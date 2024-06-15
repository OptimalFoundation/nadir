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

@dataclass
class BaseConfig:
  lr : float = 1E-3
  weight_decay : float = 0.0

  averaging: Optional[str] = None
  schedulefree: Optional[float] = None

  def dict(self):
    return self.__dict__


class BaseOptimizer (Optimizer):

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
    
    # NOTE: If averaging is enabled, would use 2x extra memory
    # to store the parameters that are getting updated as well as
    # the running average of the parameters. Many methods like 
    # Polyak averaging or Primal Averaging use one or the other
    # to calculate the update and having both at hand lets us 
    # cater to both groups much more easily. 
    if self.config.averaging is not None: 
      state['iter_params'] = torch.clone(param.data, memory_format=torch.preserve_format).detach()
      state['avg_params'] = torch.clone(param.data, memory_format=torch.preserve_format).detach()

  
  def train(self):
    if self.config.averaging is not None:
      for group in self.param_groups:
        for param in group['params']:
          state = self.state[param]

          if self.config.averaging == 'schedulefree':
             beta = self.config.schedulefree
             param.data = torch.lerp(state['iter_params'], state['avg_params'], weight=beta)

  def eval(self):
    if self.config.averaging is not None:
      for group in self.param_groups:
        for param in group['params']:
          state = self.state[param]

          if self.config.averaging == 'schedulefree':
             param.data = state['avg_params']

  
  def schedulefree(self, 
                   state: Dict[str, any], 
                   group: Dict[str, any],
                   update: torch.Tensor, 
                   param: torch.Tensor) -> None:
    r"""
    Updates the `iter_params` and `avg_params` using the schedulefree method.
    Changes the behaviour of the optimizer.train() and optimizer.eval() methods.
    By default, assumes that you need the `avg_params` at the time of eval().
    
    args:
      state: Dict[str, any]
      group: Dict[str, any]
      update: torch.Tensor
      param: torch.Tensor
    
    returns:
      None
    """
    # assert if schedulefree param is None or proper float value

    lr = group['lr']
    beta = self.config.schedulefree
    step = state['step']
    ct = 1/(step+1)

    iter_params = state['iter_params']
    avg_params = state['avg_params']
      
    iter_params.add_(update, alpha= -1 * lr)
    avg_params.lerp_(iter_params, weight=ct)
    schedulefree = torch.lerp(iter_params, avg_params, beta)

    param.data = schedulefree


  def average(self,
              state: Dict[str, any], 
              group: Dict[str, any],
              update: torch.Tensor, 
              param: torch.Tensor
              ) -> None:
    r"""
    This method handles the routing to the appropriate averaging method.

    Assumes that there is some none null value provided to the `averaging`
    argument in the config. 

    Creates `iter_params` and `avg_params` due to averaging method which 
    increase the memory requirements by 2x. Methods need to switch between
    the two at the time of train() or eval(), and having both at hand 
    makes it easier to use. 

    Current options are:
    - schedulefree

    args:
      state: Dict[str, any]
      group: Dict[str, any]
      update: torch.Tensor
      param: torch.Tensor
    
    returns:
      None
    """
    
    # Get the averaging method from config
    averaging_method = self.config.averaging

    if averaging_method == 'schedulefree':
      self.schedulefree(state, group, update, param)
    elif averaging_method == 'swa': #TODO
      raise NotImplementedError
    elif averaging_method == 'polyak': #TODO
      raise NotImplementedError
    elif averaging_method == 'primal': #TODO
      raise NotImplementedError
  
  def apply_update(self, 
                   state: Dict[str, any], 
                   group: Dict[str, any],
                   update: torch.Tensor, 
                   param: torch.Tensor
                   ) -> None:

      if self.config.averaging is not None:
        self.average(state, group, update, param)
      else:
        # Default behaviour is to take a step in the direction of update
        lr = group['lr']
        param.data.add_(update, alpha=-1*lr)

  def update(self,
             state: Dict[str, any],
             group: Dict[str, any],
             grad:  torch.Tensor,
             param: torch.Tensor):
    
    # lr = group['lr']
    # param.data.add_(grad, alpha = -1 * lr)

    # if self.config.weight_decay > 0:
    #   param.data.add_(param.data,
                      # alpha = -1 * lr * self.config.weight_decay)
    
    self.apply_update(state, group, grad, param)
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