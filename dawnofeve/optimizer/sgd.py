import math
import torch

from dawnofeve.base.base_optimizer import BaseOptimizer
from dawnofeve.base.base_optimizer import DoEConfig
from typing import Dict, Tuple, Any, Optional


__all__ = ['SGD', 'sgd']

# SGDConfig = DoEConfig()

class SGD(BaseOptimizer):
    r"""Implements SGDP algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer is used (default: None)
    
        Example:
        >>> optimizer = dawneve.optimizer.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=0.01, momentum=0, nesterov=False, defaults: Optional[Dict[str, Any]] = None):
        defaults = {} if defaults is None else defaults
        conf = DoEConfig()
        eps = conf.eps
        betas = conf.betas
        super().__init__(params, defaults, lr, momentum, nesterov, eps, betas)


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

                p.data.add_(d_p, value=-1)

        return loss