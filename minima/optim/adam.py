import math
import torch
from dataclasses import dataclass

from minima.optim import BaseOptimizer
from minima.optim import DoEConfig
from typing import Dict, Tuple, Any, Optional

__all__ = ['Adam']

@dataclass
class AdamConfig(DoEConfig):
    lr : float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-16



class Adam(BaseOptimizer):
    def __init__(self, params, config: DoEConfig, defaults: Optional[Dict[str, Any]] = None):
        if not 0.0 <= config.lr:
            raise ValueError(f"Invalid learning rate: {config.lr}")
        if not 0.0 <= config.eps:
            raise ValueError(f"Invalid epsilon value: {config.eps}")
        if not 0.0 <= config.betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {config.betas[0]}")
        if not 0.0 <= config.betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1:{config.betas[1]}")
        defaults = {} if defaults is None else defaults
        super().__init__(params, config, defaults)


    def get_mv(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor):
        beta1, beta2 = group['betas']
        m, v = state['exp_avg'], state['exp_avg_sq']
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        return m, v


    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        return group['lr']



    @torch.no_grad()
    def step(self):
        loss = None
        for group in self.param_groups:
            # weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                param_state['step'] = 0
                param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                param_state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m, v = self.get_mv(param_state, group, d_p)
                param_state['step'] += 1
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                lr = self.get_lr(param_state, group)
                denominator = v.sqrt().add_(eps)
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(m, denominator, value=-step_size)

        return loss


