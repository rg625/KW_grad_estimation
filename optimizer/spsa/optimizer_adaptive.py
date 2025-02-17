from typing import Tuple, Optional, Union
import torch
from torch.optim import Optimizer

class SPSAAdaptive(Optimizer):
    def __init__(self, 
                 params, 
                 lr: Union[float, torch.Tensor], 
                 perturbation: float = 0.05, 
                 betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, 
                 foreach: Optional[bool] = None, 
                 maximize: bool = False, 
                 capturable: bool = False, 
                 differentiable: bool = False):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= perturbation:
            raise ValueError(f"Invalid perturbation: {perturbation}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, perturbation=perturbation, betas=betas, eps=eps, 
                        maximize=maximize, foreach=foreach, capturable=capturable, differentiable=differentiable)
        super(SPSAAdaptive, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            perturbation = group['perturbation']
            beta1, beta2 = group['betas']
            eps = group['eps']
            foreach = group['foreach']
            maximize = group['maximize']
            capturable = group['capturable']
            differentiable = group['differentiable']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state['m'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['gradient_estimate'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                step = state['step']
                m = state['m']
                v = state['v']
                gradient_estimate = state['gradient_estimate']

                step += 1

                c_n = perturbation / step.sqrt()  # Decaying perturbation
                a_n = lr / step.sqrt()  # Adjusted decay rate for step size

                original_data = p.data.clone()

                # Compute gradient estimates using simultaneous perturbation
                perturb = 2 * torch.bernoulli(torch.full_like(p.data, 0.5)) - 1

                p.data.add_(c_n * perturb)
                f_plus = closure().item()

                p.data.sub_(2 * c_n * perturb)
                f_minus = closure().item()

                gradient_estimate.copy_((f_plus - f_minus) / (2 * c_n) * perturb)

                p.data = original_data  # Restore original data

                # Update biased first moment estimate
                m.mul_(beta1).add_(gradient_estimate, alpha=1 - beta1)

                # Update biased second raw moment estimate
                v.mul_(beta2).addcmul_(gradient_estimate, gradient_estimate, value=1 - beta2)

                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - beta1 ** step)

                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1 - beta2 ** step)

                p.data.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-a_n)

                state['step'] = step

        return loss