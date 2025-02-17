import torch
from torch.optim import Optimizer

class SPSAOptimizerSimple(Optimizer):
    def __init__(self, params, lr, perturbation=0.05, alpha=0.602, gamma=0.101, a=None, c=None):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if perturbation <= 0.0:
            raise ValueError("Invalid perturbation: {}".format(perturbation))
        
        if a is None:
            a = lr
        if c is None:
            c = perturbation

        defaults = dict(lr=lr, perturbation=perturbation, alpha=alpha, gamma=gamma, a=a, c=c)
        super(SPSAOptimizerSimple, self).__init__(params, defaults)

    def step(self, closure):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            perturbation = group['perturbation']
            alpha = group['alpha']
            gamma = group['gamma']
            a = group['a']
            c = group['c']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['iteration'] = 0
                    state['gradient_estimate'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                iteration = state['iteration']
                gradient_estimate = state['gradient_estimate']

                step = iteration + 1
                a_n = a / (step + 1 + 1)**alpha
                c_n = c / (step + 1)**gamma

                perturb = torch.bernoulli(torch.full_like(p.data, 0.5)) * 2 - 1

                # Compute function values at perturbed points
                p.data.add_(c_n * perturb)
                f_plus = closure().item()

                p.data.sub_(2 * c_n * perturb)
                f_minus = closure().item()

                # Gradient estimation
                gradient_estimate.copy_((f_plus - f_minus) / (2 * c_n) * perturb)

                # Parameter update
                p.data.add_(-a_n * gradient_estimate)

                # Restore original parameter state
                p.data.add_(c_n * perturb)

                state['iteration'] = step

        return loss