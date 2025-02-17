import torch
from torch.optim import Optimizer

class KieferWolfowitzOptimizerSimple(Optimizer):
    def __init__(self, params, lr=None, perturbation=0.05):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if perturbation < 0.0:
            raise ValueError("Invalid perturbation: {}".format(perturbation))
        
        if lr is None:
            raise ValueError("Learning rate is required")
        
        defaults = dict(lr=lr, perturbation=perturbation)
        super(KieferWolfowitzOptimizerSimple, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            perturbation = group['perturbation']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['iteration'] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
                    state['gradient_estimate'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                iteration = state['iteration']
                gradient_estimate = state['gradient_estimate']

                step = iteration + 1
                c_n = perturbation  # Fixed perturbation
                a_n = lr  # Fixed step size

                original_data = p.data.clone()

                # Compute gradient estimates by perturbing one element at a time
                for i in range(p.data.numel()):
                    perturb = torch.zeros_like(p.data)
                    perturb.view(-1)[i] = c_n
                    
                    p.data = original_data + perturb
                    f_plus = closure().item()

                    p.data = original_data - perturb
                    f_minus = closure().item()

                    gradient_estimate.view(-1)[i] = (f_plus - f_minus) / (2 * c_n)

                p.data = original_data  # Restore original data

                p.data.add_(-a_n * gradient_estimate)

                state['iteration'] = step

        return loss