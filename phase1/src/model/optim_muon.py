import torch
from torch.optim.optimizer import Optimizer

class Muon(Optimizer):
    """
    Simplified Muon optimizer from scratch.
    It applies Newton-Schulz orthogonalization to the momentum matrix for 2D parameters.
    For 1D parameters, it falls back to standard AdamW-style updates.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                # Weight decay
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                # If parameter is 2D, apply Newton-Schulz orthogonalization to momentum
                if len(p.shape) == 2:
                    # Newton-Schulz iteration
                    # M = buf / ||buf||_F
                    M = buf / (torch.linalg.norm(buf, 'fro') + 1e-8)
                    
                    for _ in range(ns_steps):
                        # A = M M^T
                        A = M @ M.T
                        # B = A M
                        B = A @ M
                        # M = 1.5 * M - 0.5 * B
                        M = 1.5 * M - 0.5 * B
                    
                    # Update parameters with orthogonalized momentum
                    update = M * (max(p.shape[0], p.shape[1]) ** 0.5) # Scale factor
                    p.add_(update, alpha=-lr)
                else:
                    # Fallback for 1D parameters (biases, layernorms)
                    p.add_(buf, alpha=-lr)

        return loss
