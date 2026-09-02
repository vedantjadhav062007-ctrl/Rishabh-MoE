import torch

class MuonCPUOptimizer:
    def __init__(self, model: torch.nn.Module, lr=3e-4, beta=0.95, ns_steps=5, master_weights=None):
        self.lr = lr
        self.beta = beta
        self.ns_steps = ns_steps
        
        self.master_weights = {}
        self.momentum = {}
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                if master_weights and name in master_weights:
                    self.master_weights[name] = master_weights[name]
                else:
                    self.master_weights[name] = p.data.cpu().to(torch.bfloat16)
                
                self.momentum[name] = torch.zeros_like(p.data, device='cpu', dtype=torch.float32)

    def step(self, model, lr_scale=1.0):
        current_lr = self.lr * lr_scale
        
        # Stream one parameter at a time to strictly cap CPU RAM at 19.5GB (No 13GB spike!)
        for name, p in model.named_parameters():
            if p.grad is not None:
                # 1. Pull gradient
                grad_cpu = p.grad.cpu().float()
                
                # 2. Get states
                mom = self.momentum[name]
                master = self.master_weights[name]
                
                # 3. Update momentum Nesterov
                mom.lerp_(grad_cpu, 1 - self.beta)
                
                # 4. Orthogonalization
                if len(mom.shape) >= 2:
                    G = mom.view(mom.shape[0], -1)
                    X = G / (G.norm() + 1e-8)
                    for _ in range(self.ns_steps):
                        A = X @ X.T
                        X = 3.75 * X - 3.0 * A @ X
                    update = X.view_as(master)
                else:
                    update = mom
                    
                # 5. Master weight update
                master.add_(update.to(torch.bfloat16), alpha=-current_lr)
                
                # 6. Stream back to GPU instantly
                p.data.copy_(master.cuda())
                
                # 7. Aggressively delete temporary CPU buffers for this layer
                del grad_cpu
                del update

    def get_master_weight(self, name):
        return self.master_weights[name]
