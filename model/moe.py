import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        # In actual FP8 implementation, these weights are stored in FP8
        # For simplicity, initialized as BF16
        self.w_gate = nn.Linear(d_in, d_hidden, bias=False, dtype=torch.bfloat16)
        self.w_up = nn.Linear(d_in, d_hidden, bias=False, dtype=torch.bfloat16)
        self.w_down = nn.Linear(d_hidden, d_in, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(xW_gate) * xW_up) W_down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class AuxLossFreeMoE(nn.Module):
    """
    Mixture of Experts layer with 1 Shared Expert and N Routed Experts.
    Uses DeepSeek-V3's Auxiliary-Loss-Free load balancing (bias routing).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Always-active shared expert
        self.shared_expert = SwiGLU(self.d_model, config.expert_ffn_dim)
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLU(self.d_model, config.expert_ffn_dim) 
            for _ in range(config.n_routed_experts)
        ])
        
        # Router
        self.router = nn.Linear(self.d_model, config.n_routed_experts, bias=False, dtype=torch.bfloat16)
        
        # DeepSeek-V3 Aux-Loss-Free Load Balancing Bias
        # Requires no gradients, updated manually after each batch
        self.register_buffer('expert_bias', torch.zeros(config.n_routed_experts))
        self.target_load = 1.0 / config.n_routed_experts
        
        # For tracking usage across forward passes within a batch
        self.token_counts = torch.zeros(config.n_routed_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        
        # 1. Routing score computation (affinity)
        affinity = self.router(x_flat) # [B*S, n_routed_experts]
        
        # 2. Add bias ONLY for selection, not for gating weights
        selection_scores = affinity + self.expert_bias.to(affinity.device)
        topk_scores, topk_indices = torch.topk(selection_scores, self.config.top_k_experts, dim=-1)
        
        # 3. Gating weights come from original unbiased affinity
        # Gather original affinity corresponding to selected experts
        selected_affinity = torch.gather(affinity, 1, topk_indices)
        gate_weights = F.softmax(selected_affinity, dim=-1)
        
        # Track usage for bias update
        with torch.no_grad():
            unique_idx, counts = torch.unique(topk_indices, return_counts=True)
            self.token_counts.to(unique_idx.device).scatter_add_(0, unique_idx, counts.float())

        # 4. Compute routed outputs (Gather/Scatter)
        out = torch.zeros_like(x_flat)
        # Process experts one by one for simplicity (In prod: use custom TileLang scatter/gather)
        for i, expert in enumerate(self.routed_experts):
            # Find tokens assigned to this expert
            mask = (topk_indices == i)
            if not mask.any():
                continue
            
            # Extract tokens and their gating weights
            token_idx, rank_idx = torch.where(mask)
            expert_inputs = x_flat[token_idx]
            expert_weights = gate_weights[token_idx, rank_idx].unsqueeze(1)
            
            # Compute and scatter back
            expert_outputs = expert(expert_inputs) * expert_weights
            out.scatter_add_(0, token_idx.unsqueeze(1).expand_as(expert_outputs), expert_outputs)

        # 5. Combine with Shared Expert
        shared_out = self.shared_expert(x_flat)
        final_out = out + shared_out
        
        return final_out.view(B, S, D)

    @torch.no_grad()
    def update_expert_bias(self, lr: float = 0.001):
        """
        Adjusts bias term based on token usage. 
        Called by the optimizer step (NOT via backprop).
        """
        total_tokens = self.token_counts.sum()
        if total_tokens == 0:
            return
            
        actual_load = self.token_counts / total_tokens
        
        # If overloaded (actual > target), decrease bias.
        # If underloaded (actual < target), increase bias.
        bias_adjustment = lr * torch.sign(self.target_load - actual_load)
        self.expert_bias += bias_adjustment.to(self.expert_bias.device)
        
        # Reset tracker
        self.token_counts.zero_()
