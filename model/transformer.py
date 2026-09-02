import torch
import torch.nn as nn
from liger_kernel.ops.layer_norm import LigerLayerNormFunction
from model.attention import MultiHeadLatentAttention
from model.moe import AuxLossFreeMoE
from model.mtp import MTPModule
import torch.utils.checkpoint as checkpoint

class RishabhBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = MultiHeadLatentAttention(config)
        self.moe = AuxLossFreeMoE(config)
        
        self.norm1 = nn.LayerNorm(config.d_model, dtype=torch.bfloat16)
        self.norm2 = nn.LayerNorm(config.d_model, dtype=torch.bfloat16)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture
        h = x + self.attn(self.norm1(x))
        out = h + self.moe(self.norm2(h))
        return out

class Rishabh(nn.Module):
    """
    Rishabh 3.25B MoE Transformer with CPU Offloading support.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model, dtype=torch.bfloat16)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            RishabhBlock(config, i) for i in range(config.n_layers)
        ])
        
        # Final Norm
        self.final_norm = nn.LayerNorm(config.d_model, dtype=torch.bfloat16)
        
        # Language Model Head (Tied to embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.embed.weight
        
        # Multi-Token Prediction Module
        self.mtp = MTPModule(config)
        
    def generate_causal_mask(self, seq_len: int, device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0) # [1, 1, S, S]

    def forward(self, input_ids: torch.Tensor, return_mtp: bool = True):
        B, S = input_ids.shape
        device = input_ids.device
        
        x = self.embed(input_ids)
        mask = self.generate_causal_mask(S, device)
        
        # Forward pass through layers with selective recomputation
        for layer in self.layers:
            if self.config.selective_recompute and self.training:
                # Gradient Checkpointing (Activation Recomputation)
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
        x = self.final_norm(x)
        
        # Main Next Token Prediction
        logits = self.lm_head(x)
        
        if return_mtp and self.training:
            mtp_logits = self.mtp(x, self.lm_head)
            return logits, mtp_logits
            
        return logits, None
        
    def count_active_params(self):
        """Returns the number of active parameters per token (approx 1B)."""
        # Embeddings + Attention + Shared Expert + 2 Routed Experts + Norms
        active = sum(p.numel() for p in self.embed.parameters())
        active += sum(p.numel() for p in self.layers[0].attn.parameters()) * self.config.n_layers
        active += sum(p.numel() for p in self.layers[0].moe.shared_expert.parameters()) * self.config.n_layers
        # Top-2 experts active out of 12
        routed_expert_params = sum(p.numel() for p in self.layers[0].moe.routed_experts[0].parameters())
        active += routed_expert_params * 2 * self.config.n_layers
        active += sum(p.numel() for p in self.mtp.parameters())
        return active
