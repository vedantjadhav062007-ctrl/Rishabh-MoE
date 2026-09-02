import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with Decoupled Shared Key for RoPE.
    Dramatically reduces KV cache size during training and inference.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_kv_latent = config.d_kv_latent
        self.d_q_latent = config.d_q_latent
        self.d_rope_key = config.d_rope_key
        
        # KV Compression
        self.w_dkv = nn.Linear(self.d_model, self.d_kv_latent, bias=False, dtype=torch.bfloat16)
        self.w_uk = nn.Linear(self.d_kv_latent, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.w_uv = nn.Linear(self.d_kv_latent, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        
        # Q Compression
        self.w_dq = nn.Linear(self.d_model, self.d_q_latent, bias=False, dtype=torch.bfloat16)
        self.w_uq = nn.Linear(self.d_q_latent, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        
        # Decoupled Shared Key (for RoPE)
        self.w_kr = nn.Linear(self.d_model, self.d_rope_key, bias=False, dtype=torch.bfloat16)
        self.w_qr = nn.Linear(self.d_q_latent, self.n_heads * self.d_rope_key, bias=False, dtype=torch.bfloat16)
        
        # Output Projection
        self.w_o = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False, dtype=torch.bfloat16)

    def apply_rotary_emb(self, x, seq_len):
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        dtype = x.dtype
        
        # 1. Compress KV
        c_kv = self.w_dkv(x) # [B, S, d_kv_latent]
        k_c = self.w_uk(c_kv).view(B, S, self.n_heads, self.head_dim)
        v_c = self.w_uv(c_kv).view(B, S, self.n_heads, self.head_dim)
        
        # 2. Compress Q
        c_q = self.w_dq(x) # [B, S, d_q_latent]
        q_c = self.w_uq(c_q).view(B, S, self.n_heads, self.head_dim)
        
        # 3. Decoupled Shared Key for RoPE
        k_r = self.w_kr(x).view(B, S, 1, self.d_rope_key)
        q_r = self.w_qr(c_q).view(B, S, self.n_heads, self.d_rope_key)
        
        k_r = self.apply_rotary_emb(k_r, S)
        q_r = self.apply_rotary_emb(q_r, S)
        k_r = k_r.expand(-1, -1, self.n_heads, -1)
        
        # 4. Concatenate content and RoPE parts
        q = torch.cat([q_c, q_r], dim=-1) # [B, S, n_heads, head_dim + d_rope_key]
        k = torch.cat([k_c, k_r], dim=-1) # [B, S, n_heads, head_dim + d_rope_key]
        v = v_c                           # [B, S, n_heads, head_dim]
        
        # 5. Attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if q.is_cuda:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        
        # 6. Output Projection
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.head_dim)
        out = self.w_o(attn_out)
        
        return out
