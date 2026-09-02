from dataclasses import dataclass

@dataclass
class RishabhConfig:
    """
    Master configuration for Rishabh 3.25B total / ~1B active MoE Model.
    Designed for RTX 5080 (16GB VRAM) using CPU Offloaded Muon optimizer.
    """
    name: str = "Rishabh-3.25B"
    
    # Core dimensions
    d_model: int = 1536
    n_layers: int = 32
    n_heads: int = 24
    head_dim: int = 64          # n_heads * head_dim = d_model (1536)
    vocab_size: int = 32_000
    max_seq_len: int = 2048
    
    # MLA (Multi-head Latent Attention)
    d_kv_latent: int = 160      # KV compression rank (~90% compression)
    d_q_latent: int = 640       # Q compression rank
    d_rope_key: int = 64        # Decoupled shared key for RoPE
    
    # Hybrid Attention
    full_attn_every_n: int = 4  # Full attention every 4th layer
    sliding_window: int = 512   # Sliding window size for local layers
    
    # Mixture of Experts (MoE)
    n_shared_experts: int = 1
    n_routed_experts: int = 12
    top_k_experts: int = 2
    expert_ffn_dim: int = 1536  # SwiGLU hidden dim = d_model
    
    # Multi-Token Prediction (MTP)
    mtp_depth: int = 3          # Predict t+1, t+2, t+3
    mtp_loss_weight: float = 0.3
    
    # Training & Optimizer
    muon_lr: float = 3e-4
    muon_beta: float = 0.95
    muon_ns_steps: int = 5
    batch_size: int = 8         # Upgraded to 8 for maximum GPU saturation (1024 tokens/step)
    grpo_rollouts: int = 8
    grpo_kl_beta: float = 0.04
    fp8_training: bool = True
    cpu_offload: bool = True
    
    # WSD Schedule
    warmup_steps: int = 500
    stable_fraction: float = 0.8
    lr_min_ratio: float = 0.1
    
    # Memory Management
    selective_recompute: bool = True
    kv_prune_ratio: float = 0.8
    kv_sink_tokens: int = 4
    kv_local_window: int = 256
