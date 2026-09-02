import torch
import torch.nn as nn

class MTPHead(nn.Module):
    """
    Multi-Token Prediction Head.
    Predicts token t+k given the hidden state at t.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        
        # MTP uses a shallow transformer block to map hidden states
        self.norm1 = nn.LayerNorm(self.d_model, dtype=torch.bfloat16)
        
        # Linear projections acting as a fast dense block
        self.dense_proj = nn.Linear(self.d_model, self.d_model * 2, bias=False, dtype=torch.bfloat16)
        self.dense_out = nn.Linear(self.d_model * 2, self.d_model, bias=False, dtype=torch.bfloat16)
        self.norm2 = nn.LayerNorm(self.d_model, dtype=torch.bfloat16)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Shallow dense block processing
        x = self.norm1(hidden_states)
        x = torch.nn.functional.silu(self.dense_proj(x))
        x = self.dense_out(x)
        return self.norm2(hidden_states + x)

class MTPModule(nn.Module):
    """
    Manages D heads for predicting multiple future tokens.
    For Rishabh, D=3 (t+1, t+2, t+3).
    """
    def __init__(self, config):
        super().__init__()
        self.depth = config.mtp_depth
        self.heads = nn.ModuleList([
            MTPHead(config) for _ in range(self.depth)
        ])
        
    def forward(self, hidden_states: torch.Tensor, lm_head: nn.Linear) -> list[torch.Tensor]:
        """
        Takes the final hidden states of the main trunk and produces logits for future tokens.
        lm_head is passed in because we share the main LM head projection.
        """
        mtp_logits = []
        current_hidden = hidden_states
        
        for i, head in enumerate(self.heads):
            # Process through MTP head block
            current_hidden = head(current_hidden)
            # Project to vocab size using the shared LM head
            logits = lm_head(current_hidden)
            mtp_logits.append(logits)
            
        return mtp_logits
