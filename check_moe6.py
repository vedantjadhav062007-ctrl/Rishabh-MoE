import torch
ckpt = torch.load('/opt/rishabh_experimental/rl_ckpt_200.pt', map_location='cpu', weights_only=True)
state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
w = state['layers.0.moe.router.weight']
print(f"Shape: {w.shape}")
# Expert dimension is the smaller one (likely 16)
expert_dim = 0 if w.shape[0] < w.shape[1] else 1
means = w.abs().mean(dim=1 - expert_dim).tolist()
print("Average weight magnitude per expert:")
for i, m in enumerate(means):
    print(f"Expert {i}: {m:.6f}")
