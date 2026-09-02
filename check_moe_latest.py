import torch
import os
import glob

# Find the latest checkpoint
ckpts = glob.glob("/opt/rishabh_experimental/rl_ckpt_*.pt")
latest_ckpt = max(ckpts, key=os.path.getctime) if ckpts else "/opt/rishabh_experimental/rishabh_final.pt"
print(f"Analyzing MoE Router from: {latest_ckpt}")

ckpt = torch.load(latest_ckpt, map_location='cpu', weights_only=True)
state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
w = state['layers.0.moe.router.weight']

expert_dim = 0 if w.shape[0] < w.shape[1] else 1
means = w.abs().mean(dim=1 - expert_dim).tolist()
for i, m in enumerate(means):
    print(f"Expert {i}: {m:.6f}")
