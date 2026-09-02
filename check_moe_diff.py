import torch

ckpt200 = torch.load("/opt/rishabh_experimental/rl_ckpt_200.pt", map_location='cpu', weights_only=True)
ckpt3200 = torch.load("/opt/rishabh_experimental/rl_ckpt_3200.pt", map_location='cpu', weights_only=True)

w200 = ckpt200['model_state']['layers.0.moe.router.weight']
w3200 = ckpt3200['model_state']['layers.0.moe.router.weight']

diff = (w3200 - w200).abs()
print(f"Max change in any single routing weight: {diff.max().item():.8f}")
print(f"Mean change across all routing weights: {diff.mean().item():.8f}")
