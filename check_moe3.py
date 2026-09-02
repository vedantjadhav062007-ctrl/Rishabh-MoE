import torch
ckpt = torch.load('/opt/rishabh_experimental/rl_ckpt_200.pt', map_location='cpu', weights_only=True)
state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
router_keys = [k for k in state.keys() if 'router' in k or 'gate' in k]
if router_keys:
    w = state[router_keys[0]]
    print(f"Shape: {w.shape}")
    if w.shape[0] < w.shape[1]:
        # Shape is [experts, dim]
        means = w.abs().mean(dim=1).tolist()
    else:
        # Shape is [dim, experts]
        means = w.abs().mean(dim=0).tolist()
    for i, m in enumerate(means):
        print(f"Expert {i}: {m:.6f}")
