import torch
ckpt = torch.load('/opt/rishabh_experimental/rl_ckpt_200.pt', map_location='cpu', weights_only=True)
state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
router_keys = [k for k in state.keys() if 'router' in k or 'gate' in k]
print("Router keys found:", router_keys[:5])
if router_keys:
    w = state[router_keys[0]]
    print("Shape:", w.shape)
    print("Mean routing weights per expert:", w.abs().mean(dim=1).tolist())
