import torch
ckpt = torch.load('/opt/rishabh_experimental/rl_ckpt_200.pt', map_location='cpu', weights_only=True)
state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
router_keys = [k for k in state.keys() if 'router' in k or 'gate' in k]
if router_keys:
    print(f"Shape of {router_keys[0]}: {state[router_keys[0]].shape}")
