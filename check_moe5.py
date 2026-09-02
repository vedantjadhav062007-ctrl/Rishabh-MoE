import torch
ckpt = torch.load('/opt/rishabh_experimental/rl_ckpt_200.pt', map_location='cpu', weights_only=True)
state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
router_keys = [k for k in state.keys() if 'router' in k]
print(router_keys)
