import torch
import os
import torch.nn.functional as F
from model.config import RishabhConfig
from model.transformer import Rishabh

print("[System] Loading sft_final.pt to CPU...", flush=True)
config = RishabhConfig()
model = Rishabh(config).to(torch.bfloat16)
checkpoint = torch.load('/opt/rishabh_experimental/sft_final.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
model.eval()

def encode(text): return torch.tensor([ord(c)%32000 for c in text], dtype=torch.long).unsqueeze(0)
def decode(tensor):
    val = tensor.squeeze().tolist()
    if isinstance(val, int): val = [val]
    return "".join([chr(t) for t in val if t!=0 and t<128])

prompt = "<problem>A train travels at 75 km/h for 3.5 hours. How far did the train travel in total?</problem><thinking>"

# Generate deterministic thought
print(f"PROMPT: {prompt}")
input_ids = encode(prompt)
print(f"\nGENERATING:", flush=True)
with torch.no_grad():
    for _ in range(64):
        logits, _ = model(input_ids, return_mtp=False)
        next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        print(decode(next_token), end="", flush=True)
        if '</answer>' in decode(input_ids[0, -15:]):
            break
print()
