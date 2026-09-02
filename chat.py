import torch
import gc
import sys
import warnings
warnings.filterwarnings('ignore')

from model.config import RishabhConfig
from model.transformer import Rishabh

print("\n[System] Booting Rishabh 3.25B...")
print("[System] Running in CPU Mode (so we don't crash your active GPU training!)")
print("[System] Loading weights from ckpt_step_28943.pt (This takes about 20 seconds)...\n")

config = RishabhConfig()
model = Rishabh(config).cpu().to(torch.bfloat16)

ckpt = "/opt/rishabh_experimental/ckpt_step_28943.pt"
checkpoint = torch.load(ckpt, map_location='cpu', weights_only=True, mmap=True)
model.load_state_dict(checkpoint['model_state'])
del checkpoint
gc.collect()

model.eval()

def encode(text): return torch.tensor([ord(c)%32000 for c in text], dtype=torch.long).unsqueeze(0)
def decode(tensor): 
    val = tensor.squeeze().tolist()
    if isinstance(val, int):
        val = [val]
    return "".join([chr(t) for t in val if t!=0 and t<128])

print("="*60)
print(" RISHABH INTERACTIVE TERMINAL (Phase 1: Pretraining)")
print(" Loaded Checkpoint: 28943")
print(" Type 'exit' to quit.")
print("="*60)

while True:
    try:
        user_msg = input("\nUser: ")
        if user_msg.lower() == 'exit': break
        if not user_msg.strip(): continue
        
        prompt = f"<problem>{user_msg}</problem><thinking>"
        input_ids = encode(prompt)
        
        print("Rishabh: ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(100):
                logits, _ = model(input_ids, return_mtp=False)
                next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                char = decode(next_token)
                print(char, end="", flush=True)
                
                if '</answer>' in decode(input_ids[0, -15:]):
                    break
        print("\n")
    except KeyboardInterrupt:
        break
