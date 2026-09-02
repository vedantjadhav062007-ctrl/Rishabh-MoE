import torch
import gc
from model.config import RishabhConfig
from model.transformer import Rishabh

print("Loading architecture...", flush=True)
config = RishabhConfig()
model = Rishabh(config).cpu().to(torch.bfloat16)

ckpt_path = "/opt/rishabh_experimental/ckpt_step_38622.pt"
print(f"Loading {ckpt_path} to CPU RAM...", flush=True)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True, mmap=True)
model.load_state_dict(checkpoint['model_state'])
del checkpoint
gc.collect()

model.eval()

def encode(text): return torch.tensor([ord(c)%32000 for c in text], dtype=torch.long).unsqueeze(0)
def decode(tensor):
    val = tensor.squeeze().tolist()
    if isinstance(val, int): val = [val]
    return "".join([chr(t) for t in val if t!=0 and t<128])

def generate(prompt: str, max_tokens=100):
    input_ids = encode(prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(input_ids, return_mtp=False)
            next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if '</answer>' in decode(input_ids[0, -15:]):
                break
    return decode(input_ids)

problem = "<problem>A store sells 15 shirts for $20 each and 5 hats for $10 each. What is the total revenue?</problem><thinking>"

print("\n--- CPU INFERENCE TEST ---")
print(f"User: {problem}")
result = generate(problem, max_tokens=60)
print(f"AI: {result.replace(problem, '')}")
