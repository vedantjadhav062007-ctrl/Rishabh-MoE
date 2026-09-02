import torch
import gc
from model.config import RishabhConfig
from model.transformer import Rishabh
import torch.nn.functional as F

print("Loading architecture...", flush=True)
config = RishabhConfig()
model = Rishabh(config).cpu().to(torch.bfloat16)

ckpt_path = "/opt/rishabh_experimental/ckpt_step_13539.pt"
print(f"Loading {ckpt_path} directly to CPU RAM...", flush=True)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True, mmap=True)
model.load_state_dict(checkpoint['model_state'])
del checkpoint
gc.collect()

model.eval()

def encode_prompt(text: str) -> torch.Tensor:
    bytes_data = [ord(c) % 32000 for c in text]
    return torch.tensor(bytes_data, dtype=torch.long).unsqueeze(0)

def decode_text(tensor: torch.Tensor) -> str:
    tokens = tensor.squeeze().tolist()
    if isinstance(tokens, int): tokens = [tokens]
    return "".join([chr(t) for t in tokens if t != 0 and t < 128])

def generate(prompt: str, max_tokens=150):
    input_ids = encode_prompt(prompt)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(input_ids, return_mtp=False)
            next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we hit </answer>
            if '</answer>' in decode_text(input_ids[0, -15:]):
                break
                
    return decode_text(input_ids)

problems = [
    "<problem>3 * 4</problem>\n<thinking>\n",
    "<problem>x + 5 = 15</problem>\n<thinking>\n",
    "<problem>2x = 10</problem>\n<thinking>\n"
]

print("\n--- INFERENCE TEST RESULTS ---")
for p in problems:
    print(f"\nUser: {p.strip()}")
    result = generate(p, max_tokens=150)
    print(f"AI: \n{result.replace(p, '')}")
    print("-" * 40)
