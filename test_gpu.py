import torch
import gc
import time
from model.config import RishabhConfig
from model.transformer import Rishabh

print("Initializing Model Architecture...", flush=True)
config = RishabhConfig()
model = Rishabh(config).to(torch.bfloat16)

ckpt_path = "/opt/rishabh_experimental/ckpt_step_15395.pt"
print(f"Loading {ckpt_path} to CPU RAM first (to prevent Optimizer VRAM OOM)...", flush=True)
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True, mmap=True)

print("Extracting weights and deleting optimizer states...", flush=True)
model.load_state_dict(checkpoint['model_state'])
del checkpoint
gc.collect()
torch.cuda.empty_cache()

print("Transferring pure AI brain to RTX 5080 GPU...", flush=True)
model = model.cuda()
model.eval()

def encode_prompt(text: str) -> torch.Tensor:
    bytes_data = [ord(c) % 32000 for c in text]
    return torch.tensor(bytes_data, dtype=torch.long).unsqueeze(0).cuda()

def decode_text(tensor: torch.Tensor) -> str:
    tokens = tensor.squeeze().tolist()
    if isinstance(tokens, int): tokens = [tokens]
    return "".join([chr(t) for t in tokens if t != 0 and t < 128])

def generate(prompt: str, max_tokens=150):
    input_ids = encode_prompt(prompt)
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(input_ids, return_mtp=False)
            next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if '</answer>' in decode_text(input_ids[0, -15:]):
                break
    t1 = time.time()
    gen_time = t1 - t0
    tok_sec = (input_ids.shape[1] - len(prompt)) / gen_time if gen_time > 0 else 0
    
    return decode_text(input_ids), tok_sec

problems = [
    "<problem>3 * 4</problem>\n<thinking>\n",
    "<problem>x + 5 = 15</problem>\n<thinking>\n",
    "<problem>2x = 10</problem>\n<thinking>\n"
]

print("\n=== GPU INFERENCE TEST RESULTS ===")
for p in problems:
    print(f"\nUser: {p.strip()}")
    result, speed = generate(p, max_tokens=150)
    print(f"AI: \n{result.replace(p, '')}")
    print(f"[Speed: {speed:.1f} tokens/sec]")
    print("-" * 50)
