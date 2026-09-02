import torch
import gc
import time
import torch.nn.functional as F
from model.config import RishabhConfig
from model.transformer import Rishabh

config = RishabhConfig()
model = Rishabh(config).to(torch.bfloat16)

ckpt_path = "/opt/rishabh_experimental/ckpt_step_15395.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True, mmap=True)
model.load_state_dict(checkpoint['model_state'])
del checkpoint
gc.collect()
torch.cuda.empty_cache()

model = model.cuda()
model.eval()

def encode_prompt(text: str) -> torch.Tensor:
    bytes_data = [ord(c) % 32000 for c in text]
    return torch.tensor(bytes_data, dtype=torch.long).unsqueeze(0).cuda()

def decode_text(tensor: torch.Tensor) -> str:
    tokens = tensor.squeeze().tolist()
    if isinstance(tokens, int): tokens = [tokens]
    return "".join([chr(t) for t in tokens if t != 0 and t < 128])

def generate(prompt: str, max_tokens=100, temperature=0.7):
    input_ids = encode_prompt(prompt)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(input_ids, return_mtp=False)
            
            # Apply temperature sampling to avoid greedy stutter loops
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(0)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if '</answer>' in decode_text(input_ids[0, -15:]):
                break
                
    return decode_text(input_ids)

problem = "<problem>1 + 1</problem>\n<thinking>\n"
print("User: " + problem.strip())
result = generate(problem, max_tokens=150)
print("AI: \n" + result.replace(problem, ''))
