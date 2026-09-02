import torch
from model.config import RishabhConfig
from model.transformer import Rishabh

print("Loading architecture...")
config = RishabhConfig()
model = Rishabh(config)

print("Loading step 3749 checkpoint directly into CPU RAM...")
ckpt = torch.load("/opt/rishabh/ckpt_step_3749.pt", map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
model.to(torch.bfloat16)
print("Brain loaded successfully!")

def encode_text(text: str) -> torch.Tensor:
    bytes_data = [ord(c) % 32000 for c in text]
    return torch.tensor([bytes_data], dtype=torch.long)

def decode_text(tokens: torch.Tensor) -> str:
    chars = [chr(t.item()) for t in tokens[0]]
    return "".join(chars)

prompt = "<problem>What is 15 + 7?</problem>"
print(f"\nPrompting AI: {prompt}")

input_ids = encode_text(prompt)

print("Thinking...", end="", flush=True)

with torch.no_grad():
    for _ in range(50):  # Generate 50 tokens
        logits, _ = model(input_ids, return_mtp=False)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Stream output
        print(chr(next_token.item()), end="", flush=True)
        
        if chr(next_token.item()) == ">" and decode_text(input_ids).endswith("</answer>"):
            break

print("\n\nFinished generation.")
