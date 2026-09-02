import torch
from model.config import RishabhConfig
from model.transformer import Rishabh

config = RishabhConfig()
model = Rishabh(config)
ckpt = torch.load("/opt/rishabh/ckpt_interrupted_step_4703.pt", map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
model.to(torch.bfloat16)

def encode_text(text: str) -> torch.Tensor:
    bytes_data = [ord(c) % 32000 for c in text]
    return torch.tensor([bytes_data], dtype=torch.long)

prompt = "<problem>1+1</problem>"
input_ids = encode_text(prompt)

print(f"\nPrompting AI: {prompt}")
print("Thinking...", end="", flush=True)

with torch.no_grad():
    for _ in range(50):
        logits, _ = model(input_ids, return_mtp=False)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        print(chr(next_token.item()), end="", flush=True)
        if chr(next_token.item()) == ">" and "".join([chr(t.item()) for t in input_ids[0]]).endswith("</answer>"):
            break
print("\nFinished.")
