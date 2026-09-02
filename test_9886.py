import torch
import gc
from model.config import RishabhConfig
from model.transformer import Rishabh

print("Loading architecture...", flush=True)
config = RishabhConfig()
model = Rishabh(config)

print("Loading Step 9886 directly to CPU RAM...", flush=True)
ckpt = torch.load("/opt/rishabh_experimental/ckpt_step_9886.pt", map_location='cpu')
model.load_state_dict(ckpt['model_state'])
del ckpt
gc.collect()

model.eval()
model.to(torch.bfloat16)

def encode_text(text):
    bytes_data = [ord(c) % 32000 for c in text]
    return torch.tensor([bytes_data], dtype=torch.long)

def decode_text(tokens):
    return "".join([chr(t.item()) for t in tokens[0]])

def test_ai(prompt_text):
    prompt = f"<problem>{prompt_text}</problem>"
    print(f"\nUser: {prompt}", flush=True)
    input_ids = encode_text(prompt)
    print("AI: ", end="", flush=True)
    with torch.no_grad():
        for _ in range(80):
            logits, _ = model(input_ids, return_mtp=False)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            char = chr(next_token.item())
            print(char, end="", flush=True)
            if char == ">" and decode_text(input_ids).endswith("</answer>"):
                break
    print("", flush=True)

test_ai("15 + 7")
test_ai("10 * 4")
test_ai("x + 5 = 12")
