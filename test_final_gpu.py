import torch
import gc
from model.config import RishabhConfig
from model.transformer import Rishabh

print("[System] Loading architecture...", flush=True)
config = RishabhConfig()
model = Rishabh(config).to(torch.bfloat16)

ckpt_path = "/opt/rishabh_experimental/rishabh_final.pt"
print("[System] Loading rishabh_final.pt to CPU first (to avoid OOM)...", flush=True)

# Load to CPU to strip out the 6.5GB optimizer states
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True, mmap=True)
model.load_state_dict(checkpoint['model_state'])
del checkpoint
gc.collect()

print("[System] Moving stripped model to GPU...", flush=True)
model = model.cuda()
model.eval()

def encode(text): return torch.tensor([ord(c)%32000 for c in text], dtype=torch.long, device='cuda').unsqueeze(0)
def decode(tensor):
    val = tensor.squeeze().tolist()
    if isinstance(val, int): val = [val]
    return "".join([chr(t) for t in val if t!=0 and t<128])

def generate(prompt: str, max_tokens=60):
    input_ids = encode(prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(input_ids, return_mtp=False)
            next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if '</answer>' in decode(input_ids[0, -15:]):
                break
    return decode(input_ids)

problems = [
    "A train travels at 80 km/h for 3 hours. How far did the train travel in total?",
    "A farmer harvests 250 apples and sells 125 of them. How many does he have left?",
    "If a car drives 120 miles in 2 hours, what is its average speed in miles per hour?",
    "Sarah buys 4 books for $15 each. How much money did she spend in total?",
    "A baker makes 500 cookies. He packages them in boxes of 10. How many boxes does he need?",
    "What is the result of 45 + 15?",
    "A runner completes a 10 km race in 50 minutes. What is her speed in km per minute?",
    "If you have 100 dollars and buy a video game for 60 dollars, how much is left?",
    "A rectangle has a length of 5 and a width of 4. What is its area?",
    "If 3x = 12, what is the value of x?"
]

print("\n" + "="*50)
print(" RISHABH PHASE 1 FINAL EXAM (GPU ACCELERATED)")
print("="*50)

for i, p in enumerate(problems):
    prompt = f"<problem>{p}</problem><thinking>"
    result = generate(prompt, max_tokens=100)
    clean_out = result.replace(prompt, '')
    print(f"\n[Q{i+1}] {p}")
    print(f"Rishabh: {clean_out}")
