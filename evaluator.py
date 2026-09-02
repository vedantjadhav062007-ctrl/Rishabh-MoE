import torch
import sys
import glob
import time
from model.config import RishabhConfig
from model.transformer import Rishabh
from data.math_gen import generate_math_problem
from training.verifier import extract_answer, verify_math_answer

def encode_prompt(text: str) -> torch.Tensor:
    bytes_data = [ord(c) % 32000 for c in text]
    return torch.tensor(bytes_data, dtype=torch.long).unsqueeze(0).cuda()

def decode_text(tensor: torch.Tensor) -> str:
    tokens = tensor.squeeze().tolist()
    if isinstance(tokens, int):
        tokens = [tokens]
    return "".join([chr(t) for t in tokens if t != 0 and t < 128])

@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int = 256):
    model.eval()
    input_ids = encode_prompt(prompt)
    
    for _ in range(max_new_tokens):
        # We only care about the main output (logits), not MTP heads for generation
        logits, _ = model(input_ids, return_mtp=False)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Simple stop condition: if it generates the closing answer tag
        if decode_text(input_ids).endswith("</answer>"):
            break
            
    return decode_text(input_ids)

def run_evaluation(num_problems=20, difficulty=2):
    print("========================================")
    print(" RISHABH TRUE ACCURACY EVALUATOR        ")
    print("========================================")
    
    # 1. Find latest checkpoint
    ckpts = glob.glob("/opt/rishabh/ckpt_step_*.pt")
    if not ckpts:
        ckpts = glob.glob("/opt/rishabh/rishabh_final.pt")
    if not ckpts:
        print("No checkpoints found! Cannot run evaluation.")
        return
        
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"Loading weights from {latest_ckpt}...")
    
    config = RishabhConfig()
    model = Rishabh(config).cuda()
    checkpoint = torch.load(latest_ckpt, map_location="cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Weights loaded successfully.")
    
    correct = 0
    print(f"\nEvaluating {num_problems} problems at Level {difficulty}...")
    
    start_time = time.time()
    for i in range(num_problems):
        prob, true_ans = generate_math_problem(difficulty)
        prompt = f"<problem>{prob}</problem>\n"
        
        generated_full = generate(model, prompt, max_new_tokens=256)
        extracted_ans = extract_answer(generated_full)
        
        is_correct = verify_math_answer(extracted_ans, true_ans)
        if is_correct:
            correct += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"
            
        print(f"[{i+1}/{num_problems}] {status} | True: {true_ans} | Extracted: {extracted_ans}")
        if not is_correct:
            print(f"   Generated: {generated_full.strip()}")
            
    acc = (correct / num_problems) * 100
    print("\n========================================")
    print(f" FINAL ACCURACY: {acc:.1f}% ({correct}/{num_problems})")
    print(f" TIME TAKEN: {time.time() - start_time:.1f}s")
    print("========================================")

if __name__ == "__main__":
    import os
    # Default to Level 3 since we are training Level 3 now
    run_evaluation(num_problems=10, difficulty=3)