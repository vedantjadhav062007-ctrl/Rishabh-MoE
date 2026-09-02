import torch
import torch.nn.functional as F
import time
import os
import glob
import gc
from model.config import RishabhConfig
from model.transformer import Rishabh
from data.math_gen import generate_math_problem

def encode_prompt(text: str) -> torch.Tensor:
    bytes_data = [ord(c) % 32000 for c in text]
    return torch.tensor(bytes_data, dtype=torch.long, device='cuda').unsqueeze(0)

def decode_text(tensor: torch.Tensor) -> str:
    tokens = tensor.squeeze().tolist()
    if isinstance(tokens, int): tokens = [tokens]
    return "".join([chr(t) for t in tokens if t != 0 and t < 128])

def get_reward(generated_text: str, true_ans: str) -> float:
    import re
    match = re.search(r'<answer>(.*?)</answer>', generated_text)
    if not match: 
        return -0.5  # Heavy penalty for not using the correct XML format
    
    ans = match.group(1).strip()
    try:
        if abs(float(ans) - float(true_ans)) < 1e-4: 
            return 1.0  # Perfect math
    except:
        if str(ans).strip() == str(true_ans).strip(): 
            return 1.0
            
    return -0.1  # Formatting was right, but math was wrong

def train_grpo_step(model, optimizer, difficulty=3, group_size=4):
    prob, true_ans = generate_math_problem(difficulty)
    
    # CRITICAL: Do NOT use newlines. Phase 1 was trained strictly on adjacent tags.
    prompt = f"<problem>{prob}</problem><thinking>"
    input_ids = encode_prompt(prompt)
    
    completions, old_logprobs, rewards = [], [], []
    
    model.eval()
    with torch.no_grad():
        for _ in range(group_size):
            seq_ids = input_ids.clone()
            seq_logprobs = []
            
            # Increased to 128 tokens to give it room to actually think and output the answer
            for _ in range(128):
                logits, _ = model(seq_ids, return_mtp=False)
                # Temperature = 0.8 for exploration
                probs = F.softmax(logits[0, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).unsqueeze(0)
                
                seq_logprobs.append(torch.log(probs[next_token[0,0]] + 1e-8))
                seq_ids = torch.cat([seq_ids, next_token], dim=1)
                
                if "</answer>" in decode_text(seq_ids[0, -15:]): 
                    break
                    
            completions.append(seq_ids)
            old_logprobs.append(torch.stack(seq_logprobs))
            rewards.append(get_reward(decode_text(seq_ids), true_ans))
            
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device='cuda')
    std = rewards_tensor.std()
    
    if torch.isnan(std) or std == 0: 
        advantages = torch.zeros_like(rewards_tensor)
    else: 
        advantages = (rewards_tensor - rewards_tensor.mean()) / std
        
    model.train()
    total_loss = 0
    
    for i in range(group_size):
        seq, adv, old_lp = completions[i], advantages[i], old_logprobs[i].detach()
        logits, _ = model(seq[:, :-1], return_mtp=False)
        
        gen_len = len(old_lp)
        new_lp = F.log_softmax(logits[0, -gen_len:, :], dim=-1).gather(-1, seq[0, -gen_len:].unsqueeze(-1)).squeeze(-1)
        
        ratio = torch.exp(new_lp - old_lp)
        clip_adv = torch.clamp(ratio, 0.8, 1.2) * adv
        
        # Policy Loss + KL Divergence Penalty
        loss = -torch.min(ratio * adv, clip_adv).mean() + 0.01 * (torch.exp(old_lp) * (old_lp - new_lp)).mean()
        total_loss += loss.item()
        
        # Backward pass per sequence (gradient accumulation)
        (loss / group_size).backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss, rewards_tensor.mean().item(), prob, true_ans

def main():
    print("========================================")
    print(" PHASE 2: GRPO REINFORCEMENT LEARNING   ")
    print("========================================")
    
    config = RishabhConfig()
    model = Rishabh(config).cuda()
    
    # FIX: Correct directory path
    start_step = 1
    rl_ckpts = glob.glob("/opt/rishabh_experimental/rl_ckpt_*.pt")
    if rl_ckpts:
        latest_ckpt = max(rl_ckpts, key=os.path.getctime)
        print(f"RESUMING RL from {latest_ckpt}...", flush=True)
        # Extract step number from filename
        import re
        match = re.search(r'rl_ckpt_(\d+).pt', latest_ckpt)
        if match:
            start_step = int(match.group(1)) + 1
    else:
        ckpts = glob.glob("/opt/rishabh_experimental/rishabh_final.pt")
        latest_ckpt = max(ckpts, key=os.path.getctime)
        print(f"Loading Phase 1 Brain from {latest_ckpt}...", flush=True)
    
    # Load safely to CPU first to drop old optimizer states
    checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(checkpoint['model_state'])
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    
    model = model.cuda().to(torch.bfloat16)
    
    print("Initializing ZERO-VRAM SGD Optimizer (lr=1e-4)...", flush=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    
    print("Starting GRPO Loop!", flush=True)
    for step in range(start_step, 10000):
        t0 = time.time()
        loss, avg_reward, prob, ans = train_grpo_step(model, optimizer, difficulty=3, group_size=4)
        print(f"RL Step {step} | Loss: {loss:.4f} | Avg Reward: {avg_reward:.2f} | Time: {time.time()-t0:.1f}s | Prob: {prob[:30]}... | Ans: {ans}", flush=True)
        
        if step % 200 == 0:
            print(f"Saving RL Checkpoint at step {step}...", flush=True)
            torch.save({'model_state': model.state_dict()}, f"/opt/rishabh_experimental/rl_ckpt_{step}.pt")

if __name__ == "__main__":
    main()
