import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
import time
import gc
import bitsandbytes as bnb
from model.config import RishabhConfig
from model.transformer import Rishabh

def main():
    print("========================================")
    print(" PHASE 1.5: 8-BIT SFT (ADAMW)           ")
    print("========================================")
    
    config = RishabhConfig()
    model = Rishabh(config).cuda()
    
    base_ckpt = "/opt/rishabh_experimental/rishabh_final.pt"
    print(f"Loading Base Phase 1 Brain from {base_ckpt}...", flush=True)
    
    checkpoint = torch.load(base_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    
    model = model.cuda().to(torch.bfloat16)
    
    print("Initializing 8-Bit AdamW Optimizer (lr=1e-3)...", flush=True)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    print("Loading SFT Dataset...", flush=True)
    dataset = torch.load("/opt/rishabh_experimental/data/sft_math.pt", weights_only=True)
    
    batch_size = 1
    grad_accum_steps = 4
    num_epochs = 1
    
    print("Starting SFT Logic Overwrite!", flush=True)
    model.train()
    
    step = 0
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        for i in range(0, len(dataset), batch_size):
            t0 = time.time()
            batch_data = dataset[i:i+batch_size]
            
            max_len = max(len(t) for t in batch_data)
            padded = torch.zeros(len(batch_data), max_len, dtype=torch.long, device='cuda')
            for j, t in enumerate(batch_data):
                padded[j, :len(t)] = t.cuda()
                
            x = padded[:, :-1]
            y = padded[:, 1:]
            
            logits, _ = model(x, return_mtp=False)
            
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)
            loss = loss / grad_accum_steps
            
            loss.backward()
            
            if (i // batch_size + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                if step % 10 == 0:
                    print(f"8-Bit SFT Step {step:04d} | Loss: {loss.item() * grad_accum_steps:.4f} | Time: {time.time()-t0:.2f}s", flush=True)
                
                if step == 100:
                    print("Stopping early after 200 steps to check if loss dropped!", flush=True)
                    torch.save({'model_state': model.state_dict()}, "/opt/rishabh_experimental/sft_8bit_test.pt")
                    return

if __name__ == "__main__":
    main()
