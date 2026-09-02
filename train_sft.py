import torch
import torch.nn.functional as F
import time
import glob
import os
import gc
from model.config import RishabhConfig
from model.transformer import Rishabh

def main():
    print("========================================")
    print(" PHASE 1.5: SUPERVISED FINE-TUNING (SFT)")
    print("========================================")
    
    config = RishabhConfig()
    model = Rishabh(config).cuda()
    
    # Load the latest RL checkpoint to maintain the MoE structural gains
    ckpts = glob.glob("/opt/rishabh_experimental/sft_ckpt_*.pt")
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"Loading Brain from {latest_ckpt}...", flush=True)
    
    checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint)
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    
    model = model.cuda().to(torch.bfloat16)
    
    # Standard Cross-Entropy Optimizer for SFT
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)
    
    print("Loading SFT Dataset...", flush=True)
    dataset = torch.load("/opt/rishabh_experimental/data/sft_math.pt", weights_only=True)
    
    batch_size = 2
    num_epochs = 1
    
    print("Starting SFT Logic Overwrite!", flush=True)
    model.train()
    
    step = 2500
    for epoch in range(num_epochs):
        for i in range(2500 * batch_size, len(dataset), batch_size):
            t0 = time.time()
            batch_data = dataset[i:i+batch_size]
            
            # Pad sequences in batch
            max_len = max(len(t) for t in batch_data)
            padded = torch.zeros(len(batch_data), max_len, dtype=torch.long, device='cuda')
            for j, t in enumerate(batch_data):
                padded[j, :len(t)] = t.cuda()
                
            x = padded[:, :-1]
            y = padded[:, 1:]
            
            logits, _ = model(x, return_mtp=False)
            
            # Standard autoregressive language modeling loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            if step % 10 == 0:
                print(f"SFT Step {step:04d} | Loss: {loss.item():.4f} | Time: {time.time()-t0:.2f}s", flush=True)
                
            if step % 500 == 0:
                print(f"Saving SFT Checkpoint {step}...", flush=True)
                torch.save({'model_state': model.state_dict()}, f"/opt/rishabh_experimental/sft_ckpt_{step}.pt")

    print("Saving final SFT model...")
    torch.save({'model_state': model.state_dict()}, f"/opt/rishabh_experimental/sft_final.pt")

if __name__ == "__main__":
    main()
