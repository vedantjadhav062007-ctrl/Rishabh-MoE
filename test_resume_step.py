import os
import gc
import torch
from model.config import RishabhConfig
from model.transformer import Rishabh
from training.optimizer import MuonCPUOptimizer
from training.scheduler import WSDSchedule
from data.math_gen import generate_math_problem

print("1. Initializing Model on CUDA...")
config = RishabhConfig()
model = Rishabh(config).cuda()

print("2. Initializing Muon...")
optimizer = MuonCPUOptimizer(model)

print("3. Loading checkpoint rishabh_final.pt...")
ckpt = torch.load('rishabh_final.pt', map_location='cpu')
start_step = ckpt.get('step', 0)
difficulty = ckpt.get('difficulty', 1)
print(f"Step: {start_step}, Difficulty: {difficulty}")

model.load_state_dict(ckpt['model_state'])
optimizer.master_weights = ckpt['optimizer_master']
del ckpt
gc.collect()
print("Resumed cleanly! Running test step...")

# Test 1 forward step
p, a = generate_math_problem(difficulty=2)
print("Sample Level 2 Problem:", p, "-> Answer:", a)

input_ids = torch.randint(1, 1000, (8, 128), device='cuda')
logits, mtp_logits = model(input_ids, return_mtp=True)
print("Forward pass successful! Logits shape:", logits.shape)
