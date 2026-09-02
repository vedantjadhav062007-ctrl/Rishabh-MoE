
import os

with open("/opt/rishabh_experimental/train_rl.py", "r") as f:
    content = f.read()

# Replace the checkpoint loading logic to resume from the latest rl_ckpt
old_logic = '''    ckpts = glob.glob("/opt/rishabh_experimental/rishabh_final.pt")
    if not ckpts: ckpts = glob.glob("/opt/rishabh_experimental/ckpt_step_*.pt")
    latest_ckpt = max(ckpts, key=os.path.getctime)
    
    print(f"Loading Phase 1 Brain via MMAP from {latest_ckpt}...", flush=True)'''

new_logic = '''    start_step = 1
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
        print(f"Loading Phase 1 Brain from {latest_ckpt}...", flush=True)'''

content = content.replace(old_logic, new_logic)

# Replace the for loop
content = content.replace("for step in range(1, 10000):", "for step in range(start_step, 10000):")

with open("/opt/rishabh_experimental/train_rl.py", "w") as f:
    f.write(content)
