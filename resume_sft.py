
import os

with open("/opt/rishabh_experimental/train_sft.py", "r") as f:
    content = f.read()

content = content.replace(
    'ckpts = glob.glob("/opt/rishabh_experimental/rl_ckpt_*.pt")',
    'ckpts = glob.glob("/opt/rishabh_experimental/sft_ckpt_*.pt")'
)

# Replace the loop to start at step 2500
content = content.replace('step = 0', 'step = 2500')
content = content.replace(
    'for i in range(0, len(dataset), batch_size):',
    'for i in range(2500 * batch_size, len(dataset), batch_size):'
)

with open("/opt/rishabh_experimental/train_sft.py", "w") as f:
    f.write(content)
