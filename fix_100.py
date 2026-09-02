
import os

with open("/opt/rishabh_experimental/train_sft_8bit.py", "r") as f:
    content = f.read()

content = content.replace("if step == 200:", "if step == 100:")

with open("/opt/rishabh_experimental/train_sft_8bit.py", "w") as f:
    f.write(content)
