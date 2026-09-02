
import os

with open("/opt/rishabh_experimental/train_sft.py", "r") as f:
    content = f.read()

content = content.replace(
    "optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)",
    "optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)"
)
content = content.replace("batch_size = 4", "batch_size = 2")

with open("/opt/rishabh_experimental/train_sft.py", "w") as f:
    f.write(content)
