
import os

with open("/opt/rishabh_experimental/train_sft_8bit.py", "r") as f:
    content = f.read()

content = content.replace("import torch", 'import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch')

# Gradient accumulation
content = content.replace("batch_size = 4", "batch_size = 1
    grad_accum_steps = 4")

content = content.replace(
    "loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0)",
    "loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=0) / grad_accum_steps"
)

content = content.replace(
    "optimizer.step()
            optimizer.zero_grad()",
    "if (i // batch_size + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()"
)

with open("/opt/rishabh_experimental/train_sft_8bit.py", "w") as f:
    f.write(content)
