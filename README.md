# Rishabh-MoE: A Custom 3.25B Mixture-of-Experts (MoE) LLM

Rishabh-MoE is a custom-built 3.25 Billion parameter Language Model leveraging a **Mixture-of-Experts (MoE)** architecture, engineered completely from scratch. 

This repository contains the full end-to-end training pipeline, including Pretraining (Language & Grammar), Supervised Fine-Tuning (SFT / Chain of Thought), and **Reinforcement Learning (GRPO)**.

## 🚀 Architecture Highlights
- **Parameters:** 3.25 Billion
- **MoE Topology:** 12 Total Experts (Top-2 Routing)
- **Attention:** Multi-Head Attention (Rotary Position Embeddings)
- **Infrastructure:** PyTorch DDP via PCIe Passthrough on Proxmox LXC
- **Optimizer Stack:** 8-Bit AdamW (BitsAndBytes) + Gradient Accumulation for VRAM optimization

## 🧠 Training Pipeline

The training pipeline was executed sequentially to teach the model to reason:

1. **Phase 1: Base Pretraining (`train.py`)**
   - Trained the model to speak English and understand XML `<thinking>` tags.
   - Forged the fundamental grammatical structure across the 12 experts.

2. **Phase 1.5: Supervised Fine-Tuning (`train_sft_8bit.py`)**
   - Injected explicit **Chain of Thought (CoT)** reasoning.
   - Force-fed the AI explicit algebraic equations (e.g. `Distance = Speed * Time`) to teach it how to use the `<thinking>` tag as a mathematical scratchpad.
   - Leveraged 8-Bit AdamW quantization to shatter VRAM bottlenecks and safely run `1e-3` learning rates.

3. **Phase 2: GRPO Reinforcement Learning (`train_rl.py`)**
   - Implemented Group Relative Policy Optimization (GRPO).
   - The AI generates 4 distinct mathematical thoughts per step, exploring logical pathways. Correct paths yield a `+1.0` reward, punishing deviations, mathematically re-wiring the MoE routing probabilities toward logical truth.

## ⚙️ Hardware Limitations & "The Lazy AI"
During Phase 2, an incredible AI phenomenon occurred: **Statistical Reward Hacking**. 
Because the RL dataset clustered answers in a specific numerical range (50-80), the AI learned to perfectly match the statistical probability distribution of the dataset to farm `+1.0` rewards, rather than learning strict arithmetic. To counter this, Phase 1.5 (SFT) was successfully introduced to enforce rigid equations.

## 📂 Project Structure
- `model/` - The core PyTorch Transformer & MoE implementation
- `data/` - Synthetic math dataset generators
- `train.py` - Phase 1 Pretraining Engine
- `train_sft_8bit.py` - Phase 1.5 SFT Engine (8-Bit AdamW)
- `train_rl.py` - Phase 2 GRPO RL Engine
- `evaluator.py` & `test_*.py` - Inference & Diagnostics

## 🛡️ Setup & Usage
This engine is built to run on raw PyTorch with CUDA acceleration.
```bash
pip install torch transformers bitsandbytes paramiko
python3 train.py
```
