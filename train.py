import os
import gc
import time
import argparse
import logging
import torch
import torch.nn.functional as F
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from model.config import RishabhConfig
from model.transformer import Rishabh
from data.math_gen import generate_math_problem, generate_contrastive_problem
from training.scheduler import WSDSchedule
import bitsandbytes as bnb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Rishabh.Train")

PAUSE_TEMP_C = 85
RESUME_TEMP_C = 72

import threading
import subprocess

global_gpu_temp = 45

def _temp_monitor_loop():
    global global_gpu_temp
    while True:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'], 
                                    capture_output=True, text=True)
            global_gpu_temp = int(result.stdout.strip())
        except Exception:
            pass
        time.sleep(5)

temp_thread = threading.Thread(target=_temp_monitor_loop, daemon=True)
temp_thread.start()

def check_gpu_temperature():
    return global_gpu_temp

def encode_text(text: str, max_len: int = 128) -> torch.Tensor:
    bytes_data = [ord(c) % 32000 for c in text]
    if len(bytes_data) < max_len:
        bytes_data = bytes_data + [0] * (max_len - len(bytes_data))
    else:
        bytes_data = bytes_data[:max_len]
    return torch.tensor(bytes_data, dtype=torch.long)

import multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader

class ProceduralMathDataset(IterableDataset):
    def __init__(self, seq_len, contrastive, shared_difficulty):
        self.seq_len = seq_len
        self.contrastive = contrastive
        self.shared_difficulty = shared_difficulty

    def __iter__(self):
        while True:
            curr_diff = self.shared_difficulty.value
            if self.contrastive and curr_diff >= 2 and torch.rand(1).item() < 0.35:
                _, full_text = generate_contrastive_problem()
            else:
                p, a = generate_math_problem(difficulty=curr_diff)
                full_text = f"<problem>{p}</problem><thinking>Step-by-step logic</thinking><answer>{a}</answer>"
            yield encode_text(full_text, max_len=self.seq_len)

def save_checkpoint(path, step, model, optimizer, difficulty):
    logger.info(f"Saving checkpoint to {path} at step {step}...")
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': getattr(optimizer, 'state_dict', lambda: {})(),
        'difficulty': difficulty,
    }, path)
    logger.info(f"Checkpoint {path} saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Rishabh 3.25B MoE Trainer")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint")
    parser.add_argument('--total_steps', type=int, default=70000, help="Total training steps")
    parser.add_argument('--ckpt_interval', type=int, default=1800, help="Checkpoint interval in seconds (30m)")
    parser.add_argument('--seq_len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--contrastive', action='store_true', default=True, help='Enable contrastive reasoning')
    args = parser.parse_args()

    logger.info("Initializing Rishabh 3.25B MoE Architecture on CUDA...")
    config = RishabhConfig()
    config.max_seq_len = args.seq_len
    config.batch_size = args.batch_size
    model = Rishabh(config).cuda()
    
    total_p = sum(p.numel() for p in model.parameters())
    active_p = model.count_active_params()
    logger.info(f"Model Loaded: Total {total_p/1e9:.2f}B | Active per token: {active_p/1e9:.2f}B")

    start_step = 0
    difficulty = 1
    saved_optim = None

    if args.resume and os.path.exists(args.resume):
        logger.info(f"Loading checkpoint {args.resume} sequentially...")
        ckpt = torch.load(args.resume, map_location='cpu')
        start_step = ckpt.get('step', 0)
        difficulty = ckpt.get('difficulty', 1)
        model.load_state_dict(ckpt['model_state'])
        saved_optim = None
        del ckpt
        gc.collect()
        logger.info(f"Weights transferred to GPU. Resuming at step {start_step}, difficulty {difficulty}")

    torch.backends.cudnn.benchmark = True
    logger.info("Initializing bitsandbytes Paged 8-bit AdamW (GPU)...")
    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=1e-4, weight_decay=0.01)
    if saved_optim:
        try:
            optimizer.load_state_dict(saved_optim)
            logger.info("Optimizer state loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")

    scheduler = WSDSchedule(total_steps=args.total_steps)

    logger.info("Initializing Single-Threaded DataLoader (No Ram Leaks!)...")
    shared_difficulty = mp.Value('i', difficulty)
    dataset = ProceduralMathDataset(args.seq_len, args.contrastive, shared_difficulty)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=0,           
        pin_memory=True,         
        drop_last=True
    )
    batch_gen = iter(dataloader)

    logger.info(f"=== Starting Rishabh Training: Step {start_step} -> {args.total_steps} ===")
    last_ckpt_time = time.time()
    step_start_time = time.time()

    try:
        for step in range(start_step, args.total_steps):
            if step >= 8000 and difficulty == 1:
                difficulty = 2 
                shared_difficulty.value = difficulty
                logger.info(">>> CURRICULUM UPGRADED: Entering Level 2 (Algebraic Equations) <<<")
            elif step >= 18000 and difficulty == 2:
                difficulty = 3 
                shared_difficulty.value = difficulty
                logger.info(">>> CURRICULUM UPGRADED: Entering Level 3 (Multi-step Reasoning) <<<")

            if step % 25 == 0:
                temp = check_gpu_temperature()
                if temp >= PAUSE_TEMP_C:
                    logger.warning(f"GPU Temp {temp}C >= {PAUSE_TEMP_C}C! Pausing...")
                    save_checkpoint(f"ckpt_thermal_pause_step_{step}.pt", step, model, optimizer, difficulty)
                    while check_gpu_temperature() > RESUME_TEMP_C:
                        time.sleep(30)
                    logger.info("GPU cooled down. Resuming training...")

            input_ids = next(batch_gen).cuda(non_blocking=True)
            lr_scale = scheduler.get_lr_scale(step)

            model.zero_grad(set_to_none=True)
            logits, mtp_logits = model(input_ids, return_mtp=True)

            targets = input_ids[:, 1:].contiguous()
            preds = logits[:, :-1, :].contiguous()
            loss_main = F.cross_entropy(preds.view(-1, config.vocab_size), targets.view(-1), ignore_index=0)

            loss_mtp = torch.tensor(0.0, device='cuda')
            if mtp_logits is not None:
                for k, head_logits in enumerate(mtp_logits):
                    future_offset = k + 2
                    if input_ids.shape[1] > future_offset:
                        mtp_targets = input_ids[:, future_offset:].contiguous()
                        mtp_preds = head_logits[:, :-future_offset, :].contiguous()
                        loss_mtp = loss_mtp + F.cross_entropy(
                            mtp_preds.view(-1, config.vocab_size), 
                            mtp_targets.view(-1), 
                            ignore_index=0
                        ) * config.mtp_loss_weight
            total_loss = loss_main + loss_mtp

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4 * lr_scale
            optimizer.step()
            model.zero_grad(set_to_none=True)

            if step % 5 == 0:
                for layer in model.layers:
                    layer.moe.update_expert_bias(lr=0.001)

            if step % 50 == 0 or step == start_step:
                step_elapsed = (time.time() - step_start_time) / (50 if step > start_step else 1)
                tokens_per_sec = (config.batch_size * args.seq_len) / (step_elapsed + 1e-6)
                temp = check_gpu_temperature()
                vram_gb = torch.cuda.max_memory_allocated() / 1e9
                logger.info(
                    f"Step {step:05d}/{args.total_steps} | "
                    f"Loss: {total_loss.item():.4f} (Main: {loss_main.item():.4f}, MTP: {loss_mtp.item():.4f}) | "
                    f"Speed: {tokens_per_sec:.0f} tok/s ({step_elapsed*1000:.1f}ms/step) | "
                    f"VRAM: {vram_gb:.2f}GB | "
                    f"GPU: {temp}C | "
                    f"Diff: {difficulty}"
                )
                step_start_time = time.time()

            if time.time() - last_ckpt_time >= args.ckpt_interval:
                save_checkpoint(f"ckpt_step_{step}.pt", step, model, optimizer, difficulty)
                last_ckpt_time = time.time()

    except KeyboardInterrupt:
        logger.info("\nGraceful 3-hour session completion signal received! Writing safe checkpoint to SSD...")
        save_checkpoint(f"ckpt_interrupted_step_{step}.pt", step, model, optimizer, difficulty)
        logger.info("Storage protection complete. Safe to idle.")
        return
    logger.info("Training complete! Saving final model...")
    save_checkpoint("rishabh_final.pt", args.total_steps, model, optimizer, difficulty)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
