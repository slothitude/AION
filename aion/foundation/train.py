"""DEPRECATED 2026-11-24 (owner pivot): from-scratch trainer retired — AION's foundation is now frozen Qwen3.5-4B-Base (see pretrained_loader.py); kept for reference.

Training loop for AION-Seed-Base.

AdamW, cosine schedule with warmup, bf16 on GPU (autocast) / fp32 fallback.
Data: pre-tokenized uint16 memmap (train.bin) sampled at random offsets.
Checkpoints: model + optimizer + config (+ tokenizer.json copied alongside).
Clean resume is explicitly NOT required (owner brief).
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from aion.foundation.model import AIONConfig, AIONModel  # noqa: E402


class RandomTokenDataset:
    """uint16 memmap of token ids; yields (B, T+1) int64 batches at random offsets."""

    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        assert len(self.data) > 2, f"token file too small: {path}"

    def __len__(self):
        return len(self.data)

    def batch(self, batch_size, block_size, device):
        ix = torch.randint(len(self.data) - block_size - 1, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(self.data[i + 1 : i + 1 + block_size].astype(np.int64))
                for i in ix
            ]
        )
        if device == "cuda":
            x = x.pin_memory()
            y = y.pin_memory()
            return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        return x.to(device), y.to(device)


def lr_at(step, base_lr, warmup_steps, max_steps, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def save_checkpoint(out_dir, model, optimizer, cfg, tokenizer_path, step, tokens_seen, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_last.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            **(extra or {}),
        },
        path,
    )
    # tokenizer + standalone config ride alongside the checkpoint
    if tokenizer_path and os.path.isfile(tokenizer_path):
        shutil.copyfile(tokenizer_path, os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return path


def main():
    ap = argparse.ArgumentParser(description="Train AION-Seed-Base")
    ap.add_argument("--data", required=True, help="pre-tokenized train.bin (uint16)")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json")
    ap.add_argument("--out", required=True, help="checkpoint output dir")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=0, help="overrides epochs if >0")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    elif args.dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    cfg = AIONConfig()
    model = AIONModel(cfg).to(device)
    n_params = model.num_params()
    print(f"device={device} dtype={amp_dtype} params={n_params:,} ({n_params/1e6:.2f}M)")

    data = RandomTokenDataset(args.data)
    n_tokens = len(data)
    tokens_per_step = args.batch_size * args.ctx * args.grad_accum
    max_steps = args.max_steps if args.max_steps > 0 else max(1, int(args.epochs * n_tokens / tokens_per_step))
    print(f"corpus tokens={n_tokens:,}  tokens/step={tokens_per_step:,}  max_steps={max_steps:,}  "
          f"(~{args.epochs} epoch(s))")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay
    )

    use_autocast = device == "cuda" and amp_dtype == torch.bfloat16
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"lr={args.lr} warmup={args.warmup_steps} ctx={args.ctx} B={args.batch_size} "
        f"accum={args.grad_accum} clip={args.grad_clip}")

    model.train()
    t0 = time.time()
    tokens_seen = 0
    best_loss = float("inf")

    for step in range(max_steps):
        lr = lr_at(step, args.lr, args.warmup_steps, max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = data.batch(args.batch_size, args.ctx, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                log(f"FATAL: non-finite loss at step {step}: {loss.item()}")
                sys.exit(1)
            (loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        tokens_seen += tokens_per_step

        if step % args.log_every == 0 or step == max_steps - 1:
            dt = time.time() - t0
            tps = tokens_seen / dt
            ms = dt / (step + 1) * 1000
            eta_s = (max_steps - step - 1) * (dt / (step + 1))
            eta = f"{eta_s/3600:.2f}h" if eta_s >= 3600 else f"{eta_s/60:.1f}m"
            vram = ""
            if device == "cuda":
                vram = f" vram={torch.cuda.memory_allocated()/2**30:.2f}/{torch.cuda.max_memory_allocated()/2**30:.2f}GiB"
            log(f"step {step}/{max_steps} loss={loss_acc:.4f} lr={lr:.2e} "
                f"tok/s={tps:,.0f} {ms:.0f}ms/step eta={eta}{vram}")

        if (step + 1) % args.save_every == 0 or step == max_steps - 1:
            p = save_checkpoint(args.out, model, optimizer, cfg, args.tokenizer, step + 1, tokens_seen,
                                extra={"last_loss": loss_acc})
            if loss_acc < best_loss:
                best_loss = loss_acc
            log(f"saved {p} (loss={loss_acc:.4f})")

    log(f"DONE loss={loss_acc:.4f} tokens={tokens_seen:,} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
