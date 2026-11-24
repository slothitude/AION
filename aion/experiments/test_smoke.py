"""Smoke test for the AION transformer on CPU with a tiny config (2 layers / 64 dim).

Proves: forward pass runs, backward pass runs, and loss DECREASES over training
steps. Gate: this must pass before anything ships (owner brief).
Run:  python aion/experiments/test_smoke.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from aion.foundation.model import AIONConfig, AIONModel  # noqa: E402

TINY = AIONConfig(vocab_size=256, n_layer=2, n_embd=64, n_head=4, n_ff=128, block_size=64)


def main():
    torch.manual_seed(0)
    model = AIONModel(TINY)
    n = model.num_params()
    print(f"[smoke] tiny config params={n:,}")

    # untied head must actually differ from the embedding matrix
    assert TINY.untied_head
    assert model.lm_head.weight.data_ptr() != model.wte.weight.data_ptr(), "head is tied!"

    # 1) forward
    x = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (4, TINY.block_size, TINY.vocab_size)
    assert torch.isfinite(loss)
    print(f"[smoke] forward OK  logits={tuple(logits.shape)} loss0={loss.item():.4f}")

    # 2) backward
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    print(f"[smoke] backward OK  ({len(grads)} grad tensors, all finite)")

    # 3) loss decreases over steps
    model.zero_grad()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, TINY.vocab_size, (8, TINY.block_size))
    y = torch.randint(0, TINY.vocab_size, (8, TINY.block_size))
    first = None
    for step in range(30):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        if step in (0, 10, 29):
            print(f"[smoke] step {step:2d} loss={loss.item():.4f}")
    last = loss.item()
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
    drop = first - last
    assert drop > 0.2, f"loss drop too small: {drop:.4f}"
    print(f"[smoke] loss decreased: {first:.4f} -> {last:.4f} (drop {drop:.4f})")

    # 4) param accounting matches the real config
    from aion.foundation.model import count_parameters
    rep = count_parameters()
    assert rep["total"] == 52_207_616, f"seed param count changed: {rep['total']:,}"
    print(f"[smoke] seed param count verified: {rep['total']:,} (52.21M)")

    print("[smoke] ALL PASS")


if __name__ == "__main__":
    main()
