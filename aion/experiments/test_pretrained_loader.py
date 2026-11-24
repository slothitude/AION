"""Regression test: aion.foundation.pretrained_loader.load_frozen_base (PUBLIC path).

Lane-G bug (fixed 2026-11-25): FrozenBase was constructed before its required
`model` field existed, so the documented loading path ALWAYS raised TypeError.
This test exercises the public loader end-to-end and asserts the freeze
evidence the relational smoke proved (frozen-param count + fingerprint).

Default: tiny stand-in (CPU-fast, runs anywhere) — a small Llama is saved with
save_pretrained and loaded through the SAME public from_pretrained path.
  python aion/experiments/test_pretrained_loader.py

--real: the real frozen foundation on Lappy via the documented checkpoint dir
  (/mnt/seagate/aion-models/Qwen3.5-4B-Base, CPU BF16) — asserts the smoke's
  proven numbers (426 frozen tensors / 4,205,751,296 params).
  python aion/experiments/test_pretrained_loader.py --real
"""

import argparse
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from aion.foundation.pretrained_loader import (  # noqa: E402
    FrozenBase, foundation_fingerprint, load_frozen_base,
)
from aion.relational.interface import fingerprints_equal  # noqa: E402

REAL_DIR = "/mnt/seagate/aion-models/Qwen3.5-4B-Base"
REAL_FROZEN_TENSORS = 426
REAL_FROZEN_PARAMS = 4_205_751_296  # text stack (model.language_model.*)


def check_handle(h: FrozenBase, expect_tensors=None, expect_params=None) -> None:
    """Common freeze-evidence assertions on a loaded handle."""
    d = h.describe()
    assert h.model is not None
    assert d["all_frozen"], "foundation has trainable params!"
    assert h.n_frozen_tensors > 0 and h.n_frozen_params > 0
    assert not any(p.requires_grad for p in h.model.parameters())
    assert h.hidden_size > 0
    # fingerprint evidence: non-empty, deterministic, and matching a fresh recompute
    assert len(h.fingerprint) == h.n_frozen_tensors
    ok, diffs = fingerprints_equal(h.fingerprint, foundation_fingerprint(h.model))
    assert ok, f"fingerprint mismatch vs recompute: {diffs[:3]}"
    if expect_tensors is not None:
        assert h.n_frozen_tensors == expect_tensors, \
            f"expected {expect_tensors} frozen tensors, got {h.n_frozen_tensors}"
    if expect_params is not None:
        assert h.n_frozen_params == expect_params, \
            f"expected {expect_params:,} frozen params, got {h.n_frozen_params:,}"
    print(f"[loader] frozen={h.n_frozen_params:,} params / {h.n_frozen_tensors} tensors; "
          f"hidden={h.hidden_size}; fingerprint entries={len(h.fingerprint)}; "
          f"describe.all_frozen={d['all_frozen']}")


def test_standin() -> None:
    """Tiny stand-in through the full public path (save_pretrained -> from_pretrained)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tiny = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64))
    with tempfile.TemporaryDirectory() as tmp:
        tiny.save_pretrained(tmp)
        # the documented public loading path (the one Lane G could not use)
        h = load_frozen_base(local_dir=tmp, device="cpu", dtype=torch.float32)
        check_handle(h)
        # tokenizer may or may not resolve for a bare save; model evidence is the contract
        assert h.n_frozen_params == tiny.num_parameters()
    print("[loader] stand-in public path PASS")


@pytest.mark.skipif(not os.path.isdir(REAL_DIR), reason="real checkpoint only exists on Lappy (/mnt/seagate)")
def test_real() -> None:
    """Real Qwen3.5-4B-Base through the public path (Lappy, CPU BF16)."""
    assert os.path.isdir(REAL_DIR), f"real checkpoint not found: {REAL_DIR}"
    h = load_frozen_base(local_dir=REAL_DIR, device="cpu", dtype=torch.bfloat16)
    check_handle(h, expect_tensors=REAL_FROZEN_TENSORS, expect_params=REAL_FROZEN_PARAMS)
    assert h.tokenizer is not None, "tokenizer should resolve from the checkpoint dir"
    print("[loader] REAL public path PASS (426 tensors / 4,205,751,296 params, "
          "fingerprint matches smoke evidence)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help=f"load the real foundation from {REAL_DIR} (Lappy, ~2 min)")
    args = ap.parse_args()
    if args.real:
        test_real()
    else:
        test_standin()
    print("ALL PASS")


if __name__ == "__main__":
    main()
