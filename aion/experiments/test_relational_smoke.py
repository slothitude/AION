"""AION Stage 1-2 acceptance smoke: frozen Qwen foundation + relational pathway.

Owner acceptance table (2026-11-24), each row must print PASS with numbers:
  1. Foundation loads with EVERY parameter frozen (requires_grad=False, count reported).
  2. Hidden states extract from the REAL module tree — hooks fire on selected
     layers of the HYBRID architecture (DeltaNet AND attention both exercised),
     shapes verified (2560-dim on the real model).
  3. ZERO-INITIALISED residual gate (gamma) leaves the model output UNCHANGED —
     identity at insertion, float-tolerance assert.
  4. An NNFS update CHANGES the logits-level output while foundation weights stay
     bit-identical (fingerprint before/after).

Run:
  python aion/experiments/test_relational_smoke.py                        # tiny stand-in on CPU
  python aion/experiments/test_relational_smoke.py \
      --local-dir /mnt/seagate/aion-models/Qwen3.5-4B-Base                # REAL Qwen3.5-4B-Base
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from aion.relational.interface import (  # noqa: E402
    RelationalInterface, InterfaceConfig, foundation_fingerprint, fingerprints_equal,
)


def build_base(model_id=None, local_dir=None, device="cpu"):
    """REAL Qwen3.5-4B-Base when given an id/dir; tiny in-memory stand-in otherwise."""
    if model_id or local_dir:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        src = local_dir or model_id
        model = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None,
            local_files_only=bool(local_dir))
        model = model.to(device)
        tok = AutoTokenizer.from_pretrained(src, local_files_only=bool(local_dir))
        return model, tok, src
    # stand-in: 4-layer Llama, random init, no download (proves the pipeline only)
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    return model, None, "tiny-standin(Llama-4L-64D)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    model, tok, src = build_base(args.model_id, args.local_dir, args.device)
    print(f"[smoke] base model: {src} on {args.device}")

    # ---- tokens ----
    if tok is not None:
        enc = tok(args.prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
    else:
        torch.manual_seed(1)
        input_ids = torch.randint(0, 128, (2, 12), device=args.device)

    results = {}

    # ============ criterion 1: fully frozen foundation ============ #
    iface = RelationalInterface(model)
    d = iface.describe()
    frozen = iface.freeze_stats["frozen_params"]
    frozen_numel = iface.freeze_stats["frozen_numel"]
    n_trainable = len(d["trainable"])
    results["1_frozen"] = (frozen > 0 and n_trainable == len(iface.gates),
                           f"{frozen_numel:,} foundation params ({frozen} tensors) "
                           f"requires_grad=False; trainable={n_trainable} (gates only)")
    print(f"[crit1] frozen={frozen:,} tensors / {frozen_numel:,} params, "
          f"trainable={n_trainable} gates={len(iface.gates)}")

    # ============ criterion 2: hybrid hooks fire, shapes verified ============ #
    res = iface.forward(input_ids)
    shapes_ok = True
    shape_report = []
    for (pre_i, post_i) in iface.pairs:
        hs = (tuple(res["hidden"][pre_i].shape), tuple(res["hidden"][post_i].shape))
        want = (input_ids.shape[0], input_ids.shape[1], iface.hidden_dim)
        shapes_ok = shapes_ok and hs[0] == want and hs[1] == want
        shape_report.append(f"{iface.layers[pre_i].class_name}->{iface.layers[post_i].class_name}:{hs[0]}")
    kinds = " + ".join(sorted(set(d["layer_families"])))
    hybrid_ok = ("deltanet" in kinds and "attention" in kinds) or \
                (args.model_id is None and args.local_dir is None)
    results["2_hooks"] = (shapes_ok and hybrid_ok,
                          f"{d['n_layers_discovered']} layers [families: {kinds}]; "
                          f"pairs={shape_report}; hidden_dim={iface.hidden_dim}")
    print(f"[crit2] layers={d['n_layers_discovered']} families={kinds}")
    for pr in d["hooked_pairs"]:
        print(f"[crit2]   {pr.get('pre_family', '?')} -> {pr.get('post_family', '?')}"
              f"  ({pr['pre']} -> {pr['post']})")

    # ============ criterion 3: zero-init gate = identity at insertion ============ #
    base_logits = res["base_logits"].float()
    logits0 = res["logits"].float()
    diff0 = (logits0 - base_logits).abs().max().item()
    results["3_identity"] = (torch.allclose(logits0, base_logits, atol=1e-5, rtol=0),
                             f"max|logits'-logits| at gamma=0: {diff0:.3e} (tol 1e-5)")
    gammas0 = [round(float(g.detach()), 6) for g in iface.gates]
    print(f"[crit3] gamma={gammas0} maxdiff={diff0:.3e}")

    fp_before = foundation_fingerprint(model)

    # ============ learning: NNFS first (u_fast), then gate steps (gamma) ============ #
    # Order matters: transmitted = a*psi with a = 0 at init, so the gate gradient
    # is exactly zero until the NNFS update makes u_fast non-zero.
    u_before = [c.plasticity.u_fast.norm().item() for c in iface.conns]
    iface.nnfs_update(input_ids)
    u_after = [c.plasticity.u_fast.norm().item() for c in iface.conns]
    print(f"[learn] nnfs#1 u_fast norms before={['%.4f' % u for u in u_before]} "
          f"after={['%.4f' % u for u in u_after]}")

    g = iface.gate_step(input_ids)
    print(f"[learn] gate_step#1 loss={g['loss']:.4f} gamma={g['gate_norms']}")
    for _ in range(4):
        iface.nnfs_update(input_ids)
    g2 = iface.gate_step(input_ids)
    print(f"[learn] after 4 more nnfs + gate_step#2 loss={g2['loss']:.4f} gamma={g2['gate_norms']}")

    u_last = [c.plasticity.u_fast.norm().item() for c in iface.conns]
    iface.nnfs_update(input_ids)
    u_new = [c.plasticity.u_fast.norm().item() for c in iface.conns]
    print(f"[learn] final nnfs u_fast before={['%.4f' % u for u in u_last]} "
          f"after={['%.4f' % u for u in u_new]}")

    # ============ criterion 4: NNFS update changes LOGITS, foundation bit-same ============ #
    res2 = iface.forward(input_ids)
    logits2 = res2["logits"].float()
    reach_ok = res2["logits_delta"].abs().max().item() > 0
    nnfs_delta = (logits2 - logits0).abs().max().item()
    changed = nnfs_delta > 0
    fp_ok, diffs = fingerprints_equal(fp_before, foundation_fingerprint(model))
    results["4_nnfs"] = (reach_ok and changed and fp_ok,
                         f"logits_delta max={res2['logits_delta'].abs().max().item():.3e}; "
                         f"NNFS changed logits by {nnfs_delta:.3e}; "
                         f"foundation bit-identical={fp_ok} ({len(fp_before)} params checked)"
                         + (f" DIFFS={diffs[:3]}" if diffs else ""))
    print(f"[crit4] reach_delta={res2['logits_delta'].abs().max().item():.3e} "
          f"nnfs_logit_change={nnfs_delta:.3e} fingerprint_ok={fp_ok}")

    # ---- verdict ---- #
    print("\n===== AION STAGE 1-2 ACCEPTANCE =====")
    all_pass = True
    for k in ("1_frozen", "2_hooks", "3_identity", "4_nnfs"):
        ok, msg = results[k]
        all_pass = all_pass and ok
        print(f"{'PASS' if ok else 'FAIL'}  {k}: {msg}")
    print("ALL PASS" if all_pass else "SMOKE FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
