"""LTE-1 STEP 3 — host-immutability four-point test (docs/LTE_PROTOCOL.md §3.1).

Owner authorisation 2026-11-26: "proceed" — the LTE-1 ladder (steps 3-6) with
stop-on-fail at each gate. This script demonstrates and records ALL FOUR points
across a representative LTE work sequence on the real Qwen host:

  1. No optimizer is ever constructed over any foundation parameter
     (structural: process-wide scan; this script builds none).
  2. Host parameter fingerprints identical BEFORE and AFTER the work sequence
     (per-tensor SHA-256 checksums + aggregate, step-2 instrument).
  3. No host parameter receives an update — per-phase fingerprint re-checks
     (after fixture build, after gradient computation, after gate-style step).
  4. Gradients to the plastic state are finite on expression steps; nonzero on
     this fixture (mathematically expected); legitimate zero-gradient steps
     would be recorded, never aborted (Rev 5 scoping).

Reuses the step-2 RealHostFixture verbatim (same loader, same literal seeding,
same mirror, same literal §2.2/§2.3 scoring) — import, do not duplicate.

Evidence: runs/lte_gates/host_immutable_<timestamp>.json. Stop-on-fail: any
FAILED point exits 1 — report to coordinator, do not continue the ladder.
"""

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aion.experiments.lte_gates_real import RealHostFixture, ETA_E


def tensor_digest(model) -> dict:
    """Per-tensor SHA-256 checksums + aggregate (bit-exact, dtype-reinterpreted)."""
    digests = {}
    for name, p in sorted(model.named_parameters()):
        flat = p.detach().cpu().contiguous().flatten()
        try:
            raw = flat.view(torch.uint8).numpy().tobytes()
        except Exception:
            raw = flat.to(torch.float32).view(torch.uint8).numpy().tobytes()
        digests[name] = hashlib.sha256(raw).hexdigest()
    agg = hashlib.sha256("|".join(f"{k}:{v}" for k, v in digests.items()).encode()).hexdigest()
    return {"per_tensor": digests, "aggregate": agg, "n_tensors": len(digests),
            "n_params": int(sum(p.numel() for p in model.parameters()))}


def phase_check(model, fp_before: dict, label: str, results: dict) -> bool:
    """Point 3: per-phase no-update verification against the BEFORE digest."""
    now = tensor_digest(model)
    same_agg = now["aggregate"] == fp_before["aggregate"]
    same_all = all(now["per_tensor"][k] == v for k, v in fp_before["per_tensor"].items())
    ok = bool(same_agg and same_all and now["n_tensors"] == fp_before["n_tensors"])
    results[f"phase_{label}"] = {"aggregate_unchanged": same_agg,
                                 "all_tensors_unchanged": same_all,
                                 "n_tensors": now["n_tensors"], "pass": ok}
    print(f"[host-check] phase {label}: host fingerprint unchanged = {ok}")
    return ok


def main() -> int:
    t0 = time.time()
    print("[host-check] LTE-1 STEP 3 — host-immutability four-point test (protocol §3.1, Rev 8)")
    results: dict[str, Any] = {}

    # ---- fixture (identical to step 2: exercises expression gradients through the host) ----
    print("[host-check] building real-host fixture (8 frozen-host forwards, ~6 min)...")
    fx = RealHostFixture(seed=20261126)
    model = fx.iface.base
    results["fixture"] = {"facts": len(fx.items), "n_conns": len(fx.iface.conns),
                          "n_host_tensors": None}

    # ---- POINT 1: no optimizer over foundation parameters (structural, process-wide) ----
    import gc
    optimizers_found = sum(1 for o in gc.get_objects() if isinstance(o, torch.optim.Optimizer))
    point1 = optimizers_found == 0
    results["point1_no_optimizer"] = {"optimizers_in_process": optimizers_found,
                                      "pass": bool(point1)}
    print(f"[host-check] POINT 1 no-optimizer: {optimizers_found} optimizers in process -> "
          f"{'PASS' if point1 else 'FAIL'}")

    # ---- fingerprint BEFORE ----
    fp_before = tensor_digest(model)
    results["fixture"]["n_host_tensors"] = fp_before["n_tensors"]
    results["fingerprint_before"] = {"aggregate": fp_before["aggregate"],
                                     "n_tensors": fp_before["n_tensors"],
                                     "n_params": fp_before["n_params"]}
    print(f"[host-check] fingerprint BEFORE: {fp_before['n_tensors']} tensors, "
          f"{fp_before['n_params']:,} params, digest {fp_before['aggregate'][:16]}...")

    all_pass = True
    fx.build_gate_state()                        # fixture's own plastic-state construction
                                                 # (literal seeding + constructed fallback;
                                                 # prints the owner-required diagnostics)
    state = fx.get_state()                       # pre-sequence plastic state (kept)
    assert len(state) == len(fx.iface.conns), "u_state must match connection count"
    all_pass &= phase_check(model, fp_before, "A_fixture_build", results)

    # ---- Phase B: expression gradient computation through the host (Gate-1 style) ----
    leaves = [u.detach().clone().requires_grad_(True) for u in state]
    F_before = fx.eval_F(leaves)
    F_before.backward()
    g = [t.grad.detach().clone() for t in leaves]
    finite = bool(all(torch.isfinite(t).all() for t in g))
    grad_norm = float(sum(float(t.norm()) ** 2 for t in g) ** 0.5)
    nonzero = grad_norm > 0
    results["gradient_quality"] = {"finite": finite, "nonzero": nonzero,
                                   "grad_norm": grad_norm}
    all_pass &= phase_check(model, fp_before, "B_gradient_computation", results)
    print(f"[host-check] expression gradients: finite={finite} nonzero={nonzero} "
          f"norm={grad_norm:.6f}")

    # ---- Phase C: gate-style plastic step (host untouched; plastic restored after) ----
    eta_gate = 0.1 * ETA_E
    stepped = [s - eta_gate * gv for s, gv in zip(state, g)]
    fx.set_state(stepped)
    F_after = float(fx.eval_F([s for s in fx.u_state]))
    decreased = F_after < float(F_before.item())
    fx.set_state(state)                          # plastic restored; host never touched
    all_pass &= phase_check(model, fp_before, "C_gate_style_step", results)
    results["gate_style_step"] = {"F_before": float(F_before.item()), "F_after": F_after,
                                  "decreased": bool(decreased), "eta_gate": eta_gate}
    print(f"[host-check] gate-style step: F {F_before.item():.6f} -> {F_after:.6f} "
          f"decreased={decreased} (plastic state restored)")

    # ---- POINT 2/3: fingerprint AFTER the full sequence ----
    fp_after = tensor_digest(model)
    same = fp_after["aggregate"] == fp_before["aggregate"]
    per_same = all(fp_after["per_tensor"][k] == v for k, v in fp_before["per_tensor"].items())
    point2_3 = bool(same and per_same and fp_after["n_tensors"] == fp_before["n_tensors"])
    results["point2_3_fingerprint_after"] = {
        "aggregate_unchanged": same, "all_tensors_unchanged": per_same,
        "aggregate_before": fp_before["aggregate"], "aggregate_after": fp_after["aggregate"],
        "pass": point2_3}
    print(f"[host-check] POINT 2+3 fingerprint after full sequence: unchanged = {point2_3}")

    # ---- POINT 4: gradient finiteness on the sequence (Rev 5 scoping) ----
    point4 = bool(finite)
    results["point4_gradient_finiteness"] = {"finite": finite, "pass": point4}
    print(f"[host-check] POINT 4 gradient finiteness: {finite} -> "
          f"{'PASS' if point4 else 'FAIL'}")

    verdict = "PASS" if (point1 and point2_3 and point4 and all_pass and decreased) else "FAIL"
    results["verdict"] = verdict
    results["wall_s"] = round(time.time() - t0, 1)

    out = Path("runs/lte_gates/host_immutable_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n[host-check] VERDICT: {verdict}   evidence: {out}  ({results['wall_s']}s)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
