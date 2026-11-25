# DREAM D1 BASELINE — FROZEN 2026-11-25

The validated D1-only Dream Engine re-implementation: replay is a **real,
measured intervention** (validation gate PASS per owner condition 3), the
θ-write bypass is eliminated, the guard is calibrated both ways, and the
downstream question (does replay *help*) remains deliberately open.

**Owner decision record:** docs/DREAM_REIMPL_SPEC.md §6a (D1–D6, binding).
**Validation plan:** docs/D1_REPLAY_VALIDATION_PLAN.md (V1–V4, all PASS).
**Results:** pilot_results/d1_validation_results.json.

## Archive
`freeze/DREAM_D1_BASELINE_20261125.tar.gz`
SHA-256: `ae1ed7da796a2b9e6094f9df221d934fcdefe2d58213a0c710ac010ad6d3204f`

## Frozen per-file hashes
```
86902e5a503cc20038326da1ebb14af8e372ffd83b46c0772f1de09038e6a38d  src/nnfs/dreaming/replay.py
124a4e06c74ca336be12acd20818de455ab326f3b670bdc05621a8193bfab51e  src/nnfs/dreaming/simulator.py
481381623f827c4cca2edec15ecb21402df86b14552e8ab7de5f3927a29616f9  src/nnfs/dreaming/evaluator.py
2853675f699d8acf90532704a07fa33ef77b06c0225c6a4fe752bc623c65513a  src/nnfs/dreaming/consolidator.py
95cf7a98b2d449876579f526f960966517f9542a31b62fdfa634e1f5643ea635  src/nnfs/dreaming/guard.py
```

## Validation evidence (V1–V4, all PASS)
- V1 actual updates: 54/54 cycles ‖Δu‖ 0.47–2.15; D0 exactly 0.0
- V2 provenance: 1,728 records replayed=true; 0 untagged
- V3 bounded compute: min(32, retrieved) honoured; 1.1–1.4 s/cycle
- V4 above noise: floor 0.0; D1 min delta 0.4729 — 100% above
- D1 ≠ D0: bit-identity broken
- Honest: downstream held-out batteries barely moved at this dose — replay
  is a real state intervention; whether it helps is the six-config question

## What this freeze does NOT authorise
- D2 imagination evaluation (separate owner decision — the D1 gate is not
  permission for D2)
- The six-configuration re-run (requires its own frozen preregistration)
- Any dreaming-benefit claim
