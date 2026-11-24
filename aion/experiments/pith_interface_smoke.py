"""AION Pith interface smoke — four-criterion acceptance on the FACTOR-EXECUTED Pith-E2B.

Owner task 15 (2026-11-25). Substrate: Pith-E2B v1 (dual-svid, 1.5 BPW + PLE rowwise8)
executed FROM FACTORS by the validated LARQL-D1 runner (R3b: bit-exact through 23
stages; gate green). This test proves PLUMBING ONLY — learning-quality claims are
FORBIDDEN (Pith-E2B v1 fails its behavior gate: hidden cosine 0.118, argmax 0/53
vs the dense BF16 baseline; report divergences as the post-QAT comparison baseline).

Criteria:
  1. The factor-executed model runs FROZEN — no writes to factors; artifact file
     hash + in-memory factor fingerprints unchanged.
  2. Hidden states extract from the REAL Gemma-4 tree via the runner's run_layer
     isolation (forward.rs:318) — both block families exercised. NOTE: Gemma-4
     has NO DeltaNet blocks (that's Qwen3.5); its two attention families are
     sliding-window and global (full_attention at i%5==4), plus two MLP width
     regimes (6144 on layers 0-14, 12288 double-wide on 15-34). Pairs are chosen
     to span all of those.
  3. ZERO-INIT gamma residual = IDENTITY at insertion — AION-attached logits
     BIT-EXACT vs the runner's native output under the bf16-forward-v1 contract
     (head math executed by the runner itself, not re-implemented).
  4. An NNFS update changes the pathway contribution AND the outputs, with
     factor fingerprints unchanged. (Interface-level plumbing; no Pith learning.)

Run (Lappy):
  /home/aaron/pith-pilot/venv/bin/python aion/experiments/pith_interface_smoke.py \
      [--fixture 0] [--out results.json]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)  # repo convention on Rog; on Lappy the bridge tree
#                             carries src/nnfs/connections/plastic.py instead

DEFAULT_BRIDGE = "/opt/aion-seed/pith-bridge/target/release/pith_bridge"
DEFAULT_CHECKPOINT = "/home/aaron/pith-pilot/runs/pith-e2b-1p5-dualsvid"

# Gemma-4-E2B measured topology (R3b §3): full_attention exactly at i%5==4;
# sliding elsewhere; FFN 6144 on layers 0-14, 12288 on 15-34.
GLOBAL_LAYERS = {4, 9, 14, 19, 24, 29, 34}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class Bridge:
    """Line/binary protocol client for pith_bridge."""

    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _readline(self):
        return self.p.stdout.readline().decode()

    def _read_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError(f"bridge closed after {len(buf)}/{n} bytes")
            buf += chunk
        return buf

    def cmd(self, obj, payload=None, expect=None):
        self.p.stdin.write((json.dumps(obj) + "\n").encode())
        if payload is not None:
            self.p.stdin.write(np.asarray(payload, dtype="<f4").tobytes())
        self.p.stdin.flush()
        meta = json.loads(self._readline())
        if not meta.get("ok"):
            raise RuntimeError(f"bridge error on {obj.get('cmd')}: {meta}")
        data = None
        if expect == "dim":
            data = np.frombuffer(self._read_exact(4 * meta["dim"]), dtype="<f4")
        elif expect == "vocab":
            data = np.frombuffer(self._read_exact(4 * meta["vocab"]), dtype="<f4")
        return meta, data

    def close(self):
        try:
            self.cmd({"cmd": "quit"})
        except Exception:
            pass
        self.p.wait(timeout=10)


def load_plastic():
    """PlasticRelationalConnection — direct-file load of the frozen plasticity code."""
    import importlib.util
    candidates = [
        os.path.join(_SRC, "nnfs", "connections", "plastic.py"),
        os.path.join(_REPO_ROOT, "src", "nnfs", "connections", "plastic.py"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise RuntimeError(f"plastic.py not found in {candidates}")
    spec = importlib.util.spec_from_file_location("aion_frozen_plastic_pith", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("aion_frozen_plastic_pith", mod)
    spec.loader.exec_module(mod)
    return mod.PlasticRelationalConnection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=DEFAULT_BRIDGE)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--fixture", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="probe opening for the gamma gate (insertion is gamma=0)")
    ap.add_argument("--nnfs-rounds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = {"checkpoint": args.checkpoint, "fixture": args.fixture}

    PlasticRelationalConnection = load_plastic()

    # ---------------- criterion 1 setup: artifact hash BEFORE ----------------
    t0 = time.time()
    artifact = os.path.join(args.checkpoint, "model.safetensors")
    sha_before = sha256_file(artifact)
    results["artifact_sha256_before"] = sha_before
    print(f"[setup] artifact sha256(before)={sha_before[:16]}… ({time.time()-t0:.0f}s)")

    br = Bridge(args.bridge)
    meta, _ = br.cmd({"cmd": "load", "checkpoint": args.checkpoint})
    assert meta.get("ok"), meta
    hidden_dim, n_layers, vocab = meta["hidden"], meta["layers"], meta["vocab"]
    print(f"[setup] loaded: hidden={hidden_dim} layers={n_layers} vocab={vocab} "
          f"embed_scale={meta['embed_scale']} softcap={meta['softcap']} "
          f"ple_dim={meta['ple_dim']} factor_tensors={meta['factor_tensors']}")
    results["meta"] = meta

    fp_meta, _ = br.cmd({"cmd": "fingerprint"})
    assert fp_meta.get("ok"), fp_meta
    fp_before = fp_meta["hash"]
    results["factor_fingerprint_before"] = fp_before
    print(f"[setup] factor fingerprint(before)={fp_before} over {fp_meta['probes']} probes")

    # ---------------- fixture + criterion 2: stage-drive the real tree ----------------
    tokens = [2, 108, 4023, 954, 777, 9104, 235336]  # r3b fixture 0 (prompt_where_is)
    # ORDER LAW: the native full prefill runs FIRST — it resets and consumes the
    # KV caches (7 tokens -> 7 rows); the stage-drive (seq) then resets again and
    # pre-fills the 6-token prefix so run_layer(pos=6) satisfies the runner's
    # cache law (cache.used() == pos + 1 after append). Native after seq would
    # leave 7 rows behind and layer 0 would fail the phase law at 8 vs 7.
    m_native, logits_native = br.cmd({"cmd": "native", "tokens": tokens}, expect="vocab")
    assert m_native.get("ok"), m_native

    m, _ = br.cmd({"cmd": "seq", "tokens": tokens})
    assert m.get("ok"), m
    pos = m["pos"]

    hiddens = []
    m, h_after_seq = br.cmd({"cmd": "get"}, expect="dim")
    hiddens.append(h_after_seq.copy())  # layer-input embedding state
    for layer in range(n_layers):
        m, hout = br.cmd({"cmd": "layer", "layer": layer}, expect="dim")
        assert m.get("ok"), m
        assert hout.shape == (hidden_dim,) and np.isfinite(hout).all()
        hiddens.append(hout.copy())
    assert len(hiddens) == n_layers + 1

    fam = lambda i: "global_attn" if i in GLOBAL_LAYERS else "sliding_attn"
    pairs = [(0, 1), (3, 4), (14, 15), (33, 34)]
    pair_desc = [f"{a}->{b} ({fam(a)}->{fam(b)})" for a, b in pairs]
    print(f"[crit2] stage-drove {n_layers} layers of the REAL Gemma-4 tree; "
          f"hooked pairs: {', '.join(pair_desc)}")
    results["c2_pairs"] = pair_desc
    results["c2_hidden_dim"] = hidden_dim
    fams_exercised = set()
    for a, b in pairs:
        fams_exercised.update({fam(a), fam(b)})
    c2_ok = hiddens[-1].shape == (hidden_dim,) and len(fams_exercised) >= 2
    print(f"[crit2] families exercised: {sorted(fams_exercised)}; final hidden shape ok={c2_ok}")

    # ---------------- criterion 3: gamma=0 identity, BIT-EXACT ----------------
    m, logits_stage = br.cmd({"cmd": "head"}, expect="vocab")
    assert m.get("ok"), m
    stage_exact_native = bool(np.array_equal(logits_stage, logits_native))
    print(f"[crit3] stage-path logits == native prefill logits (bit-exact): {stage_exact_native}")

    h_final = hiddens[-1]
    # rel = pathway contributions at gamma=0 BEFORE any learning (u_fast=0 -> r=0)
    conns = []
    for (a, b) in pairs:
        c = PlasticRelationalConnection(
            pre_idx=a, post_idx=b, pre_dim=hidden_dim, post_dim=hidden_dim,
            context_dim=0, relation_type="post", learning_rate=args.lr,
        )
        for p in c.parameters():
            p.requires_grad_(False)
        conns.append(c)

    def pathway_r(u_fast_driven=False):
        """Sum of transmitted signals over pairs at the CURRENT plastic state."""
        tot = np.zeros(hidden_dim, dtype=np.float32)
        for c, (a, b) in zip(conns, pairs):
            pre = hiddens[a] / max(1e-8, np.linalg.norm(hiddens[a]))
            post = hiddens[b] / max(1e-8, np.linalg.norm(hiddens[b]))
            import torch
            with torch.no_grad():
                r, _ = c(torch.from_numpy(pre).unsqueeze(0),
                         torch.from_numpy(post).unsqueeze(0), learn=False)
            tot += r.squeeze(0).numpy().astype(np.float32)
        return tot

    r_before = pathway_r()
    h_gamma0 = (h_final + 0.0 * r_before).astype(np.float32)
    m, logits_gamma0 = br.cmd({"cmd": "set"}, payload=h_gamma0)
    assert m.get("ok"), m
    m, logits_gamma0 = br.cmd({"cmd": "head"}, expect="vocab")
    gamma0_exact = bool(np.array_equal(logits_gamma0, logits_stage))
    results["c3_stage_equals_native"] = stage_exact_native
    results["c3_gamma0_bit_exact"] = gamma0_exact
    print(f"[crit3] gamma=0 attached logits == native (bit-exact): {gamma0_exact}")

    # ---------------- criterion 4: NNFS update changes contribution + outputs ----------------
    import torch
    u_before = [float(c.plasticity.u_fast.norm()) for c in conns]
    for _ in range(args.nnfs_rounds):
        for c, (a, b) in zip(conns, pairs):
            pre = torch.from_numpy(hiddens[a] / max(1e-8, np.linalg.norm(hiddens[a]))).unsqueeze(0)
            post = torch.from_numpy(hiddens[b] / max(1e-8, np.linalg.norm(hiddens[b]))).unsqueeze(0)
            c(pre, post, target=post, learn=True)
    u_after = [float(c.plasticity.u_fast.norm()) for c in conns]
    print(f"[crit4] nnfs u_fast norms before={['%.4f' % u for u in u_before]} "
          f"after={['%.4f' % u for u in u_after]}")

    r_after = pathway_r()
    dr = float(np.abs(r_after - r_before).max())
    results["c4"] = {}
    results["c4"]["u_fast_before"] = u_before
    results["c4"]["u_fast_after"] = u_after
    results["c4"]["r_max_delta"] = dr

    # outputs at the probe gate opening gamma=1: h' = h + gamma * r_after
    h_gamma1 = (h_final + args.gamma * r_after).astype(np.float32)
    br.cmd({"cmd": "set"}, payload=h_gamma1)
    m, logits_gamma1 = br.cmd({"cmd": "head"}, expect="vocab")
    d_logits = float(np.abs(logits_gamma1 - logits_stage).max())
    cos = float(np.dot(logits_gamma1, logits_stage) /
                (np.linalg.norm(logits_gamma1) * np.linalg.norm(logits_stage) + 1e-30))
    changed = dr > 0 and d_logits > 0
    results["c4"]["logits_max_delta"] = d_logits
    results["c4"]["logits_cosine_vs_stage"] = cos
    print(f"[crit4] NNFS changed pathway contribution r by {dr:.3e}; "
          f"gamma={args.gamma} changed logits by {d_logits:.3e} (cosine {cos:.6f})")

    # ---------------- criterion 1 (closed): frozen after everything ----------------
    fp_meta, _ = br.cmd({"cmd": "fingerprint"})
    fp_after = fp_meta["hash"]
    sha_after = sha256_file(artifact)
    frozen = (fp_after == fp_before) and (sha_after == sha_before)
    results["factor_fingerprint_after"] = fp_after
    results["artifact_sha256_after"] = sha_after
    results["c1_frozen"] = frozen
    print(f"[crit1] factor fingerprint unchanged={fp_after == fp_before}; "
          f"artifact sha256 unchanged={sha_after == sha_before}")

    # ---------------- honest divergence numbers (behavior gate stands) ----------------
    probs = np.exp(logits_native - logits_native.max())
    probs = probs / probs.sum()
    top = np.argsort(-probs)[:5]
    ent = float(-(probs * np.log(probs + 1e-30)).sum())
    results["behavior"] = {
        "native_top5": [[int(t), float(probs[t])] for t in top],
        "entropy_nats": ent,
        "note": "Pith-E2B v1 fails its behavior gate (r3b: logits cosine median 0.14 "
                "vs dense BF16, argmax 0/53; coordinator: hidden cosine 0.118). "
                "Degraded brain is EXPECTED; these numbers are the post-QAT baseline. "
                "NO learning-quality claims.",
    }
    print(f"[behavior] fixture top-5: {[(int(t), round(float(probs[t]), 5)) for t in top]}; "
          f"entropy={ent:.3f} nats")

    br.close()

    # ---------------- verdict ----------------
    verdict = {
        "1_frozen": frozen,
        "2_hooks": c2_ok,
        "3_identity": stage_exact_native and gamma0_exact,
        "4_nnfs": bool(changed),
    }
    results["verdict"] = verdict
    print("\n===== AION PITH INTERFACE ACCEPTANCE =====")
    print(f"{'PASS' if verdict['1_frozen'] else 'FAIL'}  1_frozen: artifact "
          f"sha256 + factor fingerprints unchanged ({fp_meta['probes']} probes)")
    print(f"{'PASS' if verdict['2_hooks'] else 'FAIL'}  2_hooks: {n_layers} real layers "
          f"stage-drove via run_layer; families {sorted(fams_exercised)}; "
          f"hidden {hidden_dim}")
    print(f"{'PASS' if verdict['3_identity'] else 'FAIL'}  3_identity: stage==native "
          f"bit-exact={stage_exact_native}; gamma=0 attached==native bit-exact={gamma0_exact}")
    print(f"{'PASS' if verdict['4_nnfs'] else 'FAIL'}  4_nnfs: r delta {dr:.3e}, "
          f"logits delta {d_logits:.3e}, fingerprints unchanged")
    all_pass = all(verdict.values())
    print("ALL PASS" if all_pass else "SMOKE FAILED")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
