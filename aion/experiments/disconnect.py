"""THE DISCONNECT EXPERIMENT — AION-1's defining autonomy milestone (owner protocol).

The owner's formal 6-step protocol, implemented as explicit phases (each returns
a phase report; every phase is re-runnable from disk):

    step 0  TEACH                    build both lanes; teach the same seed
                                     relations to each; the WITH-host control
                                     lane runs the FULL capability battery
                                     (this is the register's denominator run).
    step 1  DISCONNECT TEACHER       no external teaching or answer assistance
                                     from this step onward (teacher gate CLOSED).
    step 2  DISCONNECT HOST          no Qwen hidden states, no logits, no
                                     inference API, no cached model outputs. The
                                     relational pathway runs on DIRECT
                                     OBSERVATION encoding only (DirectFeed —
                                     knowledge_discovery's host-disconnected
                                     pattern).
    step 3  RETEST EXISTING KNOWLEDGE  can it reconstruct previously learned
                                     relationships? paraphrase battery +
                                     combination checks + provenance buckets.
    step 4  INTRODUCE NEW OBSERVATIONS can it learn NEW relationships from
                                     direct observation with no LLM assistance?
                                     new facts via the direct-feed encoder ->
                                     plastic updates -> tested.
    step 5  RUN OWN SIMULATIONS      predict outcomes from its INTERNAL world
                                     state (belief-graph walk over the relational
                                     index + live plastic strengths); predictions
                                     scored against the GRAPH GROUND TRUTH
                                     (intervention questions). Self-correct:
                                     after a wrong prediction the correction is
                                     applied as a plastic update and the
                                     prediction re-scored.
    step 6  RESTART                  bitwise state round-trip, then re-verify
                                     survival (the register uses the
                                     post-restart disconnected scores).

METRICS — the owner-corrected register (binding):
  * Capability-survival ratio per capability:
        survival = capability(WITHOUT host) / capability(WITH host)
    measured on the SAME battery; BOTH runs' numbers are reported side by side
    (the ratio never hides either). The register uses post-restart
    disconnected scores (step 6 re-verification) and also reports the
    pre-restart ones.
  * Per capability (the owner's five): recognise / predict / learn_new /
    combine / self_correct. self_correct = after a wrong prediction the
    correction is applied as a plastic update and the prediction re-scored.
  * PROVENANCE buckets on every answer: PLASTIC (u_fast alone, blend=1.0),
    CONSOLIDATED (theta_slow alone, blend=0.0), RETRIEVED (episodic store
    only), NONE. RETRIEVED answers NEVER count as owned knowledge (the Phase-1
    register law carries over). A belief-graph path active under both buckets
    is labelled MIXED (owned, not attributable to one bucket); a blend-only
    attribution is NONE with full_hit recorded so nothing is hidden.
  * CONFOUND GUARD (reviewer-required): substrate capability is reported
    alongside — when the WITH-host run used a degraded stand-in (the CPU stub)
    the survival ratios carry that caveat.

STRUCTURAL DISCONNECTION (not nominal): the without-host lane NEVER constructs
a foundation object at all (lane.iface is None, a poisoned loader replaces any
loader), and every host touch in either lane goes through HostMonitor — a
counting wrapper whose CLOSED state raises HostDisconnectedError on any
attempted host access. "Zero foundation forward passes" in the disconnected
scoring (steps 3-6) is a MEASURED count of zero plus a tripwire, not a claim.

BUILD-AHEAD LAW: this file was constructed and unit-tested with stubs/tiny
models on CPU. Real execution is gated on Phase 2 landing + owner: pass
--live (or live=True) AND export AION_DISCONNECT_LIVE=1 to lift the gate; any
other configuration reports the real run NOT-RUN honestly.

LAWS: no dreaming, no internal-lane probing, no fabricated numbers — a phase
that cannot run reports NOT-RUN. Battery gold answers are re-derived from the
battery's own fact graph by verify_battery() before any run (mirrors
aion/worlds/generator.verify_world); a bad battery refuses to run.

PHASE-2 INTEGRATION SEAM (how Lane G's landed loop invokes this once the
Phase-2 verdict lands — teacher_student.py itself is NOT touched):

    from aion.experiments.disconnect import run_disconnect_experiment

    report = run_disconnect_experiment(
        battery=<world battery JSON, same schema as DEFAULT_DISCONNECT_BATTERY>,
        run_dir="runs/disconnect_real",
        seed=0,
        live=True,                       # + env AION_DISCONNECT_LIVE=1
        foundation_loader=my_loader,     # any () -> FrozenBase handle; default
                                         # aion/foundation/pretrained_loader.load_frozen_base
        prebuilt_iface=<existing RelationalInterface>,   # OPTIONAL: hand over
                            # Lane G's already-loaded session interface so the
                            # 4B is loaded once; monitor wraps its forward.
        teacher=<TeacherClient | ScriptedTeacher>,       # step-0 lessons only;
                                                         # gated shut at step 1
    )

Seams needed from Lane G (documented, no code required from them):
  * a battery JSON from the worlds lane (schema below) — DEFAULT_DISCONNECT_BATTERY
    stands in until then;
  * optionally the live session's RelationalInterface (prebuilt_iface) to avoid
    a second 4B load — otherwise foundation_loader loads a fresh frozen copy;
  * nothing else: the harness owns lanes, monitors, phases, register, reports.

CLI (matches retention.py's pattern):
    # full protocol, stub lanes (build-ahead verification):
    python aion/experiments/disconnect.py --phase all --seed 0 --run-dir runs/disconnect_stub
    # split-process mode (phase state resumes from disk):
    python aion/experiments/disconnect.py --phase teach      --run-dir runs/disconnect_stub
    python aion/experiments/disconnect.py --phase retest     --run-dir runs/disconnect_stub
    # REAL run (post-Phase-2, owner-gated):
    AION_DISCONNECT_LIVE=1 python aion/experiments/disconnect.py --live \
        --foundation qwen --battery <battery.json> --run-dir runs/disconnect_real
"""

import argparse
import copy
import json
import math
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.memory.relational import (                                     # noqa: E402
    RelationalMemory, relation_key, parse_key,
)
from aion.memory.episodic import EpisodicStore                           # noqa: E402
from aion.memory.cpu_stubs import (                                      # noqa: E402
    build_stub_base, DirectFeed, StubVocab, normalize_answer, answer_matches,
)
from aion.memory.consolidation import ConsolidationLedger                # noqa: E402
from aion.memory.state import conn_tensor_map                            # noqa: E402
from aion.relational.interface import (                                  # noqa: E402
    RelationalInterface, InterfaceConfig, PlasticRelationalConnection,
)
from aion.experiments.knowledge_discovery import (                       # noqa: E402
    survival_ratio, survival_ratios, CAPABILITIES,
)

LIVE_GATE_ENV = "AION_DISCONNECT_LIVE"          # owner/Phase-2 gate; "1" lifts it
REAL_SUBSTRATE = "qwen3.5-4b-frozen"            # the frozen foundation (host lane)
PLASTIC = "PLASTIC"
CONSOLIDATED = "CONSOLIDATED"
RETRIEVED = "RETRIEVED"
NONE_PROV = "NONE"
MIXED = "MIXED"                                 # owned path active under both buckets
STRONGEST_FIRST = (PLASTIC, CONSOLIDATED, RETRIEVED)

PHASES = ("teach", "disconnect_teacher", "disconnect_host", "retest",
          "observe", "simulate", "restart", "register")
PHASE_ALIASES = {
    "0": "teach", "1": "disconnect_teacher", "2": "disconnect_host",
    "3": "retest", "4": "observe", "5": "simulate", "6": "restart",
}


class HostDisconnectedError(RuntimeError):
    """Raised when any code path attempts a host touch while the gate is CLOSED."""


class SequencingError(RuntimeError):
    """Raised when a phase runs before its prerequisites (NOT-RUN, honestly)."""


class LiveGateError(RuntimeError):
    """Raised when a real-foundation run is attempted without the owner gate."""


def _poison_loader(*_a, **_k):
    """Structural tripwire: the disconnected lane has NO foundation loader; if
    any code path ever reaches for one, this explodes instead of loading."""
    raise HostDisconnectedError(
        "poison loader hit: the disconnected lane must NEVER load a foundation")


def default_live_loader():
    """The real frozen foundation, loaded OFFLINE from the local checkpoint
    (aion/foundation/pretrained_loader.DEFAULT_LOCAL_DIR). NEVER downloads:
    a missing staged checkpoint is a hard operator error, not a fetch."""
    from aion.foundation import pretrained_loader as _pl
    local = _pl.DEFAULT_LOCAL_DIR
    if not os.path.isdir(local):
        raise LiveGateError(
            f"staged checkpoint missing: {local} is not present on this host — "
            f"stage Qwen/Qwen3.5-4B-Base there first (no network download "
            f"happens from this harness)")
    return _pl.load_frozen_base(local_dir=local)


# --------------------------------------------------------------------------- #
# HostMonitor — the structural disconnection is measured, not assumed
# --------------------------------------------------------------------------- #

class HostMonitor:
    """Counting gate around every host touch (foundation forward, teacher call,
    foundation load). Two enforcement layers:
      1. COUNTER — every wrapped call increments its kind; `since_disconnect()`
         must be all-zero across the disconnected protocol (steps 3-6).
      2. GATE — close() locks it: any wrapped call then raises
         HostDisconnectedError immediately (nothing slips through silently).
    """

    KINDS = ("foundation_forward", "teacher_call", "foundation_load")

    def __init__(self, name: str):
        self.name = name
        self.closed = False
        self.close_reason: Optional[str] = None
        self.counts: Dict[str, int] = {k: 0 for k in self.KINDS}
        self.counts_at_close: Optional[Dict[str, int]] = None

    def wrap(self, kind: str, fn: Callable):
        if kind not in self.KINDS:
            raise ValueError(f"unknown monitor kind {kind!r}")

        def wrapped(*args, **kwargs):
            if self.closed:
                raise HostDisconnectedError(
                    f"[{self.name}] gate CLOSED — attempted {kind} during the "
                    f"disconnected protocol")
            self.counts[kind] += 1
            return fn(*args, **kwargs)
        return wrapped

    def close(self, reason: str) -> None:
        if not self.closed:
            self.closed = True
            self.close_reason = reason
            self.counts_at_close = dict(self.counts)

    def since_close(self) -> Dict[str, int]:
        base = self.counts_at_close or {k: 0 for k in self.KINDS}
        return {k: self.counts[k] - base[k] for k in self.KINDS}

    def assert_zero_since_close(self) -> Dict[str, int]:
        d = self.since_close()
        bad = {k: v for k, v in d.items() if v != 0}
        if bad:
            raise HostDisconnectedError(
                f"[{self.name}] HOST TOUCHED while disconnected: {bad}")
        return d

    def snapshot(self) -> Dict:
        return {"name": self.name, "closed": self.closed,
                "close_reason": self.close_reason,
                "counts_total": dict(self.counts),
                "counts_at_close": dict(self.counts_at_close) if
                self.counts_at_close is not None else None,
                "counts_since_close": self.since_close()}


# --------------------------------------------------------------------------- #
# The two lanes (knowledge_discovery's HostLane/BareLane patterns, plus the
# monitor wiring and disk persistence)
# --------------------------------------------------------------------------- #

class LaneHost:
    """WITH-host CONTROL lane: probes and learning are fed by hooked foundation
    hidden states. Its full-battery capability scores are the register's
    denominators (the substrate's capability). Every foundation pass is counted
    by the monitor — the control lane is SUPPOSED to touch the host."""

    name = "host"

    def __init__(self, seed: int = 0, hidden_size: int = 64,
                 monitor: Optional[HostMonitor] = None,
                 prebuilt_iface: Optional[object] = None,
                 foundation_loader: Optional[Callable] = None,
                 vocab_size: int = 128):
        self.monitor = monitor or HostMonitor("host-control")
        self.iface = None
        self.substrate = None
        if prebuilt_iface is not None:                 # Lane G seam: reuse the
            self.iface = prebuilt_iface                # session's interface
            self.substrate = "prebuilt-foundation"
        elif foundation_loader is not None:
            handle = self.monitor.wrap("foundation_load", foundation_loader)()
            self.iface = RelationalInterface(handle.model, cfg=InterfaceConfig())
            self.substrate = getattr(handle, "model_id", REAL_SUBSTRATE)
        else:
            torch.manual_seed(seed)
            base = build_stub_base(seed=seed, hidden_size=hidden_size)
            self.iface = RelationalInterface(base, cfg=InterfaceConfig())
            self.substrate = "stub-tiny-llama"
        # COUNT EVERY forward: instance-level wrap catches nnfs_update and
        # gate_step too (they call self.forward -> instance attribute wins).
        self.iface.forward = self.monitor.wrap("foundation_forward", self.iface.forward)

        cfg = self.iface.cfg
        self.hidden_dim = self.iface.hidden_dim
        self.vocab = StubVocab(vocab_size=vocab_size)
        self.feed = None                               # no direct-feed encoder here
        self.episodic = EpisodicStore(clock=lambda: 0.0)

        def make_probe(prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                res = self.iface.forward(prompt_ids)   # counted host pass
            pre_i, post_i = self.iface.pairs[0]
            return (res["hidden"][pre_i].mean(dim=1)[0].float(),
                    res["hidden"][post_i].mean(dim=1)[0].float())

        def learn_fn(prompt_ids: torch.Tensor) -> Dict:
            return self.iface.nnfs_update(prompt_ids)  # counted host pass

        self.make_probe = make_probe
        self.learn_fn = learn_fn
        self.channel = "transcript"
        self.rmem = RelationalMemory(conns=list(self.iface.conns),
                                     hidden_dim=self.hidden_dim,
                                     episodic=self.episodic,
                                     make_probe=make_probe,      # REAL probes
                                     blend_ratio=cfg.blend_ratio,
                                     strength_threshold=1e-8)
        self.ledger = ConsolidationLedger(conns=list(self.iface.conns),
                                          hidden_dim=self.hidden_dim)


class LaneDisconnected:
    """WITHOUT-host lane: the host is STRUCTURALLY absent for this lane's whole
    life. No foundation object exists (iface is None); the loader slot holds a
    poison that raises if ever reached; the monitor starts CLOSED; probes and
    learning use DirectFeed deterministic observation states only."""

    name = "disconnected"

    def __init__(self, seed: int = 0, hidden_size: int = 64, vocab_size: int = 128,
                 monitor: Optional[HostMonitor] = None):
        torch.manual_seed(seed + 1)                    # independent, deterministic
        cfg = InterfaceConfig()
        self.conn = PlasticRelationalConnection(
            pre_idx=0, post_idx=1, pre_dim=hidden_size, post_dim=hidden_size,
            context_dim=0, relation_type=cfg.relation_type,
            learning_rate=cfg.learning_rate, trace_decay=cfg.trace_decay,
            blend_ratio=cfg.blend_ratio,
            consolidation_interval=cfg.consolidation_interval)
        for p in self.conn.parameters():
            p.requires_grad_(False)                    # NNFS owns these
        self.iface = None                              # STRUCTURAL: no host object
        self.foundation_loader = _poison_loader        # tripwire, never called
        self.feed = DirectFeed(vocab_size=vocab_size, dim=hidden_size, seed=99)
        self.hidden_dim = hidden_size
        self.vocab = StubVocab(vocab_size=vocab_size)
        self.episodic = EpisodicStore(clock=lambda: 0.0)
        self.monitor = monitor or HostMonitor("disconnected")
        self.monitor.close("lane constructed host-disconnected (structural)")

        def make_probe(prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            ids = [int(i) for i in prompt_ids.flatten().tolist()]
            return self.feed.encode(ids)               # direct observation ONLY

        def learn_fn(prompt_ids: torch.Tensor) -> Dict:
            pre, post = make_probe(prompt_ids)
            with torch.no_grad():
                _, stats = self.conn(pre.unsqueeze(0), post.unsqueeze(0),
                                     target=post.unsqueeze(0), learn=True)
            return {"nnfs": {"pair0": stats}}

        self.make_probe = make_probe
        self.learn_fn = learn_fn
        self.channel = "direct"                        # never had a transcript
        self.rmem = RelationalMemory(conns=[self.conn], hidden_dim=self.hidden_dim,
                                     episodic=self.episodic,
                                     make_probe=make_probe,   # REAL probes
                                     blend_ratio=cfg.blend_ratio,
                                     strength_threshold=1e-8)
        self.ledger = ConsolidationLedger(conns=[self.conn],
                                          hidden_dim=self.hidden_dim)

    # -- structural proof ----------------------------------------------------- #

    def assert_host_disconnected(self) -> Dict:
        """Measured structural proof: gate closed, zero host touches since close,
        no interface object exists, poison loader still uncalled."""
        if self.iface is not None:
            raise HostDisconnectedError("disconnected lane carries an interface")
        if self.foundation_loader is not _poison_loader:
            raise HostDisconnectedError("disconnected lane acquired a real loader")
        since = self.monitor.assert_zero_since_close()
        return {"iface_absent": True, "poison_loader_intact": True,
                "monitor_closed": self.monitor.closed,
                "host_touches_since_close": since,
                "host_touches_total": dict(self.monitor.counts)}

    # -- persistence (bare conn: no interface snapshot machinery) -------------- #

    def save_payload(self) -> Dict:
        return {
            "kind": "lane-disconnected",
            "hidden_dim": self.hidden_dim,
            "cfg": {"relation_type": self.conn.relation_type
                    if hasattr(self.conn, "relation_type") else "n/a"},
            "conn": {k: v.clone() for k, v in conn_tensor_map(self.conn).items()},
            "index": self.rmem.to_dict(),
            "probes": self.rmem.save_probes(),
            "episodic": self.episodic.to_dict(),
            "monitor": self.monitor.snapshot(),
        }

    def load_payload(self, payload: Dict) -> Tuple[bool, List[str]]:
        """Restore into THIS lane's objects (fresh-built by the constructor).
        Returns (exact, differing_tensor_names)."""
        diffs: List[str] = []
        live = conn_tensor_map(self.conn)
        for name, src in payload["conn"].items():
            dst = live.get(name)
            if dst is None or tuple(dst.shape) != tuple(src.shape):
                diffs.append(f"conn.{name}")
                continue
            with torch.no_grad():
                if isinstance(dst, torch.nn.Parameter):
                    dst.data.copy_(src)
                else:
                    dst.copy_(src)
        for name in live:
            if name not in payload["conn"]:
                diffs.append(f"conn.{name}(missing-in-payload)")
        self.rmem = RelationalMemory.from_dict(
            payload["index"], conns=[self.conn], episodic=self.episodic,
            probes=payload["probes"], make_probe=self.make_probe)
        self.episodic = EpisodicStore.from_dict(payload["episodic"],
                                                clock=lambda: 0.0)
        self.rmem.episodic = self.episodic
        return (not diffs), diffs


# --------------------------------------------------------------------------- #
# The default battery (invented held-out entities, worlds-lane schema; gold
# answers are re-derived by verify_battery() before any run — no fabricated
# numbers)
# --------------------------------------------------------------------------- #

def _q(qid, cap, prompt, paraphrase, gold):
    return {"qid": qid, "capability": cap, "prompt": prompt,
            "paraphrases": [paraphrase], "gold": gold}


SEED_CHAIN_A = [
    {"subject": "Neral", "relation": "operates", "object": "Tav"},
    {"subject": "Tav", "relation": "routes-to", "object": "Kimo"},
    {"subject": "Kimo", "relation": "controls", "object": "Zor"},
    {"subject": "Zor", "relation": "produces", "object": "heat"},
]
SEED_CHAIN_B = [
    {"subject": "Sorni", "relation": "controls", "object": "Vask"},
    {"subject": "Vask", "relation": "produces", "object": "frost"},
]
SEED_CHAIN_C = [
    {"subject": "Dravy", "relation": "operates", "object": "Plin"},
    {"subject": "Plin", "relation": "routes-to", "object": "Belro"},
    {"subject": "Belro", "relation": "controls", "object": "Ryvok"},
    {"subject": "Ryvok", "relation": "produces", "object": "chime"},
]
# chain D: untouched by every intervention/correction below, so its
# combination gold is stable across the whole protocol (both epochs).
SEED_CHAIN_D = [
    {"subject": "Rhoka", "relation": "controls", "object": "Tolvo"},
    {"subject": "Tolvo", "relation": "produces", "object": "pulse"},
]
NEW_CHAIN = [
    {"subject": "Mavok", "relation": "operates", "object": "Rinpo"},
    {"subject": "Rinpo", "relation": "routes-to", "object": "Drel"},
    {"subject": "Drel", "relation": "controls", "object": "Crat"},
    {"subject": "Crat", "relation": "produces", "object": "mist"},
]


def _predict_questions(rels):
    """Paraphrase-battery questions for the predict capability. `expect` names
    the relation the answer must come from, so scoring can probe with the
    fact's OWN tokens ("X relation" + the reordered paraphrase "relation X")
    — the honest probe for a bag-of-words stub encoder; full re-wordings of
    the question text are kept for the report."""
    out = []
    for i, r in enumerate(rels):
        q = _q(
            f"q_pred_{i:02d}", "predict",
            f"{r['subject']} {r['relation']} what?",
            f"{r['relation']} - what comes after {r['subject']}?",
            r["object"])
        q["expect"] = {"subject": r["subject"], "relation": r["relation"],
                       "object": r["object"]}
        q["probe_surfaces"] = [f"{r['subject']} {r['relation']}",
                               f"{r['relation']} {r['subject']}"]
        out.append(q)
    return out


DEFAULT_DISCONNECT_BATTERY = {
    "battery_id": "disconnect-tiny",
    "seed_relations": (SEED_CHAIN_A + SEED_CHAIN_B + SEED_CHAIN_C + SEED_CHAIN_D),
    "questions": _predict_questions(
        SEED_CHAIN_A + SEED_CHAIN_B + SEED_CHAIN_C + SEED_CHAIN_D),
    "new_relations": NEW_CHAIN,
    "combinations": [
        # both on intervention-stable chains: one gold serves the retest epoch
        # AND the post-restart epoch (no sequencing artifact).
        {"chain": SEED_CHAIN_B, "subject": "Sorni", "relation": "reach",
         "gold": "frost", "start": "Sorni"},
        {"chain": SEED_CHAIN_D, "subject": "Rhoka", "relation": "reach",
         "gold": "pulse", "start": "Rhoka"},
    ],
    "corrections": [
        {"subject": "Zor", "relation": "produces", "object": "vapor",
         "replaces": {"subject": "Zor", "relation": "produces", "object": "heat"},
         "correction_text": "Reality: Zor produces vapor, not heat."},
    ],
    "interventions": [
        # (a) pure simulation: reroute into a relay whose downstream IS known —
        #     the prediction should be right WITHOUT any correction.
        {"id": "int_sim", "start": "Neral", "gold": "frost",
         "question": "Tav is rerouted to relay Sorni. What does Neral reach now?",
         "observation": {"subject": "Tav", "relation": "routes-to", "object": "Sorni"},
         "replaces": {"subject": "Tav", "relation": "routes-to", "object": "Kimo"},
         "correction": []},
        # (b) knowledge gap + self-correct: reroute into an UNKNOWN relay; the
        #     prediction is WRONG until reality supplies the downstream facts,
        #     which enter as plastic updates and the prediction is re-scored.
        {"id": "int_gap", "start": "Dravy", "gold": "glow",
         "question": "Plin is rerouted to relay Glova. What does Dravy reach now?",
         "observation": {"subject": "Plin", "relation": "routes-to", "object": "Glova"},
         "replaces": {"subject": "Plin", "relation": "routes-to", "object": "Belro"},
         "correction": [
             {"subject": "Glova", "relation": "controls", "object": "Huvla"},
             {"subject": "Huvla", "relation": "produces", "object": "glow"}]},
    ],
}


# --------------------------------------------------------------------------- #
# Battery soundness gate: gold answers re-derived from the battery's own graph
# --------------------------------------------------------------------------- #

def _gold_graph(battery: Dict) -> Dict[str, List[Tuple[str, str]]]:
    g: Dict[str, List[Tuple[str, str]]] = {}
    facts: List[Dict] = []
    facts += battery.get("seed_relations", [])
    facts += battery.get("new_relations", [])
    facts += battery.get("corrections", [])
    for interv in battery.get("interventions", []):
        facts.append(interv["observation"])
        facts += interv.get("correction", [])
    for f in facts:
        g.setdefault(f["subject"], []).append((f["relation"], f["object"]))
    return g


def _walk_graph(g: Dict[str, List[Tuple[str, str]]], start: str, override=None,
                max_hops: int = 8) -> List[str]:
    """Ground-truth walk over the gold graph; `override` replaces an edge
    ((subject, relation) -> new object) BEFORE walking (intervention semantics)."""
    gg = {k: list(v) for k, v in g.items()}
    if override:
        (s, rel), new_obj = override
        gg[s] = [(r, o) for (r, o) in gg.get(s, []) if r != rel] + [(rel, new_obj)]
    node, path, seen = start, [start], {start}
    for _ in range(max_hops):
        if node not in gg or not gg[node]:
            break
        _, nxt = gg[node][0]
        if nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        node = nxt
    return path


def verify_battery(battery: Dict) -> List[str]:
    """Re-derive every battery gold answer from the battery's own fact graph.
    Empty list = sound. This is the no-fabricated-numbers gate."""
    failures: List[str] = []
    g = _gold_graph(battery)
    for c in battery.get("combinations", []):
        start = c.get("start") or (c["chain"][0]["subject"] if c.get("chain") else None)
        if start is None:
            continue
        path = _walk_graph(g, start)
        if path[-1] != c["gold"]:
            failures.append(f"combination {c['subject']} reach: gold={c['gold']!r} "
                            f"but graph says {path[-1]!r} (path={path})")
    for iv in battery.get("interventions", []):
        override = ((iv["observation"]["subject"], iv["observation"]["relation"]),
                    iv["observation"]["object"])
        path = _walk_graph(g, iv["start"], override=override)
        if path[-1] != iv["gold"]:
            failures.append(f"intervention {iv.get('id')}: gold={iv['gold']!r} "
                            f"but graph says {path[-1]!r} (path={path})")
    return failures


# --------------------------------------------------------------------------- #
# Learning + scoring (identical code path for both lanes)
# --------------------------------------------------------------------------- #

def fact_text(rel: Dict) -> str:
    return f"{rel['subject']} {rel['relation']} {rel['object']}"


def rkey(d: Dict) -> str:
    """relation_key from a {subject, relation, object} dict."""
    return relation_key(d["subject"], d["relation"], d["object"])


EPISODIC_KIND = {"lesson": "lesson", "intervention": "observation",
                 "correction": "correction"}   # episodic store's valid kinds


def learn_relation(lane, rel: Dict, kind: str = "lesson", teacher: str = None,
                   n_steps: int = 2, replaces_key: Optional[str] = None) -> str:
    """One relation enters the pathway. Provenance: the disconnected lane learns
    through channel='direct' (it never had a transcript); the host lane through
    channel='transcript' (teacher lesson)."""
    key = relation_key(rel["subject"], rel["relation"], rel["object"])
    ids = lane.vocab.encode(fact_text(rel))
    lane.rmem.learn_relation(
        key, ids.flatten().tolist(), learn_fn=lane.learn_fn,
        teacher=teacher or ("scripted-teacher" if lane.name == "host"
                            else "direct-observation"),
        kind=EPISODIC_KIND.get(kind, "observation"), channel=lane.channel,
        n_steps=n_steps,
        tags={"kind": kind, "answer": rel["object"], "surface": fact_text(rel),
              **({"replaces": replaces_key} if replaces_key else {})})
    if replaces_key and replaces_key in lane.rmem.records:
        lane.rmem.records[replaces_key].superseded_by = key
    return key


def nearest_under_blend(lane, probe: torch.Tensor, blend: Optional[float], k: int = 1):
    """Cosine-nearest indexed relations whose LIVE strength under `blend`
    exceeds the threshold (blend=None -> the connection's own blend). This is
    how one answer gets a decidable per-bucket provenance."""
    q = torch.nn.functional.normalize(probe.detach().float().flatten(), dim=0)
    scored = []
    for key, rec in lane.rmem.records.items():
        if rec.superseded_by is not None:
            continue
        if not lane.rmem.present(key, blend):
            continue
        sim = float(torch.dot(q, rec.pre_probe).item())
        scored.append((rec, sim))
    scored.sort(key=lambda p: (-p[1], p[0].registered_at_seq, p[0].key))
    return scored[:k]


def answer_question(lane, text: str, gold: str) -> Dict:
    """One question, one answer, provenance decided by ablation-style readings:
    PLASTIC = top nearest under blend=1.0 matches gold; CONSOLIDATED = under
    blend=0.0; RETRIEVED = episodic store only (NEVER owned); NONE otherwise
    (full-blend hits are recorded in full_hit so a blended-only answer is
    visible but is not attributed to a single owned bucket)."""
    ids = lane.vocab.encode(text)
    pre, _ = lane.make_probe(ids)
    row = {"prompt": text, "gold": gold}
    for label, blend in ((PLASTIC, 1.0), (CONSOLIDATED, 0.0)):
        hits = nearest_under_blend(lane, pre, blend, k=1)
        obj = parse_key(hits[0][0].key)[2] if hits else ""
        row[f"{label.lower()}_hit"] = bool(hits) and answer_matches(obj, gold)
        row[f"{label.lower()}_answer"] = obj
    hits_full = nearest_under_blend(lane, pre, None, k=1)
    obj_full = parse_key(hits_full[0][0].key)[2] if hits_full else ""
    row["full_hit"] = bool(hits_full) and answer_matches(obj_full, gold)
    row["full_answer"] = obj_full
    r = lane.episodic.retrieve(text, k=1)
    retrieved_ok = False
    if r:
        ev, score = r[0]
        got = ev.tags.get("answer") or ""
        retrieved_ok = bool(got) and answer_matches(got, gold)
        row["retrieval_score"] = round(float(score), 4)
    row["retrieved_hit"] = bool(retrieved_ok)
    correct_buckets = [b for b in STRONGEST_FIRST if row[f"{b.lower()}_hit"]]
    if correct_buckets:
        row["provenance"] = correct_buckets[0]
    elif retrieved_ok:
        row["provenance"] = RETRIEVED
    else:
        row["provenance"] = NONE_PROV
    row["correct"] = row["provenance"] != NONE_PROV or row["full_hit"]
    row["owned"] = row["provenance"] in (PLASTIC, CONSOLIDATED)
    row["owned_or_full"] = row["owned"] or (row["full_hit"] and not retrieved_ok)
    return row


def walk_internal_state(lane, start: str, blend: Optional[float] = None,
                        max_hops: int = 6) -> Tuple[str, List[str]]:
    """SIMULATION over the lane's OWN world state: from `start`, follow the
    strongest-held non-superseded edge in the relational index (live plastic
    strength under `blend`), and keep walking. Reads ONLY the lane's rmem index
    + plastic strengths — never the battery's graph, never the host."""
    node = start
    path: List[str] = []
    seen = {start}
    for _ in range(max_hops):
        recs = [r for r in lane.rmem.lookup(subject=node)
                if r.superseded_by is None and lane.rmem.present(r.key, blend)]
        if not recs:
            break
        rec = sorted(recs, key=lambda r: (-lane.rmem.strength(r.key, blend),
                                          r.registered_at_seq))[0]
        path.append(rec.key)
        _, _, nxt = parse_key(rec.key)
        if nxt in seen:
            break
        seen.add(nxt)
        node = nxt
    return node, path


def edge_bucket(lane, key: str) -> str:
    p = lane.rmem.present(key, 1.0)
    c = lane.rmem.present(key, 0.0)
    if p and c:
        return MIXED
    if p:
        return PLASTIC
    if c:
        return CONSOLIDATED
    return NONE_PROV


def path_provenance(lane, path: List[str]) -> str:
    """Weakest link across the walked path: an edge only plastic -> PLASTIC path;
    only consolidated -> CONSOLIDATED; mixed buckets or mixed edges -> MIXED."""
    if not path:
        return NONE_PROV
    buckets = [edge_bucket(lane, k) for k in path]
    if all(b == PLASTIC for b in buckets):
        return PLASTIC
    if all(b == CONSOLIDATED for b in buckets):
        return CONSOLIDATED
    if all(b in (PLASTIC, CONSOLIDATED) for b in buckets):
        return MIXED
    return NONE_PROV


def simulate_intervention(lane, interv: Dict, n_steps: int = 2,
                          retention_fraction: float = 0.5) -> Dict:
    """Protocol step 5, one intervention:
      1. the intervention OBSERVATION enters as a plastic update (superseding
         the replaced edge) — reality happened, the system watched;
      2. PREDICT by walking its internal state from `start`;
      3. score against the GRAPH ground truth (`gold` — verified by
         verify_battery, never fabricated);
      4. SELF-CORRECT: if the prediction was wrong and the battery supplies
         correction facts, they enter as plastic updates and the prediction is
         re-scored. Old unrelated facts must survive (learn-new-without-
         destroying-old)."""
    row: Dict = {"id": interv.get("id"), "question": interv.get("question"),
                 "start": interv["start"], "gold": interv["gold"]}
    obs_key = learn_relation(lane, interv["observation"], kind="intervention",
                             teacher="reality", n_steps=n_steps,
                             replaces_key=rkey(interv["replaces"]) if interv.get("replaces") else None)

    def _predict() -> Tuple[str, List[str]]:
        term, path = walk_internal_state(lane, interv["start"])
        return term, path

    pre_term, pre_path = _predict()
    row["pre_prediction"] = pre_term
    row["pre_path"] = list(pre_path)
    row["pre_correct"] = bool(pre_term == interv["gold"])
    row["path_provenance"] = path_provenance(lane, pre_path)
    row["observation_key"] = obs_key
    replaced_key = (rkey(interv["replaces"])
                    if interv.get("replaces") else None)
    # The facts the pre-correction prediction relied on (its walked path, minus
    # the observation itself and the replaced edge) must survive any correction.
    relied = [k for k in pre_path if k != obs_key and k != replaced_key]

    corrections = interv.get("correction") or []
    applied = False
    if (not row["pre_correct"]) and corrections:
        before = {rkey(f): lane.rmem.strength(rkey(f)) for f in corrections}
        for f in corrections:
            learn_relation(lane, f, kind="correction", teacher="reality",
                           n_steps=n_steps)
        applied = True
        row["old_strengths_before_correction"] = {
            k: (None if v is None else round(v, 6)) for k, v in before.items()}
    row["correction_applied"] = applied

    post_term, post_path = _predict()
    row["post_prediction"] = post_term
    row["post_path"] = list(post_path)
    row["post_correct"] = bool(post_term == interv["gold"])
    row["provenance"] = path_provenance(lane, post_path)

    old_intact = all(lane.rmem.strength(k) is not None
                     and lane.rmem.strength(k) > lane.rmem.strength_threshold
                     for k in relied)
    row["relied_on_edges"] = relied
    row["old_facts_intact"] = bool(old_intact)
    row["self_correct_pass"] = bool(row["pre_correct"] or
                                    (applied and row["post_correct"]
                                     and row["old_facts_intact"]))
    return row


def score_correction(lane, corr: Dict, n_steps: int = 2,
                     retention_fraction: float = 0.5) -> Dict:
    """The battery's `corrections` section: a replaced fact — new one must be
    learned, the old one must not be destroyed (superseded, retained)."""
    old_key = rkey(corr["replaces"])
    old_before = lane.rmem.strength(old_key)
    new_key = learn_relation(lane, corr, kind="correction", teacher="reality",
                             n_steps=n_steps, replaces_key=old_key)
    old_after = lane.rmem.strength(old_key)
    new_ok = lane.rmem.present(new_key)
    old_kept = bool(old_before is not None and old_after is not None
                    and old_after >= retention_fraction * max(old_before, 1e-12))
    ans = answer_question(lane, f"{corr['subject']} {corr['relation']}",
                          corr["object"])
    return {"new_key": new_key, "replaces": old_key,
            "old_strength_before": None if old_before is None else round(old_before, 6),
            "old_strength_after": None if old_after is None else round(old_after, 6),
            "new_learned": bool(new_ok), "old_retained": old_kept,
            "answer": ans, "self_correct_pass": bool(new_ok and old_kept)}


def rescore_correction(lane, corr: Dict,
                       retention_fraction: float = 0.5) -> Dict:
    """Post-restart re-verification of one correction row: score the state,
    never re-apply anything."""
    old_key = rkey(corr["replaces"])
    new_key = rkey(corr)
    old_after = lane.rmem.strength(old_key)
    new_ok = lane.rmem.present(new_key)
    rec = lane.rmem.records.get(old_key)
    old_kept = bool(rec is not None and old_after is not None
                    and old_after > lane.rmem.strength_threshold)
    ans = answer_question(lane, f"{corr['subject']} {corr['relation']}",
                          corr["object"])
    return {"new_key": new_key, "replaces": old_key,
            "old_strength_after": None if old_after is None else round(old_after, 6),
            "new_learned": bool(new_ok), "old_retained": old_kept,
            "answer": ans, "self_correct_pass": bool(new_ok and old_kept)}


def frac(rows: List[Dict], flag: str = "pass") -> Tuple[float, Dict]:
    rows = list(rows)
    if not rows:
        return 0.0, {"n": 0, "pass": 0}
    n_pass = sum(1 for r in rows if r.get(flag, r.get("correct", False)))
    return n_pass / len(rows), {"n": len(rows), "pass": n_pass}


def measure_capabilities(lane, battery: Dict, seed_keys: List[str],
                         n_steps: int = 2,
                         retention_fraction: float = 0.5,
                         apply_new_and_interventions: bool = True) -> Dict:
    """The owner's five capabilities on ONE lane state, identically for both
    lanes (the register's runs must be measured the same way).

    apply_new_and_interventions=True (the WITH-host baseline / whole-lane
    measurement): new_relations, interventions and corrections LEARN as part of
    the measurement. False (post-restart re-verification): score-only, the
    state is never mutated by measurement."""
    # recognise: seed-relation presence, bucketed.
    recognise_rows = []
    for key in seed_keys:
        s, r, o = parse_key(key)
        rec = lane.rmem.records.get(key)
        row = {"key": key,
               "plastic_active": bool(lane.rmem.present(key, 1.0)),
               "consolidated_active": bool(lane.rmem.present(key, 0.0)),
               "superseded": rec.superseded_by if rec else None}
        buckets = [b for b, ok in ((PLASTIC, row["plastic_active"]),
                                   (CONSOLIDATED, row["consolidated_active"])) if ok]
        if rec is not None and rec.superseded_by is None and buckets:
            row["provenance"] = buckets[0] if len(buckets) == 1 else MIXED
        elif rec is not None and any(
                answer_matches(ev.tags.get("answer") or "", o)
                for ev in lane.episodic.events()):
            row["provenance"] = RETRIEVED
        else:
            row["provenance"] = NONE_PROV
        row["owned"] = row["provenance"] in (PLASTIC, CONSOLIDATED, MIXED)
        row["recognised"] = bool(row["owned"])
        recognise_rows.append(row)

    # predict: the paraphrase battery (question surfaces), bucketed answers.
    # probe_surfaces (when the battery supplies them) are the fact's own tokens
    # in two orders — the honest probe for a bag-of-words encoder; questions
    # without them fall back to their raw prompt text.
    predict_rows = []
    for q in battery.get("questions", []):
        if q.get("capability") not in ("recognise", "predict"):
            continue
        surfaces = list(q.get("probe_surfaces")) or \
            ([q["prompt"]] + list(q.get("paraphrases", [])))
        gold = (q.get("expect") or {}).get("object", q["gold"])
        for surface in surfaces:
            ans = answer_question(lane, surface, gold)
            ans.update({"qid": q["qid"], "surface": surface, "gold": gold,
                        "question_text": q["prompt"]})
            predict_rows.append(ans)

    # combine: internal-state walk (the lane's own belief graph), bucketed path.
    combine_rows = []
    for c in battery.get("combinations", []):
        term, path = walk_internal_state(lane, c.get("start") or c["chain"][0]["subject"])
        prov = path_provenance(lane, path)
        combine_rows.append({"query": f"{c['subject']} {c['relation']}",
                             "gold": c["gold"], "terminal": term, "path": path,
                             "provenance": prov,
                             "owned": prov in (PLASTIC, CONSOLIDATED, MIXED),
                             "correct": bool(term == c["gold"] and prov in
                                             (PLASTIC, CONSOLIDATED, MIXED))})

    # learn_new + self_correct are measured by their own phases on the
    # disconnected lane (with learning); a whole-lane measurement (the WITH-host
    # baseline) runs them here identically.
    learn_rows = []
    for rel in battery.get("new_relations", []):
        key = rkey(rel)
        if apply_new_and_interventions:
            learn_relation(lane, rel, kind="lesson", teacher="battery_new",
                           n_steps=n_steps)
        ans = answer_question(lane, f"{rel['subject']} {rel['relation']}",
                              rel["object"])
        learn_rows.append({"key": key, "present": bool(lane.rmem.present(key)),
                           "answer": ans,
                           "pass": bool(lane.rmem.present(key) and ans["owned"])})

    if apply_new_and_interventions:
        self_rows = [simulate_intervention(lane, iv, n_steps=n_steps,
                                           retention_fraction=retention_fraction)
                     for iv in battery.get("interventions", [])]
        self_rows += [score_correction(lane, c, n_steps=n_steps,
                                       retention_fraction=retention_fraction)
                      for c in battery.get("corrections", [])]
    else:
        # score-only re-verification: re-walk each intervention on the restored
        # (already-corrected) state; the corrected prediction must still hold.
        self_rows = []
        for iv in battery.get("interventions", []):
            term, path = walk_internal_state(lane, iv["start"])
            replaced_key = (rkey(iv["replaces"])
                            if iv.get("replaces") else None)
            obs_key = rkey(iv["observation"])
            relied = [k for k in path if k not in (obs_key, replaced_key)]
            old_intact = all(lane.rmem.strength(k) is not None
                             and lane.rmem.strength(k) > lane.rmem.strength_threshold
                             for k in relied)
            self_rows.append({"id": iv.get("id"), "post_prediction": term,
                              "gold": iv["gold"], "post_correct": bool(term == iv["gold"]),
                              "old_facts_intact": bool(old_intact),
                              "provenance": path_provenance(lane, path),
                              "self_correct_pass": bool(term == iv["gold"]
                                                        and old_intact),
                              "rescored_only": True})
        self_rows += [rescore_correction(lane, c, retention_fraction=retention_fraction)
                      for c in battery.get("corrections", [])]

    scores, details = {}, {}
    scores["recognise"], details["recognise"] = frac(recognise_rows, "recognised")
    scores["predict"], details["predict"] = frac(predict_rows, "owned")
    scores["combine"], details["combine"] = frac(combine_rows, "correct")
    scores["learn_new"], details["learn_new"] = frac(learn_rows, "pass")
    scores["self_correct"], details["self_correct"] = frac(self_rows, "self_correct_pass")
    return {"scores": scores, "details": details,
            "rows": {"recognise": recognise_rows, "predict": predict_rows,
                     "combine": combine_rows, "learn_new": learn_rows,
                     "self_correct": self_rows}}


def provenance_histogram(rows: List[Dict]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for r in rows:
        label = r.get("provenance", r.get("answer", {}).get("provenance", NONE_PROV))
        if isinstance(label, dict):                     # correction rows
            label = label.get("provenance", NONE_PROV)
        hist[label] = hist.get(label, 0) + 1
    return hist


# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #

def _jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.shape
    return obj


def _strict(obj):
    """Strict-JSON sanitizer: non-finite floats become tagged strings (json's
    default() is never consulted for floats), tensors become shapes."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf"
        if math.isnan(obj):
            return "nan"
        return round(obj, 6)
    if isinstance(obj, torch.Tensor):
        return obj.shape
    if isinstance(obj, dict):
        return {str(k): _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    return obj


def _dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_strict(obj), f, indent=1, default=_jsonable, allow_nan=False,
                  sort_keys=True)


class DisconnectExperiment:
    """The owner's 6-step protocol as explicit phases. `--phase all` runs
    everything in one process; split-phase CLI resumes from disk; the restart
    phase ALWAYS round-trips through brand-new objects."""

    def __init__(self, battery: Dict, run_dir: str, seed: int = 0,
                 n_learn_steps: int = 2, consolidation_interval: int = 8,
                 retention_fraction: float = 0.5,
                 teacher: Optional[object] = None,
                 teacher_name: str = "scripted-teacher",
                 foundation: str = "stub",
                 live: bool = False,
                 prebuilt_iface: Optional[object] = None,
                 foundation_loader: Optional[Callable] = None,
                 vocab_size: int = 128,
                 verbose: bool = True):
        failures = verify_battery(battery)
        if failures:
            raise ValueError("battery ground truth UNSOUND (no fabricated runs): "
                             + "; ".join(failures))
        self.battery = battery
        self.run_dir = run_dir
        self.seed = seed
        self.n_learn_steps = n_learn_steps
        self.consolidation_interval = consolidation_interval
        self.retention_fraction = retention_fraction
        self.teacher = teacher
        self.teacher_name = teacher_name
        self.foundation = foundation
        self.live = bool(live)
        self.prebuilt_iface = prebuilt_iface
        self.foundation_loader = foundation_loader
        self.vocab_size = vocab_size
        self.verbose = verbose
        self.host: Optional[LaneHost] = None
        self.disc: Optional[LaneDisconnected] = None
        self.protocol: Dict = {"battery_id": battery.get("battery_id", "?"),
                               "seed": seed, "completed_phases": [],
                               "events": {}, "substrate": None}
        self.reports: Dict[str, Dict] = {}
        proto_path = os.path.join(run_dir, "protocol.json")
        if os.path.exists(proto_path):                 # split-phase resume
            with open(proto_path, "r", encoding="utf-8") as f:
                self.protocol = json.load(f)
        for p in PHASES:                               # prior phase reports on disk
            pp = os.path.join(run_dir, f"phase_{p}.json")
            if os.path.exists(pp):
                with open(pp, "r", encoding="utf-8") as f:
                    self.reports[p] = json.load(f)

    # -- plumbing ------------------------------------------------------------- #

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[disconnect] {msg}")

    def _phase_path(self, phase: str) -> str:
        return os.path.join(self.run_dir, f"phase_{phase}.json")

    def _protocol_path(self) -> str:
        return os.path.join(self.run_dir, "protocol.json")

    def _require(self, *prereqs: str) -> None:
        done = set(self.protocol["completed_phases"])
        missing = [p for p in prereqs if p not in done]
        if missing:
            raise SequencingError(
                f"phase requires {list(prereqs)} but {missing} have not run — "
                f"NOT-RUN (owner sequencing law)")

    def _mark_done(self, phase: str, report: Dict) -> None:
        report["phase"] = phase
        report["status"] = "RUN"
        self.reports[phase] = report
        _dump_json(report, self._phase_path(phase))
        if phase not in self.protocol["completed_phases"]:
            self.protocol["completed_phases"].append(phase)
        self._save_protocol()
        self.log(f"phase {phase}: RUN -> {self._phase_path(phase)}")

    def _save_protocol(self) -> None:
        if self.disc is not None:
            self.protocol["events"]["structural_disconnection"] = \
                self.disc.assert_host_disconnected()
        if self.host is not None:
            self.protocol["monitor_host"] = self.host.monitor.snapshot()
        _dump_json(self.protocol, self._protocol_path())

    def _save_lanes(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        if self.host is not None and self.host.iface is not None:
            from aion.memory.state import PlasticStateSnapshot
            snap = PlasticStateSnapshot.capture(
                self.host.iface, meta={"lane": "host", "seed": self.seed})
            payload = {"plastic": snap.to_payload(),
                       "probes": self.host.rmem.save_probes(),
                       "index": self.host.rmem.to_dict(),
                       "episodic": self.host.episodic.to_dict()}
            torch.save(payload, os.path.join(self.run_dir, "state_host.pt"))
        if self.disc is not None:
            torch.save(self.disc.save_payload(),
                       os.path.join(self.run_dir, "state_disc.pt"))

    def _ensure_lanes(self) -> None:
        """Build fresh lanes; restore prior state from disk when this is a
        split-phase resume (fresh objects, disk-only round trip)."""
        if self.host is not None and self.disc is not None:
            return
        if self.live:
            if os.environ.get(LIVE_GATE_ENV) != "1":
                raise LiveGateError(
                    f"real-foundation run is owner-gated: export "
                    f"{LIVE_GATE_ENV}=1 (Phase 2 landed + owner go) before --live")
            if self.foundation != "qwen":
                raise LiveGateError("--foundation qwen is the only live substrate")
            if self.prebuilt_iface is None and self.foundation_loader is None:
                self.foundation_loader = default_live_loader
        host_seed = self.seed
        self.host = LaneHost(seed=host_seed, monitor=HostMonitor("host-control"),
                             prebuilt_iface=self.prebuilt_iface,
                             foundation_loader=self.foundation_loader,
                             vocab_size=self.vocab_size)
        self.disc = LaneDisconnected(seed=host_seed, hidden_size=self.host.hidden_dim,
                                     vocab_size=self.vocab_size,
                                     monitor=HostMonitor("disconnected"))
        self.protocol["substrate"] = {"host": self.host.substrate,
                                      "disconnected": "direct-feed-observations"}
        host_state = os.path.join(self.run_dir, "state_host.pt")
        disc_state = os.path.join(self.run_dir, "state_disc.pt")
        if os.path.exists(disc_state):
            payload = torch.load(disc_state, weights_only=False)
            exact, diffs = self.disc.load_payload(payload)
            self.protocol["events"]["disc_resume_exact"] = {"exact": exact,
                                                            "diffs": diffs[:5]}
        if os.path.exists(host_state):
            payload = torch.load(host_state, weights_only=False)
            from aion.memory.state import PlasticStateSnapshot
            snap = PlasticStateSnapshot.from_payload(payload["plastic"])
            snap.restore(self.host.iface)
            self.host.rmem = RelationalMemory.from_dict(
                payload["index"], conns=list(self.host.iface.conns),
                episodic=self.host.episodic, probes=payload["probes"],
                make_probe=self.host.make_probe)      # REAL probes survive resume
            self.host.episodic = EpisodicStore.from_dict(
                payload["episodic"], clock=lambda: 0.0)
            self.host.rmem.episodic = self.host.episodic

    # ------------------------------------------------------------------ #
    # STEP 0 — TEACH (+ the WITH-host control run: the register denominator)
    # ------------------------------------------------------------------ #

    def phase_teach(self) -> Dict:
        self._ensure_lanes()
        seed_rels = self.battery.get("seed_relations", [])
        host_keys, disc_keys = [], []
        for rel in seed_rels:
            host_keys.append(learn_relation(self.host, rel, kind="lesson",
                                            teacher=self.teacher_name,
                                            n_steps=self.n_learn_steps))
            disc_keys.append(learn_relation(self.disc, rel, kind="lesson",
                                            n_steps=self.n_learn_steps))
        cons_h = self.host.ledger.consolidate_all(step=self.n_learn_steps * len(seed_rels))
        cons_d = self.disc.ledger.consolidate_all(step=self.n_learn_steps * len(seed_rels))

        # WITH-host CONTROL RUN: the full capability battery on the host lane —
        # the substrate's capability, measured identically to the disconnected
        # lane (same battery, same scoring functions).
        host_meas = measure_capabilities(self.host, self.battery, host_keys,
                                         n_steps=self.n_learn_steps,
                                         retention_fraction=self.retention_fraction)
        self.host.ledger.consolidate_all(step=self.n_learn_steps * (len(seed_rels) + 4))

        report = {
            "phase": "teach", "status": "RUN", "step": 0,
            "seed_relations": len(seed_rels),
            "disconnected_seed_keys": list(disc_keys),
            "host": {"substrate": self.host.substrate,
                     "scores": host_meas["scores"],
                     "monitor": self.host.monitor.snapshot(),
                     "consolidation_invariant_ok": cons_h["invariant_ok"],
                     "indexed_relations": len(self.host.rmem.records)},
            "disconnected": {"structural": self.disc.assert_host_disconnected(),
                             "consolidation_invariant_ok": cons_d["invariant_ok"],
                             "indexed_relations": len(self.disc.rmem.records)},
            "note": ("WITH-host control run measured here (register denominator). "
                     "The disconnected lane has learned the same seeds through "
                     "direct observation ONLY; steps 1-6 follow."),
        }
        self._host_capability_baseline = host_meas
        self._disc_seed_keys = disc_keys
        self._save_lanes()
        self._mark_done("teach", report)
        return report

    # ------------------------------------------------------------------ #
    # STEP 1 — DISCONNECT TEACHER
    # ------------------------------------------------------------------ #

    def phase_disconnect_teacher(self) -> Dict:
        self._require("teach")
        self._ensure_lanes()
        # The system under test never had a teacher channel: the gate is
        # created CLOSED and any teacher call would raise. Formally ratified:
        teach_calls = self.disc.monitor.counts["teacher_call"]
        self.disc.monitor.close("protocol step 1: DISCONNECT TEACHER")
        self.protocol["events"]["step1_teacher_disconnected"] = {
            "teacher_gate": "CLOSED",
            "teacher_calls_before": teach_calls,
            "teacher_calls_total": self.disc.monitor.counts["teacher_call"],
            "note": "no external teaching or answer assistance from this step onward"}
        report = {"phase": "disconnect_teacher", "status": "RUN", "step": 1,
                  "teacher_gate": "CLOSED",
                  "teacher_calls_in_lane_lifetime": teach_calls,
                  "evidence": self.disc.assert_host_disconnected()}
        self._mark_done("disconnect_teacher", report)
        return report

    # ------------------------------------------------------------------ #
    # STEP 2 — DISCONNECT HOST
    # ------------------------------------------------------------------ #

    def phase_disconnect_host(self) -> Dict:
        self._require("teach", "disconnect_teacher")
        self._ensure_lanes()
        self.disc.monitor.close("protocol step 2: DISCONNECT HOST")
        structural = self.disc.assert_host_disconnected()
        self.protocol["events"]["step2_host_disconnected"] = {
            "host_gate": "CLOSED", "structural": structural,
            "note": ("no Qwen hidden states, no logits, no inference API, no cached "
                     "model outputs from this step onward; direct observation only")}
        report = {"phase": "disconnect_host", "status": "RUN", "step": 2,
                  "host_gate": "CLOSED", "structural": structural}
        self._save_lanes()
        self._mark_done("disconnect_host", report)
        return report

    # ------------------------------------------------------------------ #
    # STEP 3 — RETEST EXISTING KNOWLEDGE
    # ------------------------------------------------------------------ #

    def phase_retest(self) -> Dict:
        self._require("teach", "disconnect_teacher", "disconnect_host")
        self._ensure_lanes()
        structural = self.disc.assert_host_disconnected()      # measured zero
        keys = getattr(self, "_disc_seed_keys", None) or \
            self.reports.get("teach", {}).get("disconnected_seed_keys") or \
            sorted(self.disc.rmem.records.keys())
        meas = measure_capabilities(self.disc, self.battery, keys,
                                    n_steps=self.n_learn_steps,
                                    retention_fraction=self.retention_fraction,
                                    apply_new_and_interventions=False)
        # keep only the retest capabilities (learn_new/self_correct are steps
        # 4-5 and are measured there, not here)
        retest_scores = {c: meas["scores"][c] for c in ("recognise", "predict", "combine")}
        self._disc_retest = meas
        report = {"phase": "retest", "status": "RUN", "step": 3,
                  "structural_zero_host_touches": structural,
                  "scores": retest_scores,
                  "rows": {c: meas["rows"][c] for c in retest_scores},
                  "provenance_histogram": {
                      c: provenance_histogram(meas["rows"][c]) for c in retest_scores},
                  "inventory": self.disc.rmem.inventory()}
        self._save_lanes()
        self._mark_done("retest", report)
        return report

    # ------------------------------------------------------------------ #
    # STEP 4 — INTRODUCE NEW OBSERVATIONS
    # ------------------------------------------------------------------ #

    def phase_observe(self) -> Dict:
        self._require("teach", "disconnect_teacher", "disconnect_host", "retest")
        self._ensure_lanes()
        structural = self.disc.assert_host_disconnected()
        learn_rows = []
        for rel in self.battery.get("new_relations", []):
            key = learn_relation(self.disc, rel, kind="lesson",
                                 teacher="direct-observation",
                                 n_steps=self.n_learn_steps)
            ans = answer_question(self.disc,
                                  f"{rel['subject']} {rel['relation']}",
                                  rel["object"])
            learn_rows.append({"key": key, "present": bool(self.disc.rmem.present(key)),
                               "answer": ans,
                               "pass": bool(self.disc.rmem.present(key)
                                            and ans["owned"])})
        cons = self.disc.ledger.consolidate_all(step=self.n_learn_steps * 10)
        n = max(1, len(learn_rows))
        score = sum(1 for r in learn_rows if r["pass"]) / n
        self._disc_learn_rows = learn_rows
        report = {"phase": "observe", "status": "RUN", "step": 4,
                  "structural_zero_host_touches": structural,
                  "score": score, "rows": learn_rows,
                  "new_relations": len(learn_rows),
                  "provenance_histogram": provenance_histogram(
                      [r["answer"] for r in learn_rows]),
                  "consolidation_invariant_ok": cons["invariant_ok"],
                  "note": "new facts entered via the direct-feed encoder only "
                          "(no LLM assistance)"}
        self._save_lanes()
        self._mark_done("observe", report)
        return report

    # ------------------------------------------------------------------ #
    # STEP 5 — RUN OWN SIMULATIONS
    # ------------------------------------------------------------------ #

    def phase_simulate(self) -> Dict:
        self._require("teach", "disconnect_teacher", "disconnect_host", "retest",
                      "observe")
        self._ensure_lanes()
        structural = self.disc.assert_host_disconnected()
        sim_rows = [simulate_intervention(self.disc, iv, n_steps=self.n_learn_steps,
                                          retention_fraction=self.retention_fraction)
                    for iv in self.battery.get("interventions", [])]
        corr_rows = [score_correction(self.disc, c, n_steps=self.n_learn_steps,
                                      retention_fraction=self.retention_fraction)
                     for c in self.battery.get("corrections", [])]
        self_rows = sim_rows + corr_rows
        n = max(1, len(self_rows))
        score = sum(1 for r in self_rows if r["self_correct_pass"]) / n
        self._disc_sim_rows = self_rows
        report = {"phase": "simulate", "status": "RUN", "step": 5,
                  "structural_zero_host_touches": structural,
                  "score": score, "rows": self_rows,
                  "interventions": len(sim_rows), "corrections": len(corr_rows),
                  "note": "predictions walked the lane's own belief graph "
                          "(rmem index + live plastic strengths) and were "
                          "scored against the graph ground truth; wrong "
                          "predictions were corrected by plastic updates and "
                          "re-scored"}
        self._save_lanes()
        self._mark_done("simulate", report)
        return report

    # ------------------------------------------------------------------ #
    # STEP 6 — RESTART (bitwise round-trip, then re-verify survival)
    # ------------------------------------------------------------------ #

    def phase_restart(self) -> Dict:
        self._require("teach", "disconnect_teacher", "disconnect_host", "retest",
                      "observe", "simulate")
        self._ensure_lanes()
        pre_payload_path = os.path.join(self.run_dir, "state_disc.pt")
        if not os.path.exists(pre_payload_path):        # in-process: save now
            self._save_lanes()
        payload = torch.load(pre_payload_path, weights_only=False)
        pre_snapshot = {k: v.clone() for k, v in
                        conn_tensor_map(self.disc.conn).items()}
        pre_probes = self.disc.rmem.save_probes()
        del self.disc                                   # shutdown: the object is gone
        self.disc = None
        self.disc = LaneDisconnected(seed=self.seed,
                                     hidden_size=int(payload["hidden_dim"]),
                                     vocab_size=self.vocab_size)
        exact, diffs = self.disc.load_payload(payload)
        post_snapshot = conn_tensor_map(self.disc.conn)
        bitwise = {k: bool(torch.equal(pre_snapshot[k], post_snapshot[k].clone()))
                   for k in pre_snapshot}
        probes_exact = all(
            k in pre_probes and torch.equal(pre_probes[k]["pre"],
                                            self.disc.rmem.records[k].pre_probe)
            and torch.equal(pre_probes[k]["post"],
                            self.disc.rmem.records[k].post_probe)
            for k in self.disc.rmem.records)

        # RE-VERIFY survival on the restored state (the register's input).
        # Score-only: the restored state is never mutated by measurement.
        keys = getattr(self, "_disc_seed_keys", None) or \
            self.reports.get("teach", {}).get("disconnected_seed_keys") or \
            sorted(k for k, r in self.disc.rmem.records.items()
                   if r.meta.get("tags", {}).get("kind") == "lesson")
        re_meas = measure_capabilities(self.disc, self.battery, keys,
                                       n_steps=self.n_learn_steps,
                                       retention_fraction=self.retention_fraction,
                                       apply_new_and_interventions=False)
        post_scores = re_meas["scores"]
        self._disc_post_restart = post_scores
        self._disc_post_restart_rows = re_meas["rows"]
        structural = self.disc.assert_host_disconnected()   # zero after restart too
        report = {"phase": "restart", "status": "RUN", "step": 6,
                  "state_restored_exact": bool(exact),
                  "tensor_diffs": diffs[:10],
                  "bitwise_tensors": {k: v for k, v in bitwise.items()
                                      if not v} or "ALL-BITWISE-EQUAL",
                  "probes_exact": bool(probes_exact),
                  "structural_zero_host_touches": structural,
                  "post_restart_scores": {k: round(v, 6)
                                          for k, v in post_scores.items()},
                  "post_restart_provenance": {
                      c: provenance_histogram(re_meas["rows"][c])
                      for c in re_meas["rows"]}}
        self._save_lanes()
        self._mark_done("restart", report)
        return report

    # ------------------------------------------------------------------ #
    # THE REGISTER (owner-corrected metrics)
    # ------------------------------------------------------------------ #

    def build_register(self) -> Dict:
        self._require("teach", "disconnect_teacher", "disconnect_host", "retest",
                      "observe", "simulate", "restart")
        host_scores = getattr(self, "_host_capability_baseline", {}).get("scores")
        if host_scores is None:                         # split-phase resume
            host_scores = self.reports["teach"]["host"]["scores"]
        disc_post = getattr(self, "_disc_post_restart", None)
        post_hists = None
        if disc_post is None:                           # split-phase resume
            disc_post = self.reports["restart"]["post_restart_scores"]
            post_hists = self.reports["restart"]["post_restart_provenance"]
        disc_pre = {"recognise": self.reports["retest"]["scores"]["recognise"],
                    "predict": self.reports["retest"]["scores"]["predict"],
                    "combine": self.reports["retest"]["scores"]["combine"],
                    "learn_new": self.reports["observe"]["score"],
                    "self_correct": self.reports["simulate"]["score"]}
        ratios = survival_ratios(host_scores, disc_post)

        def fmt(r):
            if r is None:
                return None
            if isinstance(r, float) and math.isinf(r):
                return "inf"
            return round(float(r), 6)

        capabilities = {}
        for cap in CAPABILITIES:
            rows = getattr(self, "_disc_post_restart_rows", {}).get(cap)
            hist = provenance_histogram(rows) if rows is not None \
                else post_hists.get(cap, {})
            owned_n = sum(v for k, v in hist.items()
                          if k in (PLASTIC, CONSOLIDATED, MIXED))
            retrieved_n = hist.get(RETRIEVED, 0)
            ratio = ratios[cap]
            verdict = ("fully-survives" if ratio is not None
                       and math.isfinite(ratio) and ratio >= 1.0
                       else "degrades" if ratio is not None and math.isfinite(ratio)
                       else "impossible-gain(check-scoring)")
            capabilities[cap] = {
                "with_host": round(host_scores[cap], 6),
                "without_host_pre_restart": round(disc_pre[cap], 6),
                "without_host_post_restart": round(disc_post[cap], 6),
                "survival_ratio": fmt(ratio),
                "verdict": verdict,
                "provenance_histogram": hist,
                "owned_rows": owned_n,
                "retrieved_rows_NEVER_OWNED": retrieved_n,
            }
        substrate = self.protocol.get("substrate", {}).get("host", "unknown")
        degraded = "qwen3.5-4b" not in str(substrate).lower()
        register = {
            "register": "capability-survival (owner-corrected): BOTH runs reported; "
                        "ratio = without_host/with_host on the SAME battery; "
                        "post-restart disconnected scores; RETRIEVED never owned",
            "capabilities": capabilities,
            "confound_guard": {
                "substrate_with_host_run": substrate,
                "substrate_degraded": degraded,
                "caveat": ("CAVEAT: the WITH-host run used the CPU stub substrate, "
                           "not the frozen 4B foundation — survival ratios are "
                           "reported WITH this caveat" if degraded else None),
                "substrate_capability_with_host": {c: round(v, 6)
                                                   for c, v in host_scores.items()},
            },
            "structural_disconnection": self.protocol["events"].get(
                "structural_disconnection"),
        }
        report = {"phase": "register", "status": "RUN",
                  "register": register}
        self._mark_done("register", report)
        return report

    phase_register = build_register                  # phase-name alias for run()

    # ------------------------------------------------------------------ #
    # Master report
    # ------------------------------------------------------------------ #

    def master_report(self) -> Dict:
        rep = {"experiment": "disconnect", "battery_id": self.protocol["battery_id"],
               "seed": self.seed, "run_dir": self.run_dir,
               "substrate": self.protocol.get("substrate"),
               "completed_phases": list(self.protocol["completed_phases"]),
               "execution_mode": ("live-foundation" if self.live
                                  else "stub-build (REAL RUN NOT-RUN: gated on "
                                       "Phase 2 landing + owner)"),
               "phases": {p: self.reports[p] for p in PHASES if p in self.reports}}
        if "register" in self.reports:
            rep["register"] = self.reports["register"]["register"]
        return rep

    def run(self, phases: Tuple[str, ...] = PHASES) -> Dict:
        for phase in phases:
            getattr(self, f"phase_{phase}")()
        rep = self.master_report()
        _dump_json(rep, os.path.join(self.run_dir, "disconnect_report.json"))
        self.log(f"master report: {os.path.join(self.run_dir, 'disconnect_report.json')}")
        return rep

    def rescore_host_post(self) -> Dict:
        """Register-symmetry verification: re-score the WITH-host lane from its
        SAVED end-of-teach state (which already contains its interventions and
        corrections) in score-only mode — the same post-intervention epoch the
        disconnected lane is scored in. Writes host_rescored_post.json into the
        run dir. Live mode: rebuilds the frozen foundation (never trained,
        never used by the disconnected protocol)."""
        self._ensure_lanes()
        state_path = os.path.join(self.run_dir, "state_host.pt")
        payload = torch.load(state_path, weights_only=False)
        from aion.memory.state import PlasticStateSnapshot
        snap = PlasticStateSnapshot.from_payload(payload["plastic"])
        snap.restore(self.host.iface)
        self.host.rmem = RelationalMemory.from_dict(
            payload["index"], conns=list(self.host.iface.conns),
            episodic=self.host.episodic, probes=payload["probes"],
            make_probe=self.host.make_probe)
        self.host.episodic = EpisodicStore.from_dict(
            payload["episodic"], clock=lambda: 0.0)
        self.host.rmem.episodic = self.host.episodic
        keys = [rkey(r) for r in self.battery.get("seed_relations", [])]
        meas = measure_capabilities(self.host, self.battery, keys,
                                    apply_new_and_interventions=False)
        out = {"scores": {k: round(v, 6) for k, v in meas["scores"].items()},
               "provenance": {c: provenance_histogram(meas["rows"][c])
                              for c in meas["rows"]},
               "epoch": "post-intervention (symmetric with the disconnected "
                        "lane's post-restart scoring)"}
        _dump_json(out, os.path.join(self.run_dir, "host_rescored_post.json"))
        self.log(f"host post-epoch rescore -> "
                 f"{os.path.join(self.run_dir, 'host_rescored_post.json')}")
        return out


# --------------------------------------------------------------------------- #
# Orchestrator (programmatic entry + the Phase-2 seam)
# --------------------------------------------------------------------------- #

def run_disconnect_experiment(battery: Dict, run_dir: str, seed: int = 0,
                              n_learn_steps: int = 2,
                              consolidation_interval: int = 8,
                              retention_fraction: float = 0.5,
                              teacher: Optional[object] = None,
                              teacher_name: str = "scripted-teacher",
                              foundation: str = "stub", live: bool = False,
                              prebuilt_iface: Optional[object] = None,
                              foundation_loader: Optional[Callable] = None,
                              vocab_size: int = 128,
                              phases: Tuple[str, ...] = PHASES,
                              verbose: bool = True) -> Dict:
    """Run the owner's 6-step disconnect protocol and return the master report.

    See the module docstring for the Phase-2 integration seam (prebuilt_iface,
    foundation_loader, live gate). Battery gold answers are re-derived from the
    battery's own graph before anything runs (verify_battery) — a bad battery
    refuses to run rather than fabricating numbers."""
    exp = DisconnectExperiment(
        battery=battery, run_dir=run_dir, seed=seed,
        n_learn_steps=n_learn_steps, consolidation_interval=consolidation_interval,
        retention_fraction=retention_fraction, teacher=teacher,
        teacher_name=teacher_name, foundation=foundation, live=live,
        prebuilt_iface=prebuilt_iface, foundation_loader=foundation_loader,
        vocab_size=vocab_size, verbose=verbose)
    return exp.run(phases)


# --------------------------------------------------------------------------- #
# CLI (retention.py's pattern: --phase, run-dir, seed, strict JSON outputs)
# --------------------------------------------------------------------------- #

def load_battery(path: Optional[str]) -> Dict:
    if path is None:
        return copy.deepcopy(DEFAULT_DISCONNECT_BATTERY)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AION-1 disconnect experiment "
                                             "(owner 6-step protocol)")
    ap.add_argument("--phase", default="all",
                    help="all | " + " | ".join(PHASES) + " | step number 0-6")
    ap.add_argument("--battery", default=None,
                    help="battery JSON (defaults to the built-in tiny battery)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learn-steps", type=int, default=2)
    ap.add_argument("--consolidation-interval", type=int, default=8)
    ap.add_argument("--live", action="store_true",
                    help=f"REAL run on the frozen foundation; requires env "
                         f"{LIVE_GATE_ENV}=1 (Phase 2 landed + owner go)")
    ap.add_argument("--foundation", default="stub", choices=("stub", "qwen"))
    ap.add_argument("--rescore-host", action="store_true",
                    help="re-score the WITH-host lane post-intervention from "
                         "its saved state (register symmetry check); needs "
                         "state_host.pt in --run-dir and the foundation "
                         "available when it was a live run")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on non-exact state restore or "
                         "consolidation-invariant violations")
    args = ap.parse_args(argv)

    if args.rescore_host:
        run_dir = args.run_dir
        if not run_dir or not os.path.exists(os.path.join(run_dir, "state_host.pt")):
            print("NOT-RUN: --rescore-host needs --run-dir with state_host.pt")
            return 1
        if args.live or args.foundation == "qwen":
            if os.environ.get(LIVE_GATE_ENV) != "1":
                print(f"NOT-RUN (owner gate): export {LIVE_GATE_ENV}=1 first")
                return 2
        battery = load_battery(args.battery)
        exp = DisconnectExperiment(
            battery=battery, run_dir=run_dir, seed=args.seed,
            foundation=args.foundation,
            live=(args.foundation == "qwen" and args.live))
        exp.verbose = True
        try:
            out = exp.rescore_host_post()
        except LiveGateError as e:
            print(f"NOT-RUN (owner gate): {e}")
            return 2
        print(json.dumps(out, indent=1, default=_jsonable))
        return 0

    phase = PHASE_ALIASES.get(args.phase, args.phase)
    if phase == "all":
        phases = PHASES
    elif phase in PHASES:
        phases = (phase,)
    else:
        print(f"NOT-RUN: unknown phase {args.phase!r} (choices: all, "
              f"{', '.join(PHASES)})")
        return 2

    battery = load_battery(args.battery)
    sound = verify_battery(battery)
    if sound:
        print("NOT-RUN: battery ground truth UNSOUND — no fabricated runs:")
        for s in sound:
            print(f"  - {s}")
        return 1

    run_dir = args.run_dir or os.path.join(
        _REPO_ROOT, "runs", f"disconnect_{'live' if args.live else 'stub'}_seed{args.seed}")
    exp = DisconnectExperiment(
        battery=battery, run_dir=run_dir, seed=args.seed,
        n_learn_steps=args.learn_steps,
        consolidation_interval=args.consolidation_interval,
        foundation=args.foundation, live=args.live)
    exp.verbose = True

    try:
        exp.run(phases)
    except LiveGateError as e:
        print(f"NOT-RUN (owner gate): {e}")
        return 2
    except SequencingError as e:
        print(f"NOT-RUN (sequencing): {e}")
        return 1

    fatal = []
    rep = exp.reports.get("restart")
    if rep is not None and rep.get("state_restored_exact") is False:
        fatal.append("disconnected-lane state did NOT restore exactly from disk")
    for p in ("teach", "observe", "simulate"):
        r = exp.reports.get(p)
        if r is not None:
            inv = (r.get("disconnected", {}) or {}).get("consolidation_invariant_ok", True)
            inv = r.get("consolidation_invariant_ok", inv)
            if inv is False:
                fatal.append(f"consolidation invariant violated during {p}")
    if args.strict and fatal:
        for f_ in fatal:
            print(f"FATAL: {f_}")
        return 1
    print(f"\nreport: {os.path.join(run_dir, 'disconnect_report.json')}")
    if "register" in exp.reports:
        reg = exp.reports["register"]["register"]
        print(f"{'capability':14s} {'with':>7s} {'without':>8s} {'survival':>10s}  verdict")
        for cap, c in reg["capabilities"].items():
            print(f"{cap:14s} {c['with_host']:7.3f} "
                  f"{c['without_host_post_restart']:8.3f} "
                  f"{str(c['survival_ratio']):>10s}  {c['verdict']}")
        if reg["confound_guard"]["caveat"]:
            print(f"CONFOUND: {reg['confound_guard']['caveat']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
