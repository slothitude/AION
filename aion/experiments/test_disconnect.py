"""Unit tests for the Disconnect Experiment harness (CPU stubs only, seeded).

Owner acceptance for build-ahead (2026-11-24/25): construct + unit-test with
stubs/tiny models on CPU; NO real experiments, NO real model/endpoint calls,
everything deterministic with seeds. Covers:
  1. phase sequencing enforced (can't retest before teach+disconnect; CLI
     split-phase reports NOT-RUN exit 1)
  2. STRUCTURAL zero-host proof in the disconnected lanes (measured zero
     monitor counts, closed gate raises, no interface object, poison loader)
  3. survival-ratio math on fixtures + the register reports BOTH runs' numbers
     and is strict JSON (no inf/nan tokens)
  4. restart round-trip: bitwise tensor equality + exact probes on the
     disconnected lane
  5. provenance discipline: RETRIEVED answers NEVER count as owned; owned
     counts come from PLASTIC/CONSOLIDATED/MIXED only
  6. self-correct: wrong prediction -> correction as plastic update -> re-scored
     correct, old facts intact
  7. battery ground-truth gate: a corrupted battery refuses to run (no
     fabricated numbers)
  8. determinism: same seed -> identical register
  9. split-phase CLI: each phase is its own process invocation, resuming from
     disk, ending with the same register as the in-process run
 10. confound guard present on stub runs; live gate refuses without the owner env

Run:
    python -m pytest aion/experiments/test_disconnect.py -q   # pytest
    python aion/experiments/test_disconnect.py                # standalone
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import torch

try:
    import pytest
except ImportError:            # run-venvs may lack pytest; standalone still works
    pytest = None


class _Raises:
    """Minimal pytest.raises stand-in (assert-based)."""

    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), \
            f"expected {self.exc.__name__}, got {et.__name__ if et else None}"
        return True


def raises(exc):
    return pytest.raises(exc) if pytest is not None else _Raises(exc)


def approx(x, rel=1e-6):
    if pytest is not None:
        return pytest.approx(x, rel=rel)
    return _Approx(x, rel)


class _Approx:
    def __init__(self, x, rel):
        self.x, self.rel = x, rel

    def __eq__(self, other):
        return math.isclose(float(other), float(self.x), rel_tol=self.rel)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.experiments.disconnect import (                                   # noqa: E402
    DisconnectExperiment, HostDisconnectedError, HostMonitor, LiveGateError,
    SequencingError, answer_question, learn_relation, main, measure_capabilities,
    rkey, run_disconnect_experiment, survival_ratio, survival_ratios,
    verify_battery, DEFAULT_DISCONNECT_BATTERY, LIVE_GATE_ENV,
    LaneDisconnected, LaneHost, PHASES,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _run_full(run_dir: str, seed: int = 0) -> dict:
    return run_disconnect_experiment(dict(DEFAULT_DISCONNECT_BATTERY),
                                    run_dir=run_dir, seed=seed, verbose=False)


def _phase(run_dir: str, name: str, argv_extra=()) -> int:
    """One CLI invocation of a single phase against run_dir (split-process law)."""
    return main(["--phase", name, "--run-dir", run_dir, "--seed", "0",
                 *argv_extra])


def _capture(capsys) -> str:
    return capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 1. phase sequencing
# --------------------------------------------------------------------------- #

def test_phase_sequencing_enforced(tmp_path):
    rd = str(tmp_path / "seq")
    exp = DisconnectExperiment(dict(DEFAULT_DISCONNECT_BATTERY), run_dir=rd, seed=0,
                               verbose=False)
    with raises(SequencingError):
        exp.phase_retest()                      # no teach yet
    with raises(SequencingError):
        exp.phase_disconnect_host()             # no teach/disconnect_teacher
    with raises(SequencingError):
        exp.phase_restart()                     # nothing ran
    exp.phase_teach()
    with raises(SequencingError):
        exp.phase_retest()                      # teacher+host not disconnected yet
    exp.phase_disconnect_teacher()
    with raises(SequencingError):
        exp.phase_retest()                      # host not disconnected yet
    exp.phase_disconnect_host()
    # NOW the retest may run; observe/simulate/restart stay locked behind it
    exp.phase_retest()
    with raises(SequencingError):
        exp.phase_simulate()
    with raises(SequencingError):
        exp.build_register()                    # restart hasn't run


def test_cli_sequencing_reports_not_run(tmp_path, capsys):
    rd = str(tmp_path / "cli_seq")
    rc = _phase(rd, "retest")
    out = _capture(capsys)
    assert rc == 1
    assert "NOT-RUN" in out


# --------------------------------------------------------------------------- #
# 2. structural zero-host proof
# --------------------------------------------------------------------------- #

def test_structural_zero_host_proof(tmp_path):
    rd = str(tmp_path / "zero_host")
    rep = _run_full(rd)
    # phases 3-6 (retest/observe/simulate/restart) all carry the measured proof
    for p in ("retest", "observe", "simulate", "restart"):
        s = rep["phases"][p]["structural_zero_host_touches"]
        assert s["monitor_closed"] is True
        assert s["iface_absent"] is True
        assert s["poison_loader_intact"] is True
        assert s["host_touches_since_close"] == {"foundation_forward": 0,
                                                 "teacher_call": 0,
                                                 "foundation_load": 0}
        assert s["host_touches_total"] == {"foundation_forward": 0,
                                           "teacher_call": 0,
                                           "foundation_load": 0}
    # the control lane DID use the host by design, and the counts say so
    host_counts = rep["phases"]["teach"]["host"]["monitor"]["counts_total"]
    assert host_counts["foundation_forward"] > 0

    # the gate is a real tripwire: a wrapped call while CLOSED raises
    m = HostMonitor("t")
    m.close("test")
    def boom():
        return "host-touched"
    with raises(HostDisconnectedError):
        m.wrap("foundation_forward", boom)()
    # and the poison loader (the disconnected lane's only loader slot) raises
    from aion.experiments.disconnect import _poison_loader
    with raises(HostDisconnectedError):
        _poison_loader()
    # the persisted lane state carries no interface either
    payload = torch.load(os.path.join(rd, "state_disc.pt"), weights_only=False)
    assert payload["kind"] == "lane-disconnected"
    assert "conn" in payload and "index" in payload


# --------------------------------------------------------------------------- #
# 3. survival-ratio math + register discipline
# --------------------------------------------------------------------------- #

def test_survival_ratio_math():
    assert survival_ratio(0.8, 0.4) == approx(0.5)
    assert survival_ratio(1.0, 1.0) == approx(1.0)
    assert survival_ratio(0.0, 0.0) == 1.0          # nothing to lose -> full
    assert survival_ratio(0.0, 0.5) == math.inf     # impossible gain flag
    r = survival_ratios({"recognise": 0.8, "predict": 0.0},
                        {"recognise": 0.4, "predict": 0.0})
    assert r["recognise"] == approx(0.5)
    assert r["predict"] == 1.0


def test_register_reports_both_runs_and_strict_json(tmp_path):
    rd = str(tmp_path / "reg")
    rep = _run_full(rd)
    reg = rep["register"]
    for cap in ("recognise", "predict", "learn_new", "combine", "self_correct"):
        c = reg["capabilities"][cap]
        # BOTH runs' numbers present — the ratio never hides either
        assert isinstance(c["with_host"], (int, float))
        assert isinstance(c["without_host_post_restart"], (int, float))
        assert "survival_ratio" in c and "verdict" in c
    # strict JSON on disk: no Infinity/NaN tokens anywhere
    with open(os.path.join(rd, "disconnect_report.json"), "r", encoding="utf-8") as f:
        text = f.read()
    assert "Infinity" not in text and "NaN" not in text
    json.loads(text)                                 # valid JSON round-trip
    # provenance discipline fields present per capability
    for cap, c in reg["capabilities"].items():
        owned = sum(v for k, v in c["provenance_histogram"].items()
                    if k in ("PLASTIC", "CONSOLIDATED", "MIXED"))
        assert c["owned_rows"] == owned
        assert "retrieved_rows_NEVER_OWNED" in c


# --------------------------------------------------------------------------- #
# 4. restart round-trip (bitwise)
# --------------------------------------------------------------------------- #

def test_restart_round_trip_bitwise(tmp_path):
    rd = str(tmp_path / "restart")
    rep = _run_full(rd)
    r = rep["phases"]["restart"]
    assert r["state_restored_exact"] is True
    assert r["probes_exact"] is True
    assert r["bitwise_tensors"] == "ALL-BITWISE-EQUAL"
    # the reload happened into a brand-new object with zero host touches after
    assert r["structural_zero_host_touches"]["host_touches_total"][
        "foundation_forward"] == 0


# --------------------------------------------------------------------------- #
# 5. provenance discipline
# --------------------------------------------------------------------------- #

def test_provenance_discipline_retrieved_never_owned(tmp_path):
    lane = LaneDisconnected(seed=0)
    key = rkey({"subject": "wex", "relation": "binds", "object": "qor"})
    # (a) knowledge ONLY in the episodic log (never learned): retrieved works,
    #     but it is NEVER owned.
    lane.episodic.record(kind="observation", text="wex binds qor",
                         channel="direct", teacher="nobody",
                         tags={"answer": "qor"})
    ans = answer_question(lane, "wex binds", "qor")
    assert ans["retrieved_hit"] is True
    assert ans["provenance"] == "RETRIEVED"
    assert ans["owned"] is False
    assert ans["plastic_hit"] is False and ans["consolidated_hit"] is False
    # (b) learned knowledge: owned (PLASTIC reading), independent of the log.
    lane2 = LaneDisconnected(seed=0)
    learn_relation(lane2, {"subject": "wex", "relation": "binds", "object": "qor"})
    ans2 = answer_question(lane2, "wex binds", "qor")
    assert ans2["provenance"] == "PLASTIC"
    assert ans2["owned"] is True
    # (c) nothing anywhere -> NONE
    ans3 = answer_question(lane2, "zub flurbs", "mik")
    assert ans3["provenance"] == "NONE" and ans3["owned"] is False


# --------------------------------------------------------------------------- #
# 6. self-correct when reality disagrees
# --------------------------------------------------------------------------- #

def test_self_correct_when_reality_disagrees(tmp_path):
    rd = str(tmp_path / "selfcorr")
    rep = _run_full(rd)
    rows = {r.get("id", r.get("new_key")): r
            for r in rep["phases"]["simulate"]["rows"]}
    # int_sim: the reroute into a KNOWN downstream chain predicts correctly
    # with no correction at all (pure internal simulation).
    a = rows["int_sim"]
    assert a["pre_correct"] is True and a["correction_applied"] is False
    assert a["post_prediction"] == "frost" and a["self_correct_pass"] is True
    # int_gap: the reroute into an UNKNOWN relay predicts WRONG first; reality
    # disagrees; the correction enters as plastic updates; re-scored CORRECT.
    b = rows["int_gap"]
    assert b["pre_correct"] is False
    assert b["pre_prediction"] == "Glova"            # walk dead-ends at the gap
    assert b["correction_applied"] is True
    assert b["post_prediction"] == "glow"
    assert b["old_facts_intact"] is True
    assert b["self_correct_pass"] is True
    # corrections section: replaced fact learned + old fact retained
    c = rows["Zor|produces|vapor"]
    assert c["new_learned"] is True and c["old_retained"] is True
    assert c["self_correct_pass"] is True


# --------------------------------------------------------------------------- #
# 7. battery ground-truth gate (no fabricated numbers)
# --------------------------------------------------------------------------- #

def test_battery_soundness_gate(tmp_path, capsys):
    bad = json.loads(json.dumps(DEFAULT_DISCONNECT_BATTERY))
    bad["interventions"][0]["gold"] = "definitely-wrong-gold"
    failures = verify_battery(bad)
    assert failures and "intervention" in failures[0]
    with raises(ValueError):
        run_disconnect_experiment(bad, run_dir=str(tmp_path / "bad"), verbose=False)
    # CLI refuses too
    bad_path = tmp_path / "bad_battery.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    rc = main(["--phase", "all", "--battery", str(bad_path),
               "--run-dir", str(tmp_path / "bad2")])
    assert rc == 1 and "NOT-RUN" in _capture(capsys)


# --------------------------------------------------------------------------- #
# 8. determinism
# --------------------------------------------------------------------------- #

def test_determinism_same_seed(tmp_path):
    rep1 = _run_full(str(tmp_path / "d1"), seed=3)
    rep2 = _run_full(str(tmp_path / "d2"), seed=3)
    a, b = rep1["register"], rep2["register"]
    assert a["capabilities"] == b["capabilities"]
    for p in ("teach", "retest", "observe", "simulate"):
        s1 = rep1["phases"][p].get("scores", rep1["phases"][p].get(
            "host", {}).get("scores"))
        s2 = rep2["phases"][p].get("scores", rep2["phases"][p].get(
            "host", {}).get("scores"))
        assert s1 == s2, p


# --------------------------------------------------------------------------- #
# 9. split-phase CLI: separate invocations resume from disk
# --------------------------------------------------------------------------- #

def test_split_phase_cli_resume(tmp_path, capsys):
    rd = str(tmp_path / "split")
    for name in PHASES:
        rc = _phase(rd, name)
        assert rc == 0, f"phase {name} exit={rc}: {_capture(capsys)}"
    rep = json.load(open(os.path.join(rd, "disconnect_report.json"),
                         encoding="utf-8"))
    assert rep["completed_phases"] == list(PHASES)
    assert rep["phases"]["restart"]["state_restored_exact"] is True
    # the register produced by split invocations matches the in-process one
    whole = _run_full(str(tmp_path / "whole"))
    assert rep["register"]["capabilities"] == \
        whole["register"]["capabilities"]


# --------------------------------------------------------------------------- #
# 10. confound guard + live gate
# --------------------------------------------------------------------------- #

def test_confound_guard_on_stub_run(tmp_path):
    rep = _run_full(str(tmp_path / "confound"))
    cg = rep["register"]["confound_guard"]
    assert cg["substrate_with_host_run"] == "stub-tiny-llama"
    assert cg["substrate_degraded"] is True
    assert "CAVEAT" in (cg["caveat"] or "")
    # substrate capability reported alongside (reviewer requirement)
    assert set(cg["substrate_capability_with_host"].keys()) == \
        {"recognise", "predict", "learn_new", "combine", "self_correct"}
    assert rep["execution_mode"].startswith("stub-build")
    assert "NOT-RUN" in rep["execution_mode"]


def test_live_gate_refuses_without_owner_env(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(LIVE_GATE_ENV, raising=False)
    rd = str(tmp_path / "live")
    with raises(LiveGateError):
        run_disconnect_experiment(dict(DEFAULT_DISCONNECT_BATTERY), run_dir=rd,
                                  live=True, foundation="qwen", verbose=False)
    rc = main(["--phase", "teach", "--live", "--foundation", "qwen",
               "--run-dir", rd + "2"])
    assert rc == 2 and "NOT-RUN" in _capture(capsys)
    # env set but wrong substrate still refuses
    monkeypatch.setenv(LIVE_GATE_ENV, "1")
    with raises(LiveGateError):
        run_disconnect_experiment(dict(DEFAULT_DISCONNECT_BATTERY),
                                  run_dir=rd + "3", live=True,
                                  foundation="stub", verbose=False)


# --------------------------------------------------------------------------- #
# extra: host monitor counts real host passes in the control lane
# --------------------------------------------------------------------------- #

def test_host_monitor_counts_foundation_passes(tmp_path):
    rd = str(tmp_path / "mon")
    rep = _run_full(rd)
    m = rep["phases"]["teach"]["host"]["monitor"]
    assert m["closed"] is False                       # control lane stays open
    assert m["counts_total"]["foundation_forward"] > 0
    # protocol events carry the structural disconnection evidence
    proto = json.load(open(os.path.join(rd, "protocol.json"), encoding="utf-8"))
    ev = proto["events"]["structural_disconnection"]
    assert ev["monitor_closed"] is True
    assert ev["host_touches_since_close"]["foundation_forward"] == 0
    assert proto["completed_phases"] == list(PHASES)


# --------------------------------------------------------------------------- #
# standalone runner
# --------------------------------------------------------------------------- #

def _main() -> int:
    tests = [
        test_phase_sequencing_enforced,
        test_cli_sequencing_reports_not_run,
        test_structural_zero_host_proof,
        test_survival_ratio_math,
        test_register_reports_both_runs_and_strict_json,
        test_restart_round_trip_bitwise,
        test_provenance_discipline_retrieved_never_owned,
        test_self_correct_when_reality_disagrees,
        test_battery_soundness_gate,
        test_determinism_same_seed,
        test_split_phase_cli_resume,
        test_confound_guard_on_stub_run,
        test_live_gate_refuses_without_owner_env,
        test_host_monitor_counts_foundation_passes,
    ]
    failures = 0
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        for t in tests:
            name = t.__name__
            try:
                varnames = t.__code__.co_varnames
                kwargs = {}
                if "tmp_path" in varnames:
                    d = Path(tmp) / name
                    d.mkdir(parents=True, exist_ok=True)
                    kwargs["tmp_path"] = d
                monkey = None
                if "monkeypatch" in varnames:
                    monkey = _SimpleMonkey(os.environ)
                    kwargs["monkeypatch"] = monkey
                buf = io.StringIO()
                if "capsys" in varnames:
                    kwargs["capsys"] = _NullCapsys(buf)
                with contextlib.redirect_stdout(buf):
                    t(**kwargs)
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001 — runner reports, not crashes
                failures += 1
                import traceback
                print(f"FAIL  {name}: {e}")
                traceback.print_exc()
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


class _NullCapsys:
    """Standalone-mode stand-in for pytest's capsys (the runner redirects
    stdout into the shared buffer)."""

    def __init__(self, buf):
        self._buf = buf

    def readouterr(self):
        class R:
            pass
        r = R()
        r.out = self._buf.getvalue()
        return r


class _SimpleMonkey:
    """Minimal monkeypatch stand-in (setenv/delenv only)."""

    def __init__(self, environ):
        self._environ = environ

    def setenv(self, k, v):
        self._environ[k] = v

    def delenv(self, k, raising=True):
        self._environ.pop(k, None)


if __name__ == "__main__":
    sys.exit(_main())
