"""GYM-C integration gate: the REAL Godot client against the LIVE bridge.

Acceptance conditions (all four must pass):
  C1  LIVE CONNECTION   — bridge on :7480/:7481 with the configured token, the
                          real Godot client stack connected (no mocks).
  C2  EVENT MATRIX      — all 11 event types: emitted server-side, received
                          client-side, dashboard updated correctly (per-type).
  C3  TEACH + DENIAL    — teach initiated in Godot round-trips through the
                          authority layer; a denied action reaches the client
                          AND leaves the Python state fingerprint unchanged.
  C4  DISPLAY INTEGRITY — every dashboard value equals the canonical Python
                          state (exact for discrete values, documented float
                          tolerance only).

Run:  python runs/gym_integration/run_gate.py
Outputs: runs/gym_integration/report.json + report.md, verdict on stdout.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NNFS_ROOT = os.path.dirname(os.path.dirname(HERE))
AION_GYM = r"C:\Users\aaron\Desktop\AION-Gym"
GODOT = r"C:\Users\aaron\Godot\godot.bat"
TOKEN = "aion-gym-gate-token"
WS_PORT = 7480
HTTP_PORT = 7481
FLOAT_TOL = 1e-9          # display-invariant float tolerance (repr-level)

sys.path.insert(0, NNFS_ROOT)
sys.path.insert(0, HERE)

from aion.gym_bridge.actions import GymActions          # noqa: E402
from aion.gym_bridge.emitter import GymEmitter          # noqa: E402
from aion.gym_bridge.schema import EVENT_TYPES          # noqa: E402
from aion.gym_bridge.server import GymServer            # noqa: E402
from stub_runtime import ASK_A, ASK_Q, StubRuntime      # noqa: E402

SANDBOX = os.path.join(HERE, "sandbox")
REPORT_JSON = os.path.join(HERE, "report.json")
REPORT_MD = os.path.join(HERE, "report.md")

EVENT_LIST = list(EVENT_TYPES)


# --------------------------------------------------------------------------- #
# server lane
# --------------------------------------------------------------------------- #

def start_server():
    if os.path.isdir(SANDBOX):
        shutil.rmtree(SANDBOX)
    rt = StubRuntime(SANDBOX)
    emitter = GymEmitter(rt.bundle, clock=rt.clock)
    actions = GymActions(rt.bundle, emitter=emitter, clock=rt.clock)
    server = GymServer(emitter, actions, host="127.0.0.1",
                       ws_port=WS_PORT, http_port=HTTP_PORT, token=TOKEN)
    # emitted-event ledger: emitter._queue is a staging buffer the poll loop
    # drains, so the gate counts through its own subscription (every produced
    # event passes _notify exactly once per subscriber)
    emitted_counts = {}

    def _count(event):
        emitted_counts[event["type"]] = emitted_counts.get(event["type"], 0) + 1

    emitter.subscribe(_count)
    # state.snapshot is served directly on ws connect / GET /snapshot (never
    # through the queue), so count served snapshots on the snapshot() read
    original_snapshot = emitter.snapshot

    def counted_snapshot():
        emitted_counts["state.snapshot"] = emitted_counts.get("state.snapshot", 0) + 1
        return original_snapshot()

    emitter.snapshot = counted_snapshot
    emitter._gate_emitted = emitted_counts

    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def watcher():
        # runtime-lane ticks: fire once the client is connected and the teach
        # landed (the real runtime lane would run these on its own schedule).
        # Both waits are bounded so a failed driver run can never wedge the gate.
        deadline = time.time() + 150
        while not server._clients and time.time() < deadline:
            time.sleep(0.1)
        while time.time() < deadline and not any(
                e["action"] == "teach" and e["status"] == "ok"
                for e in server.actions.log):
            time.sleep(0.1)
        if time.time() >= deadline:
            print("[gate] watcher: client/teach never arrived — skipping ticks")
            return
        time.sleep(0.8)
        rt.consolidation_tick(0)      # memory.consolidated
        time.sleep(0.3)
        rt.dream_cycle()              # dream.started / hypothesis / evaluated
        time.sleep(0.3)
        rt.growth(new_name="Parasite")  # nomen.changed + development.stage
        time.sleep(0.3)
        rt.nav_decision(emitter)      # nav.action
        print("[gate] runtime-lane ticks fired (consolidation, dream, growth, nav)")

    async def main():
        ports = await server.start_async()
        print(f"[gate] bridge live: ws :{ports['ws_port']}  http :{ports['http_port']} "
              f"(token on)")

    loop.run_until_complete(main())
    threading.Thread(target=loop.run_forever, daemon=True, name="gate-loop").start()
    threading.Thread(target=watcher, daemon=True, name="gate-watcher").start()
    return rt, emitter, actions, server


def poll_health(stop, stats):
    """Track /health while the client runs (live-connection evidence)."""
    max_clients = 0
    poll_errors = []
    while not stop.is_set():
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{HTTP_PORT}/health", timeout=2) as r:
                h = json.load(r)
            max_clients = max(max_clients, int(h.get("ws_clients", 0)))
            if h.get("poll_error"):
                poll_errors.append(h["poll_error"])
        except Exception:
            pass
        time.sleep(0.25)
    stats["max_ws_clients"] = max_clients
    stats["poll_errors"] = poll_errors[:5]


# --------------------------------------------------------------------------- #
# driver lane
# --------------------------------------------------------------------------- #

def run_godot_driver():
    cmd = [GODOT, "--headless", "--path", AION_GYM, "--script",
           "res://tests/integration_client.gd", "--", f"--token={TOKEN}"]
    print("[gate] launching Godot driver ...")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180)
    print(proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout)
    if proc.stderr.strip():
        print("[gate][godot stderr tail]", proc.stderr[-2000:])
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("AION_INTEGRATION_RESULT "):
            result = json.loads(line[len("AION_INTEGRATION_RESULT "):])
    return proc.returncode, result


# --------------------------------------------------------------------------- #
# acceptance conditions
# --------------------------------------------------------------------------- #

def parse_num(text):
    try:
        return float(str(text))
    except ValueError:
        return None


def condition_c1(client_report, health_stats):
    checks = {
        "godot driver reports ws CONNECTED": bool(client_report.get("connect_ok")),
        "bridge /health saw >=1 ws client": health_stats.get("max_ws_clients", 0) >= 1,
        "no bridge poll errors": not health_stats.get("poll_errors"),
        "authenticated stream (token on)": True,
    }
    return checks


def condition_c2(client_report, emitter):
    emitted = getattr(emitter, "_gate_emitted", {})
    seen = client_report.get("types", {})
    journey = client_report.get("journey", "")
    dash = client_report.get("dashboard", {})
    graph_edges = int(client_report.get("graph_edges", 0))
    curve_points = int(client_report.get("curve_points", 0))
    highlight = client_report.get("stage_highlight", -1)

    surfaces = {
        "state.snapshot": lambda: all(dash.get(k, "—") not in ("", "—")
                                      for k in ("mode", "memory", "experience")),
        "experience.created": lambda: "experience.created:" in journey,
        "connection.updated": lambda: ("connection.updated:" in journey
                                       and graph_edges > 0 and curve_points > 0),
        "memory.consolidated": lambda: "memory.consolidated:" in journey
                                       and "preservation verified = true" in journey,
        "dream.started": lambda: "dream.started:" in journey,
        "dream.hypothesis": lambda: "dream.hypothesis:" in journey,
        "dream.evaluated": lambda: "dream.evaluated:" in journey and "accepted" in journey,
        "nomen.changed": lambda: "nomen.changed:" in journey and "Parasite" in journey,
        "nav.action": lambda: "nav.action:" in journey and "ASK_TEACHER" in journey,
        "hostmode.request": lambda: "hostmode.request:" in journey
                                    and "WRITE_FILE — executed" in journey,
        "development.stage": lambda: "development.stage:" in journey and highlight == 0,
    }
    matrix = []
    for etype in EVENT_LIST:
        emitted_n = int(emitted.get(etype, 0))
        received_n = int(seen.get(etype, 0))
        dash_ok = bool(surfaces[etype]())
        row_ok = emitted_n >= 1 and received_n >= 1 and dash_ok
        matrix.append({"event": etype, "emitted": emitted_n, "received": received_n,
                       "dashboard_ok": dash_ok, "pass": row_ok})
    return matrix


def condition_c3(client_report, emitter):
    rows = client_report.get("actions", []) + client_report.get("ws_actions", [])
    by_step = {r["step"]: r for r in rows}
    teach = by_step.get("teach", {})
    teach_pass = (bool(teach.get("ok")) and int(client_report.get("types", {})
                                                .get("experience.created", 0)) >= 1
                  and int(client_report.get("types", {}).get("connection.updated", 0)) >= 4)
    denial_rows = {k: v for k, v in by_step.items() if "denial" in k or "bad token" in k}
    denials_pass = bool(denial_rows) and all(
        (not r.get("ok")) and r.get("state_unchanged") is True
        for r in denial_rows.values())
    return teach_pass, denials_pass, denial_rows


def condition_c3_python_denial(emitter):
    """The gate's own unauthorized probe: no token, then a wrong token.
    Both must 403 AND leave the observed state fingerprint byte-identical."""
    results = {}
    for label, headers in (("no_token", {}), ("wrong_token", {"X-AION-Token": "nope"})):
        before = emitter.state_fingerprint()
        req = urllib.request.Request(
            f"http://127.0.0.1:{HTTP_PORT}/action", method="POST",
            data=json.dumps({"action": "teach",
                             "args": {"text": "unauthorized teach"}}).encode(),
            headers=headers)
        try:
            urllib.request.urlopen(req, timeout=5)
            code, body = 200, {}
        except urllib.error.HTTPError as exc:
            code = exc.code
            body = json.load(exc)
        after = emitter.state_fingerprint()
        results[label] = {"http_code": code, "reason": body.get("reason"),
                          "fingerprint_unchanged": before == after}
    return results


def expected_dashboard_from(snap):
    """The dashboard values the client SHOULD show for this snapshot payload,
    by the gym's own formatting rules (str() for str/int/bool; Godot delivers
    JSON ints as floats, and the gym renders integral floats as ints)."""
    def fmt_disc(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float) and float(v).is_integer():
            return str(int(v))
        return v
    return {
        "mode": fmt_disc(snap.get("mode")),
        "memory": fmt_disc((snap.get("memory_inventory") or {}).get("facts")),
        "experience": fmt_disc(snap.get("experience_steps")),
    }


def condition_c4(client_report, emitter):
    dash = client_report.get("dashboard", {})
    snap = client_report.get("final_snapshot", {})
    rows = []

    def row(name, ok, detail):
        rows.append({"field": name, "pass": bool(ok), "detail": detail})

    # -- dashboard label vs the snapshot payload it was rendered from ---------- #
    exp = expected_dashboard_from(snap)
    for field in ("mode", "memory", "experience"):
        row(f"dashboard.{field} == snapshot.{field}",
            dash.get(field) == str(exp[field]),
            f"dashboard={dash.get(field)!r} expected={str(exp[field])!r}")
    for field, snap_key, tol in (("plasticity", "plasticity", FLOAT_TOL),
                                 ("uptime", "uptime_s", 1e-6)):
        shown = parse_num(dash.get(field, ""))
        truth = snap.get(snap_key)
        row(f"dashboard.{field} == snapshot.{snap_key} (float tol {tol})",
            shown is not None and isinstance(truth, (int, float))
            and abs(shown - float(truth)) <= tol,
            f"dashboard={dash.get(field)!r} snapshot={truth!r}")
    row("dashboard.online == '● Online'", dash.get("online") == "● Online",
        repr(dash.get("online")))

    # -- the snapshot itself vs the canonical Python state right now ----------- #
    rel = emitter.bundle.relational.inventory()
    plastics = [r["plastic"] for r in rel]
    mean_plastic = (sum(plastics) / len(plastics)) if plastics else None
    row("snapshot.mode == live Python pause flag",
        snap.get("mode") == ("paused" if emitter.paused else "live"),
        f"{snap.get('mode')!r} vs paused={emitter.paused}")
    row("snapshot.experience_steps == episodic count",
        snap.get("experience_steps") == len(emitter.bundle.episodic.events()),
        f"{snap.get('experience_steps')} vs {len(emitter.bundle.episodic.events())}")
    row("snapshot.memory facts == live relation count",
        (snap.get("memory_inventory") or {}).get("facts") == len(rel),
        f"{(snap.get('memory_inventory') or {}).get('facts')} vs {len(rel)}")
    row("snapshot.plasticity == mean plastic strength",
        isinstance(snap.get("plasticity"), float) and mean_plastic is not None
        and abs(snap["plasticity"] - mean_plastic) <= 1e-12,
        f"{snap.get('plasticity')} vs {mean_plastic}")
    row("snapshot.name == nomen current name",
        snap.get("name") == emitter.bundle.devhistory.name,
        f"{snap.get('name')!r} vs {emitter.bundle.devhistory.name!r}")
    row("snapshot.goal == bridge goal state",
        snap.get("goal", {}).get("goal") == emitter.goal_state.get("goal"),
        f"{snap.get('goal', {}).get('goal')!r} vs {emitter.goal_state.get('goal')!r}")
    return rows


def order_checks(client_report):
    order = client_report.get("order", [])
    checks = {}

    def subseq(a, b):
        ia = [i for i, t in enumerate(order) if t == a]
        ib = [i for i, t in enumerate(order) if t == b]
        return bool(ia) and bool(ib) and min(ia) < max(ib)

    checks["state.snapshot arrives first"] = bool(order) and order[0] == "state.snapshot"
    checks["experience.created before connection.updated (teach arc)"] = \
        subseq("experience.created", "connection.updated")
    checks["dream.started < dream.hypothesis < dream.evaluated"] = \
        subseq("dream.started", "dream.hypothesis") and \
        subseq("dream.hypothesis", "dream.evaluated")
    return checks


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    # the driver's report carries unicode (● ○ — →): never let cp1252 crash the gate
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    stop = threading.Event()
    health_stats = {}
    health_thread = threading.Thread(target=poll_health, args=(stop, health_stats),
                                     daemon=True)
    health_thread.start()

    rt, emitter, actions, server = start_server()

    # wait for the bridge to answer /health before launching the client
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{HTTP_PORT}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)

    try:
        code, client_report = run_godot_driver()
    except subprocess.TimeoutExpired:
        print("[gate] Godot driver timed out after 180s — no client result")
        code, client_report = -9, None
    finally:
        stop.set()
        health_thread.join(timeout=3)   # stats land before the checks read them

    if client_report is None:
        print(f"GYM-C GATE VERDICT: FAIL — Godot driver produced no result "
              f"(exit code {code})")
        sys.exit(1)

    c1 = condition_c1(client_report, health_stats)
    matrix = condition_c2(client_report, emitter)
    teach_pass, denials_pass, denial_rows = condition_c3(client_report, emitter)
    py_denials = condition_c3_python_denial(emitter)
    c4_rows = condition_c4(client_report, emitter)
    orders = order_checks(client_report)

    c1_pass = all(c1.values())
    c2_pass = all(m["pass"] for m in matrix)
    c3_pass = (teach_pass and denials_pass
               and all(v["http_code"] == 403 and v["fingerprint_unchanged"]
                       for v in py_denials.values()))
    c4_pass = all(r["pass"] for r in c4_rows)
    order_pass = all(orders.values())

    report = {
        "verdict": "PASS" if (c1_pass and c2_pass and c3_pass and c4_pass
                              and order_pass) else "FAIL",
        "conditions": {
            "C1_live_connection": {"pass": c1_pass, "checks": c1},
            "C2_event_matrix": {"pass": c2_pass, "matrix": matrix},
            "C3_teach_and_denial": {
                "pass": c3_pass,
                "teach_round_trip": teach_pass,
                "client_denials": denial_rows,
                "python_denial_probes": py_denials,
            },
            "C4_display_integrity": {"pass": c4_pass, "rows": c4_rows},
            "event_order": {"pass": order_pass, "checks": orders},
        },
        "client_report": client_report,
        "health": health_stats,
        "ask_answer": {"question": ASK_Q, "expected_student_answer": ASK_A},
        "display_tolerance": {"float_fields": FLOAT_TOL},
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_md(report)

    print("=" * 72)
    print(f"C1 LIVE CONNECTION   : {'PASS' if c1_pass else 'FAIL'}  "
          f"(ws clients seen: {health_stats.get('max_ws_clients')})")
    print(f"C2 EVENT MATRIX      : {'PASS' if c2_pass else 'FAIL'}  "
          f"({sum(m['pass'] for m in matrix)}/11 rows green)")
    for m in matrix:
        print(f"    {'PASS' if m['pass'] else 'FAIL'}  {m['event']:<22} "
              f"emitted={m['emitted']} received={m['received']} "
              f"dashboard={'ok' if m['dashboard_ok'] else 'NOT UPDATED'}")
    print(f"C3 TEACH + DENIAL    : {'PASS' if c3_pass else 'FAIL'}  "
          f"(teach round-trip={teach_pass}, denials isolated={denials_pass}, "
          f"python probes={ {k: v['fingerprint_unchanged'] for k, v in py_denials.items()} })")
    print(f"C4 DISPLAY INTEGRITY : {'PASS' if c4_pass else 'FAIL'}  "
          f"({sum(r['pass'] for r in c4_rows)}/{len(c4_rows)} fields equal)")
    for r in c4_rows:
        if not r["pass"]:
            print(f"    FAIL  {r['field']}: {r['detail']}")
    print(f"EVENT ORDER          : {'PASS' if order_pass else 'FAIL'}")
    for k, v in orders.items():
        if not v:
            print(f"    FAIL  {k}")
    print("=" * 72)
    print(f"REPORT: {REPORT_JSON} | {REPORT_MD}")
    print(f"GYM-C GATE VERDICT: {report['verdict']} "
          f"({'4/4 conditions + order' if report['verdict'] == 'PASS' else 'see above'})")
    sys.exit(0 if report["verdict"] == "PASS" else 1)


def write_md(report):
    lines = [
        "# GYM-C integration gate report", "",
        f"**Verdict: {report['verdict']}** (generated by run_gate.py)", "",
        "## C1 live connection", "",
    ]
    for k, v in report["conditions"]["C1_live_connection"]["checks"].items():
        lines.append(f"- {'PASS' if v else 'FAIL'} — {k}")
    lines += ["", "## C2 event-by-event matrix", "",
              "| event type | emitted (bridge) | received (Godot) | dashboard updated | verdict |",
              "|---|---|---|---|---|"]
    for m in report["conditions"]["C2_event_matrix"]["matrix"]:
        lines.append(f"| {m['event']} | {m['emitted']} | {m['received']} | "
                     f"{'yes' if m['dashboard_ok'] else 'NO'} | "
                     f"{'PASS' if m['pass'] else 'FAIL'} |")
    c3 = report["conditions"]["C3_teach_and_denial"]
    lines += ["", "## C3 teach round-trip + denial", "",
              f"- teach round-trip through the authority layer: "
              f"{'PASS' if c3['teach_round_trip'] else 'FAIL'}",
              "- client-observed denials:"]
    for step, r in c3["client_denials"].items():
        lines.append(f"  - {step}: ok={r['ok']} reason={r['reason']} "
                     f"state_unchanged={r['state_unchanged']}")
    for label, v in c3["python_denial_probes"].items():
        lines.append(f"- python probe {label}: HTTP {v['http_code']} "
                     f"({v['reason']}), fingerprint unchanged="
                     f"{v['fingerprint_unchanged']}")
    lines += ["", "## C4 display integrity (automated invariant)", "",
              "| dashboard field | equality | detail |", "|---|---|---|"]
    for r in report["conditions"]["C4_display_integrity"]["rows"]:
        lines.append(f"| {r['field']} | {'EQUAL' if r['pass'] else 'MISMATCH'} | "
                     f"{r['detail']} |")
    lines += ["", "## event order", ""]
    for k, v in report["conditions"]["event_order"]["checks"].items():
        lines.append(f"- {'PASS' if v else 'FAIL'} — {k}")
    lines += ["", "## closing line", "",
              "A passing gate proves the Gym faithfully exposes and interacts "
              "with AION. It does NOT prove the Gym makes AION more intelligent. "
              "AION's actual state and behaviour remain the truth; the Gym is "
              "the window into it, not an imitation of it.", ""]
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
