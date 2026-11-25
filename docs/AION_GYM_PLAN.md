# AION Gym — Plan of Record

**The design principle:** The Gym is the world. AION is the learner. The owner is the observer, teacher and environment designer.
**Display-integrity law:** AION must not fake its learning for the display. Every visible change corresponds to real state, real events, or an explicitly labelled visualization. The Gym never reimplements AION's intelligence — it observes and requests.

---

## 1. Locked choices — **AMENDED 2026-11-25: Ant's is the Gym foundation** (supersedes the separate-repo decision)

| Choice | Decision |
|---|---|
| Repository | **SUPERSEDED — Ant's hosts the Gym.** `slothitude/AION-Gym` (standalone) is retired as the shell and preserved as the validated design reference (145/145 client tests; its schema.gd/event-bus/state-cache/client core transplants into Ant's). One Godot application, one mobile experience, shared infrastructure. `ants/` is on the project's editable list, so the transplant is legitimate. |
| Visual direction | **Hybrid** — immersive living habitat at the centre, full scientific instrumentation around it |
| Engine | **Godot 4.7.1** (Ant's house version) |
| Authority | Python (the AION runtime) remains authoritative. The Gym — inside Ant's — subscribes to events and requests permitted actions **through AION's authenticated bridge** (which retains independent authority checks). It never writes directly into plastic state and never bypasses the runtime's authority controls. Ant's HTTP server stays 127.0.0.1/Tailscale-bound; it is not exposed directly to the network. |

**The reuse map (owner-identified):** Ant's portrait interface → mobile-first Gym layout · lobby/chat pages → AION selection + conversation · agent tiles/nest canvas → AION activity + relationship visualisation · HTTP API → Gym-to-runtime communication · sequenced event ring → live learning event stream · Android export → native APK · PyBridge → optional on-device Python · Tailscale → remote monitoring. **What must be richer in the Gym:** AION's model is not process+transcript+IDLE/BUSY — it is evolution stages, plasticity+memory, living connection graph, chat/teach/correct, dreams, and experiments.

---

## 2. Architecture — the brain stays separate

```
AION/                          # existing repo (Python organism, authoritative)
├── aion/
│   ├── ...                   # everything built to date (foundation, relational,
│   │                         #   dialogue, worlds, memory, nomen, nav, hostmode)
│   └── gym_bridge/           # NEW: the event interface (the only new aion code)
│       ├── emitter.py        #   subscribes to runtime state; emits the event stream
│       ├── actions.py        #   permitted-action request API (routes through
│       │                     #   the runtime's authority gates — hostmode law)
│       └── schema.py         #   event payload schemas (versioned)
├── docs/AION_GYM_PLAN.md     # this document
AION-Gym/                      # NEW separate repo (the Godot client)
├── project.godot
├── scenes/
│   ├── gym.tscn              # root: layout, lens switching
│   ├── habitat.tscn          # 3D living environment
│   ├── mind_viewer.tscn      # relationships, plastic state, fast/slow
│   ├── training_world.tscn   # tasks, objects, questions, attempts
│   ├── memory_archive.tscn   # experiences, corrections, provenance, dreams
│   └── control_room.tscn     # chat, teach, pause/resume, goals, approvals
├── scripts/
│   ├── core/                 # event bus, state cache, connection manager
│   ├── aion_client/          # WebSocket + HTTP client for gym_bridge
│   ├── visualization/        # plastic-connection graph, mind renderer
│   ├── interaction/          # teaching, questioning, correction flows
│   └── worlds/               # training-world scene logic
├── assets/
├── themes/
└── tests/
```

**The only new Python code is `aion/gym_bridge/`** — an observation-and-request layer. It contains no learning logic. It reads state (already public to the project), emits events, and forwards action requests into the runtime's existing authority gates (the hostmode law: AION proposes, the runtime disposes; the Gym is just another proposer, with the same zero-direct-write discipline).

---

## 3. The event interface

Transport: **WebSocket** (push stream, Godot subscribes) + **HTTP** (request/response for snapshots and permitted actions). All payloads versioned (`schema_version` field); additive-only evolution.

| Event | Emitted when | Payload core |
|---|---|---|
| `state.snapshot` | on connect + on request | full inspectable state: relationships, u_fast/θ strengths, gate states, memory inventory, devhistory stage |
| `experience.created` | a lesson/observation enters memory | episode id, provenance, fact extracted |
| `connection.updated` | plastic strengths change (throttled) | connection id, Δu_fast, Δθ, strengths before/after |
| `memory.consolidated` | a consolidation event fires | what moved u→θ, preservation-verified flag |
| `dream.started` | a dream cycle begins | cycle id, seeds/retrieved experiences |
| `dream.hypothesis` | a hypothesis is proposed | the composed relationship, confidence |
| `dream.evaluated` | reality tests a hypothesis | measured improvement, accepted/rejected + reason |
| `nomen.changed` | NOMEN names or renames | entity id, old name, new name, lineage |
| `nav.action` | AION-NAV chooses an action | decision record (goal, options, choice, outcome) |
| `hostmode.request` | a host-mode proposal is made | proposal, authority verdict (granted/denied + reason) |
| `development.stage` | a gestation transition is recorded | stage name (NOMEN-generated), evidence |

**Request API (permitted actions only):** `teach(experience)`, `ask(question)`, `correct(fact)`, `pause()/resume()`, `set_goal(goal)`, `approve_request(id)/deny_request(id)`, `snapshot()`. Every request is validated by the runtime's authority layer; denials return reasons and are logged.

---

## 4. The six surfaces

### 4.1 Habitat (immersive)
A living 3D environment with AION's visible presence. **Cosmetic presence, honest representation**: its appearance may develop with real developmental events (stage changes alter the habitat), but visuals never *invent* capability. Activity shown = activity that happened.

### 4.2 Mind viewer (scientific)
The live plastic graph: connections as edges, thickness = strength, fast/slow as separate layers, γ-gates visible. Inspect an individual connection: its history, what changed, when, and which experience caused it. Drill into the decision records behind every event.

### 4.3 Training worlds
Create tasks and challenges; introduce objects and concepts; ask questions. **Watch the attempts, not just the answers** — each attempt is an event with provenance. Batteries from `aion/worlds/` plug in directly.

### 4.4 Memory archive & dream laboratory
Browse experiences, corrections, provenance (PLASTIC/CONSOLIDATED/RETRIEVED never conflated), patterns. Dream cycles: what was imagined, what reality tested, what was retained — the full acceptance trail.

### 4.5 Control room
Chat, teach, correct, pause/resume, set goals, provide experiences, configure environments, inspect decision records, approve/deny host-mode resource requests. **Every owner action here routes through the runtime's authority layer** — the owner has the largest permitted menu, still not a bypass.

### 4.6 Event history & replay (part of slice 1)
Every event is journaled. The owner can replay any past session — the same events, the same order, against the state snapshot of that moment. Replayable history is the honest answer to "watch real learning happen".

---

## 5. First build milestone — the vertical slice

1. Godot 4.7.1 project skeleton, polished main interface, lens switching
2. `aion/gym_bridge/` emitter + action API, wired to the live AION runtime
3. Godot WebSocket client + state cache + event journal
4. 3D habitat with AION's visible presence (cosmetic, honest)
5. **Interactive teaching and questioning** — the defining loop: teach *Kav* → see the experience enter memory → see connections change → test with a different red object → hear the answer → correct → watch the correction land
6. **Live plastic-connection visualization** — real strengths, real deltas
7. Pause/resume + safe action controls (runtime-validated)
8. Replayable event history

**Slice acceptance (preregistered):** teach 5 new facts through the Gym; the Mind viewer shows the same strengths the Python state reports (numbers match); restart AION mid-session; state restores bitwise; the owner's questions after restart are answered from plastic provenance. If the display and the state ever disagree, that is a shipped bug, not a visualization choice.

---

## 6. Later milestones

| # | Milestone | Surfaces |
|---|---|---|
| M2 | Training worlds (task builder, attempt viewer) | training_world.tscn |
| M3 | Dreaming laboratory (cycles, hypotheses, acceptance trails) | memory_archive + dream events |
| M4 | Memory archive deep views (provenance buckets, consolidation trails) | memory_archive.tscn |
| M5 | NAV interface (decision records, action outcomes) | control_room + nav.action |
| M6 | Host Mode interface (requests, grants, denials — the authority gate made visible) | control_room + hostmode.request |
| M7 | Multi-substrate view (Qwen vs Pith substrates as switchable habitats) | the two-host future |

---

## 7. Sequencing & budget

- The Gym build is **docs + client + bridge** work — it does not consume the experiment pipeline and does not touch the frozen plasticity baseline.
- M1 (vertical slice) is compatible with Phase-3 execution running in parallel: disjoint files, disjoint compute (Godot editor on Rog; Phase-3 CPU on Lappy).
- Effort estimate M1: bridge (small-medium) + Godot project (medium-large — the habitat polish is the long pole). The polish tier can lag the functional tier: functional slice first, habitat beauty second.

## 8. Laws this plan operates under

1. Display integrity (no faked learning; explicit labelling of any visualization)
2. Authority (Godot requests; the runtime disposes; zero direct plastic writes)
3. Register (the Gym reports results in the corrected register — it inherits the honesty laws, being the public face)
4. Single-writer, stay-in-lane (bridge code lives in the AION repo; the Gym repo is separate)
5. No automatic progression between milestones without reviewing the previous milestone's report
