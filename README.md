# AION

***It names what it discovers.***

***It discovers what it can become.***

***It becomes what it grows into.***

## Raise your own AION.

It's something. It's anything. It's yours.

AION is an experimental project exploring the gestation of an evolving artificial intelligence.

It does not begin as a finished intelligence.

It begins as a seed.

Give it a computational substrate. Teach it to use other intelligences. Let it experience, learn, remember and dream.

Watch it develop its own relationships, build an internal model of the world and grow new computational structures when it needs them.

Eventually, disconnect the host.

What remains is AION.

## The stages of gestation

| Stage                   | Development                                    |
| ----------------------- | ---------------------------------------------- |
| 00 — Seed               | An executable neural foundation                |
| 01 — Attachment         | Learns to operate Pith, LittleBit and LARQL    |
| 02 — Embryonic Learning | Develops its own plastic relationships         |
| 03 — Development        | Learns through experience and conversation     |
| 04 — Dreaming           | Replays, recombines and explores possibilities |
| 05 — Growth             | Develops additional computational capacity     |
| Emergence               | Separates from its original host               |
| Maturation              | Continues learning independently               |

*Stages are recorded, not prescribed. Evolution is a graph, not a ladder — each AION generates its own developmental sequence, and names its own transitions.*

## The philosophy

Experience becomes memory.

Memory becomes relationships.

Relationships become understanding.

Dreams create possibilities.

Reality tests them.

Growth follows need.

**NOMEN** — AION's evolving naming system. Everything it encounters can be named; nothing has to keep the same name. Names emerge from observation, not from a list we supply. Each entity carries a permanent identity, a current name, and a lineage. In time, each AION names itself.

## The growth law

**Grow when insufficient. Consolidate when sufficient.**

An AION is not defined by how large it becomes, but by what it learns to become capable of.

## The experiment

Can an intelligence begin by borrowing a brain, learn how to use it, develop its own internal world and eventually become independent?

We intend to find out.

## Raise your own AION.

It's something.

It's anything.

It's yours.

---

## Technical notes (working repo)

*This section is working documentation for the researchers and agents gestating AION; the text above is the project's identity.*

### Current status (2026-11-25)

| Stage | Status |
|---|---|
| 00 — Seed | Pith-E2B artifact exists (post-QAT behaviour gate pending); Qwen3.5-4B-Base hosted on Lappy |
| 01 — Attachment | ✅ Interface proven on BOTH substrates: Qwen dense (4/4 smoke, bit-exact identity) + Pith factor-executed (4/4, bit-exact under bf16-forward-v1) |
| 02 — Embryonic Learning | ✅ Plasticity verified end-to-end (gates 1–4); baseline v2 frozen (`freeze/aion_learning_baseline_v2_20261125.tar.gz`) |
| 03 — Development | AION-1 Phase 1 run complete: **3/5** — retention, correction, transcript-removal PASS (bitwise restart; old facts intact under correction; purge-invariance); acquisition/generalisation honest FAIL (Base-model expression gap) — Phase 2 pending owner call |
| 04 — Dreaming | Diagnosed + bookkeeping repaired; real replay + reality-grounded evaluation deferred (owner decision); dream-side consolidation routing unblocked |
| 05 — Growth | DreamGrowthManager designed; need-based growth law locked |
| Emergence | The Disconnect Experiment armed (capability-survival metric) |
| Maturation | Beyond |

### Design laws

- **The learning objective and the generation objective must describe the same computation.**
- **Every useful experience must be capable of becoming AION's own knowledge.**
- **A dream cannot validate itself — and neither can a teacher.** Reality (executable environments, held-out observations) is the referee.
- **Plasticity first. Gate learning second.**
- **Grow when insufficient. Consolidate when sufficient.**
- **We define the conditions for growth, not the final form of the thing that grows.**
- Borrowed knowledge → Internalised knowledge → Independent capability.
- NOMEN safeguard: naming never changes an entity's underlying identity, and never promotes an imagined concept to a verified fact. Names are AION's evolving interpretation of its world.

### Repo layout

```
nnfs/                      # code home of AION (package: nnfs)
├── src/nnfs/              # NNFS learning mechanism (frozen baselines in freeze/)
│   ├── connections/       # PlasticRelationalConnection, RelationalNetwork
│   ├── dreaming/          # Dream Engine (diagnosed; bookkeeping repaired)
│   ├── benchmarks/        # validation benchmarks (lag-1 corrected)
│   └── experiments/       # pilot + validation runners
├── aion/
│   ├── foundation/        # pretrained loader (Qwen3.5-4B-Base, frozen)
│   ├── relational/        # γ-residual interface + Pith bridge
│   ├── dialogue/          # teacher/student protocol + lesson extractor
│   ├── worlds/            # synthetic world generator + question batteries
│   ├── memory/            # episodic / relational / consolidation provenance
│   ├── nomen/             # evolving naming, identity, lineage (NOMEN)
│   └── experiments/       # smoke tests, retention harness, AION-1 loop
├── freeze/                # SHA-tagged known-good baselines
└── pilot_results/         # preregistered experiment records
```

### Environment

- Python 3.13 (`C:\Python313\python.exe`), torch 2.11.0+cpu on Rog; foundation models on Lappy (192.168.0.33)
- Tests: `PYTHONPATH=src python -m pytest tests/ -q` (baseline: 100+ passed / 3 pre-existing training-env failures)
- Connection diagnostic: `PYTHONPATH=src python scripts/diagnostic_learning.py`
- Foundation host: Lappy `/mnt/seagate/aion-models/Qwen3.5-4B-Base` (CPU BF16)
