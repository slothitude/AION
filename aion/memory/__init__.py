"""AION-1 retention/provenance apparatus.

Owner brief (2026-11-24): the memory-of-experiences apparatus around the plastic
relational pathway. Four modules:

  state          — exact serialization of AION-side plastic state (save/restart)
  episodic       — experience store (lessons/observations/corrections + provenance)
  relational     — the owned relational knowledge index over plastic state
  consolidation  — wrapper around the FROZEN consolidation machinery (call, never
                   reimplement) + the output-preservation invariant (1e-6)
  cpu_stubs      — CPU-only stubs for build-ahead unit tests (never for real runs)

LAWS honoured here:
  - src/nnfs/** is FROZEN: imported read-only, never edited.
  - aion/relational/interface.py and aion/foundation/pretrained_loader.py are
    landed and imported read-only.
  - no dream-engine integration anywhere (stub hook only, see retention.py).
  - no real model/endpoint calls in tests; everything is seeded and deterministic.
"""

from aion.memory.state import PlasticStateSnapshot, conn_tensor_map
from aion.memory.episodic import EpisodicStore, Episode, PURGE_TOMBSTONE
from aion.memory.relational import RelationalMemory, RelationRecord, relation_key
from aion.memory.consolidation import ConsolidationLedger

__all__ = [
    "PlasticStateSnapshot",
    "conn_tensor_map",
    "EpisodicStore",
    "Episode",
    "PURGE_TOMBSTONE",
    "RelationalMemory",
    "RelationRecord",
    "relation_key",
    "ConsolidationLedger",
]
