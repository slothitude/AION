"""HOST MODE — AION's first measured host-manipulation capability.

OWNER LAW (2026-11-25, verbatim): "AION learns what it can do; the host runtime
controls what it is permitted to do."

The split of power in this package is the whole point:
  - aion/hostmode/policy.py     the PERMITTED-OPERATION closed enum + caps (the envelope)
  - aion/hostmode/authority.py  the RUNTIME — the authority gate. It validates every
                                proposal, enforces every cap, and it is the ONLY thing
                                that acts. A proposer can never execute anything and
                                can never self-authorize.
  - aion/hostmode/proposer.py   three proposer arms (FixedRandom / Heuristic / Plastic).
                                They only PROPOSE. Plastic adapts proposal weights from
                                outcome feedback, local to this directory.
  aion/hostmode/task.py         the demonstration task + host-side verifier
  aion/hostmode/state.py        AION's own state file (the READ_OWN_STATE root)
  aion/hostmode/experiment.py   the 20-episode x 3-arm experiment + metrics
  aion/hostmode/run_demo.py     entry point: runs everything, folds devhistory, reports
  aion/hostmode/test_hostmode.py  tests: gate enforcement, caps, grants, adaptation
  aion/hostmode/sandbox/        THE JAIL ROOT. Every proposer-touched path is resolved
                                and validated strictly inside it.

SEAMS (documented, not built here):
  - NNFS plasticity: PlasticProposer adapts weights with a local update rule. The
    follow-up seam for the connections lane's adapter is `PlasticProposer.observe`
    (outcome -> signal) and `PlasticProposer.weights` (state). Swapping the local
    rule for real plastic updates must not change the runtime or the gate.
  - Lane K (aion/nav): the proposer-integration seam is `BaseProposer.choose(
    observation) -> Proposal` plus the `Observation` shape in proposer.py. NAV
    lanes can propose into the same gate unchanged; this package never imports
    aion/nav (read-only lane, do not collide).
  - DEVHISTORY (aion/memory/devhistory.py): landmark events are folded through
    `DevelopmentalHistory.record_growth(...)`; see run_demo.fold_devhistory().
"""

from aion.hostmode.policy import PermittedOperation, Policy, CAPACITY_RESOURCES
from aion.hostmode.authority import HostRuntime, Proposal, Outcome
from aion.hostmode.proposer import (
    FixedRandomProposer, HeuristicProposer, PlasticProposer, Observation,
)
from aion.hostmode.task import HostTask

__all__ = [
    "PermittedOperation", "Policy", "CAPACITY_RESOURCES",
    "HostRuntime", "Proposal", "Outcome",
    "FixedRandomProposer", "HeuristicProposer", "PlasticProposer", "Observation",
    "HostTask",
]
