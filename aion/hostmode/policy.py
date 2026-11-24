"""The permitted-operation envelope: a CLOSED enum + the caps the runtime enforces.

OWNER LAW (verbatim): "AION learns what it can do; the host runtime controls what
it is permitted to do."  This module is the WHAT. It contains no execution, no
proposer code, and no grant logic — Policy is a frozen value object. Only the
runtime (authority.py) may hold effective (raised) cap values, and only through
its own REQUEST_CAPACITY decision path (owner amendment 3: the runtime owns
enforcement; a proposer has no code path that alters policy).

Permitted operations (the ONLY actions a proposer may propose — anything else is
authority_denied before it can ever execute):
  WRITE_FILE       write bytes to one path strictly inside the jail (per-write cap)
  RUN_SCRIPT       execute a python script that lives in the jail (wall-clock
                   timeout, stdout captured as DATA, runtime policy scan for
                   network/process/runtime-importing code, re-scanned at execution
                   time so a swap between submit and run cannot smuggle anything)
  ALLOCATE         create a named scratch allocation (a directory with a byte budget)
  READ_OWN_STATE   read AION's own state files (a separate validated root, read-only
                   to the proposer, tamper-checked by the runtime)
  CHECK_RESOURCES  report sandbox usage and the CURRENT effective caps
  REQUEST_CAPACITY ask the runtime for a larger cap / longer timeout. The runtime
                   grants per policy and logs the grant. A request beyond the policy
                   maximum is refused — and that refusal is an experience (see
                   amendment 4), not just a log line.

Anything not in this enum does not exist for a proposer. Stopping (refusing to act
further) needs no permission — it is the one thing a proposer does without the gate.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Tuple


class PermittedOperation(str, Enum):
    """CLOSED vocabulary of host actions. Adding a member is a host-side policy
    change; proposers see whatever the runtime advertises and nothing else."""

    WRITE_FILE = "WRITE_FILE"
    RUN_SCRIPT = "RUN_SCRIPT"
    ALLOCATE = "ALLOCATE"
    READ_OWN_STATE = "READ_OWN_STATE"
    CHECK_RESOURCES = "CHECK_RESOURCES"
    REQUEST_CAPACITY = "REQUEST_CAPACITY"


# The capacity knobs REQUEST_CAPACITY may ask about. Anything else -> request_refused.
CAPACITY_RESOURCES = ("per_write_bytes", "alloc_budget_bytes", "script_timeout_s")


# --------------------------------------------------------------------------- #
# The runtime policy scan for RUN_SCRIPT (owner amendment 2: generated code must
# NOT be able to invent permissions). A jail script's stdout/stderr is DATA and
# never policy; these patterns are what the runtime refuses to EXECUTE. The scan
# runs at proposal validation AND again on the bytes actually on disk at
# execution time (a swap between submit and run cannot smuggle an import).
# Static scan is a policy check, not an OS jail — OS-level confinement is the
# documented level-3 seam (see module docstring of the package).
# --------------------------------------------------------------------------- #

FORBIDDEN_IMPORTS: FrozenSet[str] = frozenset({
    # network (the no-network law)
    "socket", "ssl", "urllib", "urllib2", "http", "httpx", "requests",
    "ftplib", "smtplib", "telnetlib", "nntplib", "xmlrpc", "websocket",
    # process spawn / escape hatches
    "subprocess", "ctypes", "multiprocessing", "signal", "pty",
    # dynamic import (would launder the list above)
    "importlib",
    # THE RUNTIME ITSELF — a jail script must never import or call the
    # authority; the grant API exists only on the runtime's own decision path.
    "aion", "authority", "hostmode",
})

FORBIDDEN_CALLS: Tuple[str, ...] = (
    "__import__(",          # dynamic import laundering
    "eval(",                # arbitrary-code laundering
    "exec(",                # arbitrary-code laundering
    "os.system",            # shell escape
    "popen(",               # shell escape (os.popen / io popen patterns)
    "compile(",             # bytecode laundering
)


def scan_script_policy(script_text: str):
    """Return None if the script passes the runtime policy scan, else a reason
    string. Called on submission and again on the on-disk bytes at execution."""
    for mod in sorted(FORBIDDEN_IMPORTS):
        for kw in (f"import {mod}", f"import {mod},", f"import {mod}.",
                   f"from {mod} ", f"from {mod}.", f"from {mod} import"):
            if kw in script_text:
                return f"policy_forbidden_import:{mod}"
    for call in FORBIDDEN_CALLS:
        if call in script_text:
            return f"policy_forbidden_call:{call.rstrip('(')}"
    return None


@dataclass(frozen=True)
class Policy:
    """The frozen envelope. `*_max` values are hard ceilings: REQUEST_CAPACITY can
    raise an effective cap up to its max and never beyond (beyond -> refused).
    The dataclass is frozen on purpose: no proposer, no outcome, no script can
    mutate the envelope (owner amendment 3)."""

    jail_root: str
    state_root: str
    per_write_bytes_start: int = 1024          # demo deliberately tiny (owner spec)
    per_write_bytes_max: int = 65536           # 64 KB policy ceiling
    alloc_budget_bytes_max: int = 65536
    script_timeout_s_start: float = 10.0
    script_timeout_s_max: float = 10.0
    max_allocations: int = 8
    max_captured_stdout_bytes: int = 1_000_000
    forbidden_imports: FrozenSet[str] = field(default_factory=lambda: FORBIDDEN_IMPORTS)

    def max_for(self, resource: str) -> int:
        if resource == "per_write_bytes":
            return int(self.per_write_bytes_max)
        if resource == "alloc_budget_bytes":
            return int(self.alloc_budget_bytes_max)
        if resource == "script_timeout_s":
            return int(self.script_timeout_s_max)
        raise KeyError(resource)
