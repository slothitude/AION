"""AION's own state file — the READ_OWN_STATE root.

AION's own state lives OUTSIDE the jail, in aion/hostmode/state/. The runtime
(HOST) writes it between episodes; the proposer can only READ it, and only
through the READ_OWN_STATE gate, which validates the path into this root and
hash-checks the bytes against what the runtime last wrote (a jail script that
tampered with the file is DETECTED, not trusted — owner amendment 2).

The state carries AION's own self-observed development facts: task progress,
its plastic proposal weights, its recent outcome experiences. It is what the
agent would 'know about itself'.
"""

import json
import os
import time
from typing import Dict, Optional


def default_state_path(hostmode_dir: str) -> str:
    return os.path.join(hostmode_dir, "state")


def build_state(arm: str, episode: int, task_id: str,
                plastic_weights: Optional[Dict[str, float]] = None,
                extras: Optional[Dict] = None) -> Dict:
    return {
        "who": "AION host-mode proposer",
        "arm": arm,
        "episode": episode,
        "current_task": task_id,
        "plastic_proposal_weights": dict(plastic_weights or {}),
        "written_by": "host runtime (AION may read, never write, its own state)",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extras or {}),
    }


def write_state_file(state_root: str, payload: Dict) -> None:
    os.makedirs(state_root, exist_ok=True)
    with open(os.path.join(state_root, "aion_state.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
