"""Frozen-foundation loader for AION: Qwen/Qwen3.5-4B-Base via HF Transformers.

Owner pivot 2026-11-24: AION's foundation is the FROZEN Qwen3.5-4B-Base; a small
learning system (aion/relational) grows around it. Base weights NEVER train.

Checkpoint facts (verified against the HF repo 2026-11-24):
  - Qwen/Qwen3.5-4B-Base, Apache 2.0, Base/pretrained-only (LoRA-PEFT friendly)
  - repo total 8.68 GiB BF16 across 2 safetensors shards, 738 tensors:
      model.language_model.*  426 tensors  (the text stack — what this lane loads)
      model.visual.*          297 tensors  (vision encoder — dropped here)
      mtp.*                    15 tensors  (multi-token prediction heads — dropped)
  - text stack: 32 layers, hidden 2560, FFN 9216, vocab 248,320,
    TIED embeddings, hybrid layout 8x(3x(Gated DeltaNet->FFN) -> 1x(Gated
    Attention->FFN)), native context 262,144 (extensible ~1M)
  - AutoModelForCausalLM maps this checkpoint to Qwen3_5ForCausalLM (text-only).

Load strategies (documented per owner brief):
  - CPU BF16 on LAPPY (chosen for now): 39 GB RAM / ~24 GB available; the ~7.5 GiB
    text stack loads fine. Lesson-scale forward passes are CPU-feasible.
  - 4-bit / 8-bit (bitsandbytes) on the 3060 (~3-4 GB VRAM): NOT installed on
    either host today; Linux-only wheel, would go into /opt/aion-seed/venv, and
    would need a gpu-manager lease like any GPU job. Future lane, not wired.
  - CPU BF16 on ROG: NOT viable now (15 GB total / ~6 GB available RAM, 8.8 GB
    free disk) — the checkpoint does not fit either RAM or disk on Rog today.
  - Checkpoint staging: /mnt/seagate/aion-models/Qwen3.5-4B-Base on Lappy
    (seagate mount, 28 GB free at staging time; aaron-writable).
  - NO llama-swap interaction of any kind: this is a plain HF checkpoint read,
    never the :8180 resident; no model flags touched (OWNER LAW).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.relational.interface import freeze_base, foundation_fingerprint  # noqa: E402

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
DEFAULT_LOCAL_DIR = "/mnt/seagate/aion-models/Qwen3.5-4B-Base"  # on Lappy
REPO_TOTAL_GIB = 8.68  # measured from the HF repo file listing


@dataclass
class FrozenBase:
    """Loaded foundation + its freeze evidence.

    `model` defaults to None so a handle can be built before the weights land
    (load_frozen_base fills it immediately after from_pretrained); every public
    accessor asserts it is loaded.
    """

    model: Optional[object] = None                  # transformers causal LM
    tokenizer: Optional[object] = None
    model_id: str = DEFAULT_MODEL_ID
    local_dir: Optional[str] = None
    dtype: str = "bfloat16"
    n_frozen_params: int = 0
    n_frozen_tensors: int = 0
    fingerprint: dict = field(default_factory=dict)  # bit-identity baseline

    def _require_model(self) -> object:
        if self.model is None:
            raise RuntimeError("FrozenBase has no model loaded (use load_frozen_base)")
        return self.model

    @property
    def hidden_size(self) -> int:
        return self._require_model().config.hidden_size

    def describe(self) -> dict:
        model = self._require_model()
        return {
            "model_id": self.model_id,
            "local_dir": self.local_dir,
            "dtype": self.dtype,
            "class": type(model).__name__,
            "n_frozen_params": self.n_frozen_params,
            "hidden_size": self.hidden_size,
            "n_layers": len(model.model.layers),
            "all_frozen": not any(p.requires_grad for p in model.parameters()),
        }


def check_disk_fit(target_dir: str, required_gib: float = REPO_TOTAL_GIB,
                   margin: float = 0.10) -> dict:
    """Report whether `target_dir`'s filesystem can hold the checkpoint.

    Requires free >= required * (1 + margin) — a raw pass with <10% headroom is
    reported as not-fits (never fill an OS drive to the brim).
    """
    import shutil
    usage = shutil.disk_usage(target_dir if os.path.isdir(target_dir) else os.path.dirname(target_dir) or ".")
    free_gib = usage.free / 2**30
    needed = required_gib * (1 + margin)
    return {"target": target_dir, "free_gib": round(free_gib, 2),
            "required_gib": required_gib, "headroom_needed_gib": round(needed, 2),
            "fits": free_gib >= needed}


def check_ram_fit(required_gib: float = REPO_TOTAL_GIB) -> dict:
    """Report whether host RAM can hold a BF16 load (with modest overhead)."""
    try:
        import psutil
        avail_gib = psutil.virtual_memory().available / 2**30
    except ImportError:
        avail_gib = None
    return {"available_gib": round(avail_gib, 2) if avail_gib else None,
            "required_gib": required_gib,
            "fits": bool(avail_gib and avail_gib >= required_gib)}


def load_frozen_base(local_dir: Optional[str] = None,
                     model_id: str = DEFAULT_MODEL_ID,
                     device: str = "cpu",
                     dtype: torch.dtype = torch.bfloat16,
                     tokenizer_id: Optional[str] = None) -> FrozenBase:
    """Load the text stack from Qwen3.5-4B-Base and FREEZE it. Asserts the freeze.

    `local_dir` (e.g. the seagate staging path) forces offline load. Memory-wise
    the text stack is ~7.5 GiB BF16; check_ram_fit() first on constrained hosts.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = local_dir or model_id
    offline = bool(local_dir)
    handle = FrozenBase(model_id=model_id, local_dir=local_dir,
                        dtype=str(dtype).split(".")[-1])

    handle.model = AutoModelForCausalLM.from_pretrained(
        src,
        dtype=dtype,  # transformers 5.x canonical name (torch_dtype deprecated)
        low_cpu_mem_usage=True,
        device_map=None,
        local_files_only=offline,
    ).to(device)
    handle.model.eval()

    try:
        handle.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id or src, local_files_only=offline)
    except Exception:
        handle.tokenizer = None  # tokenizer optional for hidden-state work

    stats = freeze_base(handle.model)  # asserts requires_grad=False everywhere
    handle.n_frozen_params = sum(p.numel() for p in handle.model.parameters())
    handle.n_frozen_tensors = stats["frozen_params"]
    handle.fingerprint = foundation_fingerprint(handle.model)
    assert not any(p.requires_grad for p in handle.model.parameters())
    return handle


if __name__ == "__main__":
    ap = __import__("argparse").ArgumentParser(description="Frozen Qwen3.5-4B-Base load check")
    ap.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-load", action="store_true", help="disk/RAM fit checks only")
    args = ap.parse_args()
    print("disk:", check_disk_fit(args.local_dir))
    print("ram :", check_ram_fit())
    if not args.no_load:
        h = load_frozen_base(local_dir=args.local_dir, device=args.device)
        print("loaded:", h.describe())
