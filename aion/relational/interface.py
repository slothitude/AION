"""Relational interface: model-aware hidden-state hooks + plastic relational residual
on top of a FROZEN foundation model (Qwen/Qwen3.5-4B-Base).

Architecture (owner Stage 1 + Stage 2, 2026-11-24):

    base (FROZEN, no_grad)  ->  hidden states via forward hooks
                    v
    PlasticRelationalConnection per hooked layer pair  (NNFS online learning,
    src.nnfs.connections.plastic — gate-4-verified, read-only import)
                    v
    r  = transmitted relational feature
    h' = h + gamma * r                gamma = ZERO-INIT residual gate (the ONLY
                                      thing zero-initialised — internal features
                                      stay non-zero capable; identity at insertion)
    logits' = base_logits + (gamma * r) @ W_head.T     <- REACH: lands in the
                                      ACTUAL language-model logits, not just an
                                      intermediate hidden tensor.

Training ownership (strictly separated):
  - foundation weights: NEVER train (requires_grad=False everywhere, fingerprinted
    before/after to prove bit-identity).
  - gamma (residual gate): autograd-trained (zero-init, so insertion is identity).
  - u_fast (fast plastic weights): updated ONLY by the NNFS plasticity rule
    (FastSlowPlasticity.plasticity_step inside PlasticRelationalConnection),
    autograd-disabled. relationship.W likewise autograd-disabled so the NNFS
    update is the sole driver of feature change.

Model-awareness: the decoder layer list is DISCOVERED from the real module tree
(Qwen3.5 is HYBRID — Gated DeltaNet and Gated Attention blocks alternate; never
assume every block is attention). Hooks attach to the selected module objects and
capture their hidden-state outputs, with the block class name recorded.
"""

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PLASTIC_PATH = os.path.join(_REPO_ROOT, "src", "nnfs", "connections", "plastic.py")

# Read-only import of the gate-4-verified plasticity code. Loaded directly by file
# path (not package import) so src/nnfs package-wide re-inits by other lanes never
# break this module; plastic.py itself is frozen.
_spec = importlib.util.spec_from_file_location("aion_frozen_plastic", _PLASTIC_PATH)
_plastic_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("aion_frozen_plastic", _plastic_mod)
_spec.loader.exec_module(_plastic_mod)
PlasticRelationalConnection = _plastic_mod.PlasticRelationalConnection


# --------------------------------------------------------------------------- #
# Foundation: freeze + fingerprint
# --------------------------------------------------------------------------- #

def freeze_base(model: nn.Module) -> Dict[str, int]:
    """Set requires_grad=False on EVERY foundation parameter and assert it."""
    n = 0
    numel = 0
    for p in model.parameters():
        p.requires_grad_(False)
        assert p.requires_grad is False
        n += 1
        numel += p.numel()
    return {"frozen_params": n, "frozen_numel": numel}


def foundation_fingerprint(model: nn.Module) -> Dict[str, tuple]:
    """Cheap but strict bit-identity evidence for the foundation.

    Per parameter: numel, fp64 sum, fp64 abs-sum, and a sha256 over the first
    4 KiB of raw bytes. Any bit flip in a sampled prefix or any value change
    moves the stats. Deterministic, order-stable.
    """
    fp: Dict[str, tuple] = {}
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            t = p.detach()
            head = t.reshape(-1)[:2048].contiguous()
            fp[name] = (
                t.numel(),
                float(t.sum(dtype=torch.float64).item()),
                float(t.abs().sum(dtype=torch.float64).item()),
                hashlib.sha256(head.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
                if head.numel() else "",
            )
    return fp


def fingerprints_equal(a: Dict[str, tuple], b: Dict[str, tuple]) -> Tuple[bool, List[str]]:
    if a.keys() != b.keys():
        return False, ["param name sets differ"]
    diffs = [k for k in a if a[k] != b[k]]
    return (not diffs), diffs


# --------------------------------------------------------------------------- #
# Model-aware decoder layer discovery
# --------------------------------------------------------------------------- #

@dataclass
class DiscoveredLayer:
    name: str
    module: nn.Module
    class_name: str
    family: str   # block family: "deltanet" | "attention" | "other" — derived from
                  # the layer's INNER submodule classes (hybrid stacks wrap every
                  # block in a uniform DecoderLayer)
    index: int


def _layer_family(layer: nn.Module) -> str:
    names = {type(sm).__name__.lower() for sm in layer.modules()}
    if any("deltanet" in n or "delta_net" in n for n in names):
        return "deltanet"
    if any("attention" in n for n in names):
        return "attention"
    return "other"


def discover_decoder_layers(model: nn.Module) -> List[DiscoveredLayer]:
    """Find the decoder-layer ModuleList in the real module tree.

    Heuristic (model-aware, not architecture-assuming): the largest ModuleList
    exposed as an attribute named `layers` on some module in the tree. Works for
    Llama-style stacks and for Qwen3.5's hybrid text stack regardless of config
    nesting (text_config etc.).
    """
    best_name, best_list = None, None

    def scan(mod: nn.Module, prefix: str):
        nonlocal best_name, best_list
        child = getattr(mod, "layers", None)
        if isinstance(child, nn.ModuleList) and len(child) >= 2:
            if best_list is None or len(child) > len(best_list):
                best_list, best_name = child, f"{prefix}layers"
        for name, c in mod.named_children():
            if c is not mod:
                scan(c, f"{prefix}{name}.")

    scan(model, "")
    if best_list is None:
        raise RuntimeError("no `layers` ModuleList found in module tree")
    return [
        DiscoveredLayer(name=f"{best_name}.{i}", module=m,
                        class_name=type(m).__name__, family=_layer_family(m), index=i)
        for i, m in enumerate(best_list)
    ]


def pick_hook_pairs(layers: List[DiscoveredLayer]) -> List[Tuple[int, int]]:
    """Pick (pre_layer_idx, post_layer_idx) pairs covering BOTH hybrid block
    families when they exist (DeltaNet + attention), else adjacent pairs."""
    pairs: List[Tuple[int, int]] = []
    used = set()

    def first_with(*keys):
        for i, l in enumerate(layers):
            if any(k == l.family or k in l.family for k in keys):
                return i
        return None

    for fam in ("deltanet", "attention"):
        i = first_with(fam)
        if i is not None and i not in used and (i + 1) < len(layers):
            pairs.append((i, i + 1))
            used.add(i)
    if not pairs:  # generic fallback: first adjacent pairs
        for i in range(0, min(2, len(layers) - 1)):
            pairs.append((i, i + 1))
    last = (len(layers) - 2, len(layers) - 1)
    if len(layers) >= 2 and last not in pairs:
        pairs.append(last)  # always also cover the deepest pair
    return pairs


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #

@dataclass
class InterfaceConfig:
    trace_decay: float = 0.95
    learning_rate: float = 0.1        # NNFS plasticity lr (u_fast owns its own scale)
    blend_ratio: float = 0.7
    gate_lr: float = 1e-2             # SGD step size for the residual gate gamma
    consolidation_interval: int = 1000
    relation_type: str = "post"       # frozen codebase's prescribed relation for
                                      # self-prediction targets (bilinear zeroes the
                                      # expected error-driven gradient there)
    normalize_states: bool = True     # unit-L2 relational states: scale-invariant
                                      # features and active-mask-safe psi magnitudes


class RelationalInterface(nn.Module):
    """Plastic relational residual pathway around a frozen foundation model.

    Only the zero-init residual `gates` carry autograd gradients; NNFS plasticity
    owns u_fast / theta_slow / relationship.W (autograd-disabled there).
    """

    def __init__(self, base_model, cfg: Optional[InterfaceConfig] = None,
                 layer_pairs: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.base = base_model
        self.cfg = cfg or InterfaceConfig()
        self.layers = discover_decoder_layers(self.base)
        self.pairs = layer_pairs or pick_hook_pairs(self.layers)
        self.layer_types = {i: self.layers[i].class_name for i, _ in self.pairs}
        self.hidden_dim = self._infer_hidden_dim()

        self.conns = nn.ModuleList()
        self.gates = nn.ParameterList()
        for (pre_i, post_i) in self.pairs:
            conn = PlasticRelationalConnection(
                pre_idx=pre_i, post_idx=post_i,
                pre_dim=self.hidden_dim, post_dim=self.hidden_dim,
                context_dim=0, relation_type=self.cfg.relation_type,
                learning_rate=self.cfg.learning_rate,
                trace_decay=self.cfg.trace_decay,
                blend_ratio=self.cfg.blend_ratio,
                consolidation_interval=self.cfg.consolidation_interval,
            )
            for p in conn.parameters():
                p.requires_grad_(False)          # NNFS owns these
            self.conns.append(conn)
            self.gates.append(nn.Parameter(torch.zeros(1)))  # ZERO-INIT gate

        self._hooks = []
        self._captured: Dict[int, torch.Tensor] = {}
        self._attach_hooks()
        self.freeze_stats = freeze_base(self.base)

    # -- setup -------------------------------------------------------------- #

    def _infer_hidden_dim(self) -> int:
        c = getattr(self.base, "config", None)
        for holder in (c, getattr(c, "text_config", None)):
            v = getattr(holder, "hidden_size", None)
            if isinstance(v, int):
                return v
        raise RuntimeError("cannot infer hidden size from config")

    def _make_capture(self, layer_idx: int):
        def hook(module, args, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            self._captured[layer_idx] = h.detach()
        return hook

    def _attach_hooks(self):
        for idx in sorted({i for pair in self.pairs for i in pair}):
            h = self.layers[idx].module.register_forward_hook(self._make_capture(idx))
            self._hooks.append(h)

    # -- forward ------------------------------------------------------------ #

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                learn: bool = False) -> Dict:
        """Run frozen base, extract hooked hidden states, build relational logits."""
        assert not any(p.requires_grad for p in self.base.parameters()), \
            "foundation must be fully frozen"
        self._captured.clear()
        with torch.no_grad():
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = out.logits if hasattr(out, "logits") else out[0]

        hidden = dict(self._captured)  # layer idx -> (B, T, D), detached
        for pair in self.pairs:
            for i in pair:
                assert i in hidden, f"hook did not fire for layer {i}"

        feats, contribs, stats = [], [], {}
        residual = torch.zeros(input_ids.shape[0], self.hidden_dim,
                               device=input_ids.device, dtype=base_logits.dtype)
        for k, (pre_i, post_i) in enumerate(self.pairs):
            pre_s = hidden[pre_i].mean(dim=1).float()    # (B, D) seq-mean states
            post_s = hidden[post_i].mean(dim=1).float()
            if self.cfg.normalize_states:
                pre_s = nn.functional.normalize(pre_s, dim=-1)   # unit-L2
                post_s = nn.functional.normalize(post_s, dim=-1)
            with torch.no_grad():
                r, cstats = self.conns[k](pre_s, post_s,
                                          target=post_s if learn else None,
                                          learn=learn)
            if learn:
                stats[f"pair{k}"] = cstats
            r = r.to(base_logits.dtype)
            feats.append(r)
            contribs.append(self.gates[k] * r)           # gamma * r (grad -> gamma)
            residual = residual + self.gates[k] * r

        # REACH: logits-level contribution through the (frozen) output head
        head = self.base.get_output_embeddings()
        assert head is not None, "model exposes no output embedding/head"
        logits_delta = residual.to(head.weight.dtype) @ head.weight.T  # (B, V)
        logits = base_logits + logits_delta.unsqueeze(1)  # broadcast over sequence

        return {
            "logits": logits,
            "base_logits": base_logits,
            "logits_delta": logits_delta,
            "hidden": hidden,
            "feats": feats,
            "contributions": contribs,
            "nnfs_stats": stats,
        }

    # -- learning ------------------------------------------------------------ #

    def nnfs_update(self, input_ids, attention_mask=None) -> Dict:
        """One NNFS online-learning pass (plasticity only; no autograd)."""
        res = self.forward(input_ids, attention_mask, learn=True)
        return {"nnfs": res["nnfs_stats"]}

    def gate_step(self, input_ids, attention_mask=None) -> Dict:
        """One autograd SGD step on the residual gates via next-token loss.

        Only the zero-init gates train; foundation and NNFS weights do not.
        """
        res = self.forward(input_ids, attention_mask, learn=False)
        logits = res["logits"][:, :-1, :]
        targets = input_ids[:, 1:]
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        loss.backward()
        with torch.no_grad():
            for g in self.gates:
                if g.grad is not None:
                    g -= self.cfg.gate_lr * g.grad
        self.zero_grad(set_to_none=True)
        return {"loss": float(loss.item()),
                "gate_norms": [float(g.item()) for g in self.gates]}

    # -- reporting ------------------------------------------------------------ #

    def describe(self) -> Dict:
        return {
            "n_layers_discovered": len(self.layers),
            "layer_class_names": sorted({l.class_name for l in self.layers}),
            "layer_families": [l.family for l in self.layers],
            "hooked_pairs": [
                {"pre": self.layers[a].name, "post": self.layers[b].name,
                 "pre_type": self.layers[a].class_name,
                 "post_type": self.layers[b].class_name,
                 "pre_family": self.layers[a].family,
                 "post_family": self.layers[b].family}
                for (a, b) in self.pairs
            ],
            "hidden_dim": self.hidden_dim,
            "frozen_params": self.freeze_stats["frozen_params"],
            "frozen_numel": self.freeze_stats["frozen_numel"],
            "trainable": [n for n, p in self.named_parameters() if p.requires_grad],
        }
