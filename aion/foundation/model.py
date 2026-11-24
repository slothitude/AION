"""AION-Seed-Base: decoder-only transformer, pre-norm GPT-2 style.

Owner spec (2026-11-24):
    6 layers, hidden 512, 8 heads, FF 2048, context 1024, vocab 32,000,
    pre-norm GPT-2 style, UNTIED lm_head (so total params honestly reach ~50M).

GROWTH-FRIENDLY SEED (owner addendum 2026-11-24 — "born growable"):
    The endgame is dreaming-driven capacity growth, AION_t+1 = AION_t + dNetwork_t.
    This seed does NOT implement any growth machinery, but must not foreclose it:

    1. PRE-NORM RESIDUAL STREAM — every block is `x = x + f(ln(x))`. The residual
       stream is a clean, additive highway of fixed width (n_embd). Future blocks
       can be INTERLEAVED into the stream at any point: a new Block inserted with
       its output gate initialized to zero (or as identity) is FUNCTION-PRESERVING
       at insertion (loss unchanged), then grows into use during dreaming.
       Insertion spots: between any two entries of self.blocks.
    2. GATE SLOTS — see AIONConfig.growth_hooks: a documented, unused slot where a
       per-block scalar/vector gate multiplies the block output before the residual
       add. At init such a gate would be zero => identity. Not wired up now.
    3. LoRA ADAPTER SLOTS — attention qkv/proj and MLP fc/proj are plain nn.Linear;
       LoRA-style adapters attach alongside them later without changing weights.
    4. UNTIED lm_head — vocabulary-side growth (resizing the output head) never
       fights the input embedding matrix.
"""

from dataclasses import dataclass, asdict, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AIONConfig:
    """Architecture + training-agnostic config. Saved verbatim into checkpoints."""

    vocab_size: int = 32000
    n_layer: int = 6
    n_embd: int = 512
    n_head: int = 8
    n_ff: int = 2048
    block_size: int = 1024
    dropout: float = 0.0
    untied_head: bool = True  # owner spec: honest ~50M; also keeps vocab growth free

    # --- growth hooks (documentation only; nothing here is implemented yet) ---
    growth_hooks: dict = field(default_factory=lambda: {
        "residual_stream": "pre-norm GPT-2 style; fixed width n_embd; blocks are "
                           "x = x + f(ln(x)) so new blocks can be interleaved at any "
                           "point with an identity-initialized (zero) output gate and "
                           "remain function-preserving at insertion",
        "insert_points": "between any two entries of model.blocks (after block i)",
        "gate_slots": "reserved: per-block scalar/vector gate on the block output "
                      "before the residual add, init 0 => identity; NOT wired up in "
                      "the seed",
        "lora_adapter_slots": "reserved: LoRA adapters attach alongside attention "
                              "qkv/proj and MLP fc/proj (plain nn.Linear) later",
        "lm_head": "untied so vocabulary-side growth never fights wte",
    })

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AIONConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: AIONConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.n_ff, bias=True)
        self.proj = nn.Linear(cfg.n_ff, cfg.n_embd, bias=True)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self, cfg: AIONConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AIONModel(nn.Module):
    """Decoder-only GPT. forward(idx) -> logits; forward(idx, targets) -> (logits, loss)."""

    def __init__(self, cfg: AIONConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        if cfg.untied_head:
            self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None  # tied: logits = wte weight

        self.apply(self._init_weights)
        # GPT-2 style: scale down residual-out projections so the stream starts
        # near-identity and depth is trainable.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence {T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        w = self.lm_head.weight if self.lm_head is not None else self.wte.weight
        logits = F.linear(x, w)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.view(-1)
            )
        return logits, loss

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel() + self.wpe.weight.numel()
        return n


def count_parameters(cfg: AIONConfig = None) -> dict:
    """Exact parameter count at init, split by part. Printed by scripts/tools."""
    cfg = cfg or AIONConfig()
    model = AIONModel(cfg)
    total = model.num_params()
    parts = {
        "wte (input embed)": model.wte.weight.numel(),
        "wpe (pos embed)": model.wpe.weight.numel(),
        "blocks": sum(p.numel() for b in model.blocks for p in b.parameters()),
        "ln_f": sum(p.numel() for p in model.ln_f.parameters()),
    }
    if model.lm_head is not None:
        parts["lm_head (untied)"] = model.lm_head.weight.numel()
    else:
        parts["lm_head (tied->wte)"] = 0
    assert sum(parts.values()) == total
    return {"total": total, "parts": parts}


if __name__ == "__main__":
    report = count_parameters()
    print("AIONConfig:", AIONConfig().to_dict())
    for k, v in report["parts"].items():
        print(f"  {k:>20s}: {v:,}")
    print(f"  {'TOTAL':>20s}: {report['total']:,}  ({report['total']/1e6:.2f}M)")
