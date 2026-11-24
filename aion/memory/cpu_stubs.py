"""CPU-ONLY STUBS for the build-ahead unit tests — never for real runs.

Everything here exists so the retention/provenance apparatus can be constructed
and unit-tested on CPU without a real model, a real tokenizer, or any endpoint
call (owner law: no real model/endpoint calls in tests; everything seeded and
deterministic). The real AION-1 runs swap these out for the frozen Qwen3.5-4B
foundation via aion/foundation/pretrained_loader.py.

  build_stub_base  — tiny 4-layer Llama stand-in with the module-tree shape the
                     interface discovers (layers ModuleList, hidden_size,
                     get_output_embeddings, causal LM logits).
  StubVocab        — deterministic, cross-process-stable invertible word<->id
                     table (md5-derived ids with linear-probe collision escape).
  DirectFeed       — WITHOUT-host probe encoder: prompt ids -> deterministic
                     (pre, post) observation states via a seeded projector. The
                     host model is never touched (the disconnect condition).
  normalize_answer — gold-vs-answer comparison normalisation.
"""

import hashlib
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

RESERVED_IDS = 4  # ids [0, RESERVED) left for pad/bos/eos/unk like a real vocab


def build_stub_base(seed: int = 0, vocab_size: int = 128, hidden_size: int = 64,
                    n_layers: int = 4):
    """Tiny in-memory causal LM with the structure RelationalInterface needs."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size, hidden_size=hidden_size, intermediate_size=128,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg)


class StubVocab:
    """Deterministic invertible word<->id map.

    id = RESERVED + (md5(word) % (vocab_size - RESERVED)), collisions resolved by
    linear probing; the table is recorded so decode() is exact. Same word -> same
    id in any process, any insertion order (probing is per-word deterministic).
    """

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = int(vocab_size)
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}

    def _base_id(self, word: str) -> int:
        h = hashlib.md5(word.encode("utf-8")).hexdigest()
        return RESERVED_IDS + (int(h, 16) % (self.vocab_size - RESERVED_IDS))

    def encode_word(self, word: str) -> int:
        if word in self.word2id:
            return self.word2id[word]
        wid = self._base_id(word)
        while wid in self.id2word:                      # collision: probe upward
            wid = (wid + 1) % self.vocab_size
        self.word2id[word] = wid
        self.id2word[wid] = word
        return wid

    def encode(self, text: str) -> torch.Tensor:
        ids = [self.encode_word(t) for t in text.split()]
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)   # (1, T)

    def encode_ids(self, text: str) -> List[int]:
        return [self.encode_word(t) for t in text.split()]

    def decode(self, token_id: int) -> str:
        return self.id2word.get(int(token_id), f"<tok{int(token_id)}>")

    def decode_ids(self, ids: List[int]) -> str:
        return " ".join(self.decode(i) for i in ids)


class DirectFeed:
    """WITHOUT-host probe encoder.

    Host disconnected ENTIRELY: no model, no hidden states, no hooks. Prompt ids
    go through a fixed seeded projector (vocab x dim) — deterministic direct
    observations only. pre uses the id sequence as-is; post uses the reversed
    sequence so pre/post states differ (the connection needs both sides).
    """

    def __init__(self, vocab_size: int, dim: int, seed: int = 99):
        g = torch.Generator().manual_seed(seed)
        self.projector = torch.randn(vocab_size, dim, generator=g) / (dim ** 0.5)
        self.dim = int(dim)

    def encode(self, prompt_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [int(i) for i in prompt_ids] or [0]
        idx = torch.tensor(ids, dtype=torch.long)
        pre = F.normalize(self.projector[idx].mean(dim=0), dim=0)
        post = F.normalize(self.projector[idx.flip(0)].mean(dim=0), dim=0)
        return pre, post


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().replace("?", " ").replace(".", " ")
                    .replace(",", " ").split())


def answer_matches(predicted: str, gold: str) -> bool:
    return bool(predicted) and normalize_answer(predicted) == normalize_answer(gold)
