"""AION-1 student side: extracted relationship facts -> plastic training signal; questions -> answers.

Owner brief (2026-11-24): this file COORDINATES, the pathway LEARNS. It renders extracted
relationship facts into training examples for the relational pathway (aion/relational/
interface.py public API - imported READ-ONLY, the frozen mechanism), runs the pathway's NNFS
plasticity on them, queries the pathway for answers and formats them. The interface is thin on
purpose: any object exposing forward(input_ids, attention_mask) -> {"logits": ...} and
nnfs_update(input_ids) -> stats works (the real RelationalInterface, or a stub in tests).
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aion.dialogue.lesson_extractor import Fact  # noqa: E402  (read-only sibling)

RENDER_TEMPLATES = {
    "operates": "{subject} operates {object}.",
    "routes-to": "{subject} routes to {object}.",
    "controls": "{subject} controls {object}.",
    "produces": "{subject} produces {object}.",
}


@dataclass
class StudentConfig:
    answer_tokens: int = 8            # greedy continuation budget when answering
    gate_steps: int = 0               # extra autograd gate steps per fact (0 = plasticity only)
    correction_repeats: int = 3       # plastic repetition weight for corrected facts
    top_k_report: int = 5             # first-step top-k kept in the answer meta


@dataclass
class StudentAnswer:
    """The student's formatted answer plus what the pathway actually produced."""

    question: str
    text: str                         # decoded greedy continuation
    token_ids: List[int]
    stop_reason: str                  # "budget" | "eos" | "sentence_end"
    meta: Dict = field(default_factory=dict)


class Student:
    """Coordinates fact -> training text -> pathway plasticity, and question -> answer text."""

    def __init__(self, pathway, tokenizer=None, cfg: Optional[StudentConfig] = None):
        self.pathway = pathway
        self.tokenizer = tokenizer
        self.cfg = cfg or StudentConfig()
        self.n_facts_learned = 0
        self.learned_texts: List[str] = []

    # -- fact rendering ------------------------------------------------------ #

    def render_fact(self, fact: Fact) -> str:
        """Fact -> the canonical training sentence fed to the pathway."""
        if fact.relation == "converts-to":
            if fact.agent:
                return f"{fact.agent} converts {fact.subject} into {fact.object}."
            return f"{fact.subject} converts into {fact.object}."
        template = RENDER_TEMPLATES.get(fact.relation)
        if template is None:
            raise ValueError(f"no render template for relation {fact.relation!r}")
        return template.format(subject=fact.subject, object=fact.object)

    def encode(self, text: str) -> List[List[int]]:
        if self.tokenizer is None:
            raise ValueError("Student needs a tokenizer (encode/decode) to talk to the pathway")
        return [self.tokenizer.encode(text)]

    # -- learning (facts -> PLASTIC pathway, never just a transcript row) ------ #

    def learn_fact(self, fact: Fact, repeats: int = 1) -> Dict:
        """Render the fact and drive the pathway's NNFS plasticity on it (repeats = weight)."""
        text = self.render_fact(fact)
        ids = torch.tensor(self.encode(text), dtype=torch.long)
        nnfs, gate = [], None
        for _ in range(max(1, repeats)):
            res = self.pathway.nnfs_update(ids)
            nnfs.append(res.get("nnfs", res) if isinstance(res, dict) else res)
            if self.cfg.gate_steps > 0 and hasattr(self.pathway, "gate_step"):
                gate = [self.pathway.gate_step(ids) for _ in range(self.cfg.gate_steps)]
        self.n_facts_learned += 1
        self.learned_texts.append(text)
        return {"text": text, "n_tokens": int(ids.shape[1]), "repeats": repeats,
                "nnfs": nnfs, "gate": gate}

    # -- answering (query the pathway, format the answer) ---------------------- #

    def ask(self, question: str) -> StudentAnswer:
        """Greedy continuation from the pathway's next-token logits, sentence-terminated."""
        if self.tokenizer is None:
            raise ValueError("Student needs a tokenizer to answer questions")
        cur = self.encode(question)[0]
        generated: List[int] = []
        stop_reason, first_top = "budget", None
        for _ in range(self.cfg.answer_tokens):
            ids = torch.tensor([cur], dtype=torch.long)
            res = self.pathway.forward(ids, attention_mask=torch.ones_like(ids))
            logits = res["logits"] if isinstance(res, dict) else res.logits
            step = logits[0, -1]
            if first_top is None:
                k = min(self.cfg.top_k_report, step.shape[-1])
                top = torch.topk(step, k)
                first_top = [{"token_id": int(i), "logit": float(v)}
                             for v, i in zip(top.values.tolist(), top.indices.tolist())]
            tid = int(step.argmax())
            piece = self.tokenizer.decode([tid])
            if piece == "" or tid == getattr(self.tokenizer, "eos_id", -1):
                stop_reason = "eos"
                break
            generated.append(tid)
            cur.append(tid)
            if "." in piece or "\n" in piece:
                stop_reason = "sentence_end"
                break
        text = (self.tokenizer.decode(generated) if generated else "").strip()
        return StudentAnswer(question=question, text=text, token_ids=generated,
                             stop_reason=stop_reason, meta={"first_step_top": first_top})

    def answer(self, question: str) -> str:
        return self.ask(question).text
