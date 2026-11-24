"""Teacher client for AION-1 dialogue: an OpenAI-compatible chat endpoint, structured protocol.

Owner brief (2026-11-24): the teacher side of the dialogue apparatus. The intended default
endpoint is the 4B teacher lane on Lappy :8180 — this is a plain HTTP client only. It NEVER
reads llama-swap config and NEVER touches model flags (OWNER LAW); the caller supplies the
model alias. Structured protocol first: the owner specified structured conversation, not
free-form chat, for a small student — every teacher turn is requested as strict JSON with an
action-scoped schema, parsed tolerantly (code fences / prose tolerated, raw text kept).

Network seam: post_chat() is the single HTTP call — tests override it (or use ScriptedTeacher)
so NOTHING here ever hits a real endpoint from the suite.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# intended default: the 4B teacher lane on Lappy (:8180); never probed by tests
DEFAULT_BASE_URL = "http://192.168.0.33:8180/v1"

SYSTEM_PROMPT = (
    "You are the TEACHER in a structured lesson protocol for a small student model. "
    "Reply with STRICT JSON only - no prose outside the JSON object. "
    "Schema by action:\n"
    '  teach:    {"action":"teach","lesson":"<repeated lesson>","facts":[{"subject":...,'
    '"relation":...,"object":...}]}\n'
    '  answer:   {"action":"answer","question":"<question>","answer":"<short answer>"}\n'
    '  correct:  {"action":"correct","mistake":"<wrong student answer>","correction":'
    '"<right fact, e.g. X routes to relay Y, not Z>"}\n'
    "Keep answers to a single short entity name or phrase. "
    "Never invent entities outside the lesson."
)


@dataclass
class TeacherConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = ""                    # caller supplies the alias; this client never reads
                                       # llama-swap config (OWNER LAW)
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_s: float = 60.0
    json_mode: bool = True             # sends response_format={"type":"json_object"}


@dataclass
class TeacherTurn:
    """One teacher response: raw content plus the parsed structured payload when available."""

    action: str                        # teach | answer | correct
    content: str                       # raw assistant text
    parsed: Optional[Dict] = None      # strict-JSON payload when it parsed
    structured: bool = False           # True when the JSON payload carried the schema fields
    error: Optional[str] = None
    meta: Dict = field(default_factory=dict)


def parse_structured(content: str) -> Optional[Dict]:
    """Tolerant strict-JSON extraction: direct parse, then code-fenced, then first {...} block."""
    if not content:
        return None
    candidates = [content.strip()]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1))
    else:
        brace = re.search(r"\{.*\}", content, re.DOTALL)
        if brace:
            candidates.insert(0, brace.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


class TeacherClient:
    """Thin OpenAI-compatible chat client with the three protocol actions."""

    def __init__(self, config: Optional[TeacherConfig] = None):
        self.config = config or TeacherConfig()

    # -- protocol actions ---------------------------------------------------- #

    def teach(self, lesson_text: str) -> TeacherTurn:
        return self._action("teach", [
            {"role": "user", "content": f"TEACH this lesson: {lesson_text}\n"
                                        "Return the teach-schema JSON with its facts."}])

    def answer(self, question: str) -> TeacherTurn:
        """Teacher's authoritative answer to a question (the grading reference)."""
        return self._action("answer", [
            {"role": "user", "content": f"ANSWER this question about the taught lesson: "
                                        f"{question}\nReturn the answer-schema JSON."}])

    def correct(self, mistake: str, correction: str) -> TeacherTurn:
        return self._action("correct", [
            {"role": "user", "content": f"The student answered wrongly. Mistake: {mistake}\n"
                                        f"Give the correction: {correction}\n"
                                        "Return the correct-schema JSON."}])

    # -- plumbing ------------------------------------------------------------ #

    def _action(self, action: str, user_messages: List[Dict[str, str]]) -> TeacherTurn:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages
        try:
            content = self.chat(messages)
        except Exception as exc:                       # network/HTTP failure -> typed turn
            return TeacherTurn(action=action, content="", error=f"{type(exc).__name__}: {exc}")
        parsed = parse_structured(content)
        structured = bool(parsed) and all(k in parsed for k in ("action",)) \
            and parsed.get("action") == action
        return TeacherTurn(action=action, content=content, parsed=parsed, structured=structured)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Full completion round-trip; returns assistant text. Override point for stubs."""
        payload = self.post_chat({
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        })
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("teacher endpoint returned no choices")
        return choices[0].get("message", {}).get("content", "")

    def post_chat(self, payload: Dict) -> Dict:
        """Single POST {base_url}/chat/completions. The ONLY network call in this module."""
        import requests                                # imported lazily: tests never need it

        cfg = self.config
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        if cfg.json_mode:
            payload = {**payload, "response_format": {"type": "json_object"}}
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        resp = requests.post(url, json=payload, headers=headers, timeout=cfg.timeout_s)
        resp.raise_for_status()
        return resp.json()


class ScriptedTeacher:
    """Offline teacher stub: canned structured JSON per action, zero network. For tests and
    build-ahead dry runs until the endpoint lane is greenlit."""

    def __init__(self, scripted: Optional[Dict[str, str]] = None,
                 reference_answers: Optional[Dict[str, str]] = None):
        self.scripted = scripted or {}                # action -> raw JSON-ish content
        self.reference_answers = reference_answers or {}  # question -> short answer
        self.calls: List[str] = []

    def teach(self, lesson_text: str) -> TeacherTurn:
        self.calls.append(f"teach:{lesson_text}")
        return self._turn("teach", lesson_text)

    def answer(self, question: str) -> TeacherTurn:
        self.calls.append(f"answer:{question}")
        if question in self.reference_answers:
            content = json.dumps({"action": "answer", "question": question,
                                  "answer": self.reference_answers[question]})
            return TeacherTurn(action="answer", content=content,
                               parsed=json.loads(content), structured=True)
        return self._turn("answer", question)

    def correct(self, mistake: str, correction: str) -> TeacherTurn:
        self.calls.append(f"correct:{correction}")
        return self._turn("correct", f"{mistake}||{correction}")

    def _turn(self, action: str, key: str) -> TeacherTurn:
        content = self.scripted.get(action, "")
        parsed = parse_structured(content)
        return TeacherTurn(action=action, content=content, parsed=parsed,
                           structured=parsed is not None, meta={"scripted_key": key})
