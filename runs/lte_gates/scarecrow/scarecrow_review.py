#!/usr/bin/env python3
"""scarecrow_review.py — send a document to the ScareCrow nvapi lane for adversarial review.
Usage: python scarecrow_review.py <doc-path> <out-path> [review-focus]
Lane law: ScareCrow trio only (http://192.168.0.33:8486/v1, key evo-local, lane nvapi pinned).
Kwargs law: nemotron is a thinking model — enable_thinking:false travels via chat_template_kwargs.
"""
import json
import sys
import urllib.request

URL = "http://192.168.0.33:8486/v1/chat/completions"

doc_path, out_path = sys.argv[1], sys.argv[2]
focus = sys.argv[3] if len(sys.argv) > 3 else (
    "overclaims (claims beyond what was measured), internal contradictions, "
    "undefined jargon, citation problems, structure, and any sentence a skeptical "
    "ML reviewer would flag. Also verify numbers stay consistent with the text's own claims.")
lane = sys.argv[4] if len(sys.argv) > 4 else "nvapi"

doc = open(doc_path, encoding="utf-8").read()
prompt = (
    "You are an adversarial ML-paper reviewer for a Kaggle competition paper track.\n"
    "Review the following paper draft. Focus on: " + focus + "\n"
    "Output format:\n"
    "VERDICT: SUBMIT / FIX-THEN-SUBMIT / MAJOR-REWORK (one line)\n"
    "FINDINGS: numbered, each with severity (FATAL/MAJOR/MINOR/POLISH), location, and the exact suggested fix.\n"
    "Be specific and quote the offending text. Do not invent facts about the work.\n\n"
    "=== PAPER DRAFT ===\n" + doc)

body = {
    "model": lane,
    "messages": [{"role": "user", "content": prompt}],
    "stream": False,
    "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": True},  # owner order 11-12: let the reviewer think
}
req = urllib.request.Request(
    URL, data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer evo-local"})
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.loads(r.read().decode())
text = d["choices"][0]["message"].get("content") or ""
usage = d.get("usage", {})
out = "SCARECROW NVAPI REVIEW — %s\n(lane nvapi, temp 0.2, thinking off, usage: %s)\n\n%s" % (
    doc_path, json.dumps(usage), text)
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(out)
print("REVIEW-WRITTEN", out_path, "| review chars:", len(text))
