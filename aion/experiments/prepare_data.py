"""DEPRECATED 2026-11-24 (owner pivot): TinyStories corpus + BPE pre-tokenization retired with the from-scratch trainer — kept for reference.

Prepare the AION-Seed-Base corpus: train tokenizer + encode corpus -> train.bin + manifest.

Steps:
  1. train BPE tokenizer (32k) on the corpus slice (skipped if tokenizer.json exists)
  2. encode the corpus to a uint16 memmap (train.bin), EOT id 0 marks story boundaries
     (the raw TinyStories txt already contains <|endoftext|> separators)
  3. write corpus_manifest.json: source, byte size, sha256, exact token count
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from aion.foundation.tokenizer import train_tokenizer, load_tokenizer, eot_id  # noqa: E402

CHUNK_BYTES = 16 * 1024 * 1024  # text read chunk


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Prepare AION corpus + tokenizer")
    ap.add_argument("--corpus", required=True, help="corpus .txt")
    ap.add_argument("--out-dir", required=True, help="dir for tokenizer.json, train.bin, manifest")
    ap.add_argument("--source", default="", help="provenance string for the manifest")
    ap.add_argument("--vocab-size", type=int, default=32000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    bin_path = os.path.join(args.out_dir, "train.bin")
    manifest_path = os.path.join(args.out_dir, "corpus_manifest.json")

    t0 = time.time()
    size = os.path.getsize(args.corpus)

    if not os.path.isfile(tok_path):
        print(f"training tokenizer ({args.vocab_size} vocab) on {args.corpus} ...")
        train_tokenizer([args.corpus], tok_path, args.vocab_size)
    tok = load_tokenizer(tok_path)
    eid = eot_id(tok)
    print(f"tokenizer ready: vocab={tok.get_vocab_size(True)} EOT id={eid} "
          f"({time.time()-t0:.0f}s)")

    # encode corpus -> train.bin (uint16), streaming in chunks with overlap
    tenc = time.time()
    n_tokens = 0
    with open(args.corpus, "r", encoding="utf-8", errors="replace") as f, open(bin_path, "wb") as out:
        tail = ""
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            text = tail + chunk
            # keep a small tail so we never split the literal <|endoftext|> marker
            cut = text.rfind("\n")
            if cut <= 0:
                tail = text
                continue
            tail = text[cut + 1 :]
            body = text[: cut + 1]
            ids = tok.encode(body).ids
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(out)
            n_tokens += len(ids)
            print(f"  encoded {n_tokens:,} tokens ({n_tokens*2/2**20:.0f} MiB)", flush=True)
        if tail.strip():
            ids = tok.encode(tail).ids
            np.asarray(ids, dtype=np.uint16).tofile(out)
            n_tokens += len(ids)

    est_chars_per_token = size / max(1, n_tokens)
    manifest = {
        "corpus": os.path.abspath(args.corpus),
        "source": args.source or "local",
        "size_bytes": size,
        "size_mib": round(size / 2**20, 1),
        "sha256": sha256_file(args.corpus),
        "n_tokens": n_tokens,
        "chars_per_token": round(est_chars_per_token, 2),
        "tokenizer": {
            "path": os.path.abspath(tok_path),
            "vocab_size": tok.get_vocab_size(True),
            "eot_id": eid,
            "type": "byte-level BPE (HF tokenizers)",
        },
        "bin": {"path": os.path.abspath(bin_path), "dtype": "uint16",
                "size_mib": round(n_tokens * 2 / 2**20, 1)},
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"MANIFEST: {n_tokens:,} tokens from {size:,} bytes "
          f"({est_chars_per_token:.2f} chars/token) in {time.time()-tenc:.0f}s")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
