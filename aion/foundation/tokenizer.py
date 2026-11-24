"""DEPRECATED 2026-11-24 (owner pivot): BPE-32k tokenizer retired with the from-scratch trainer — Qwen3.5-4B-Base brings its own tokenizer; kept for reference.

BPE tokenizer for AION-Seed-Base: HF `tokenizers`, byte-level BPE, 32k vocab.

Trained on the TinyStories corpus slice. Special token <|endoftext|> (id 0) marks
story/document boundaries during pre-tokenization.
"""

import argparse
import json
import os
import sys

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

EOT = "<|endoftext|>"
VOCAB_SIZE = 32000


def train_tokenizer(corpus_files, out_path, vocab_size=VOCAB_SIZE):
    """Train byte-level BPE to `vocab_size` (incl. specials) and save tokenizer.json."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],  # gets the first ids; EOT -> 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train(files=corpus_files, trainer=trainer)
    tokenizer.save(out_path)
    return tokenizer


def load_tokenizer(path):
    tok = Tokenizer.from_file(path)
    return tok


def eot_id(tok) -> int:
    return tok.token_to_id(EOT)


def main():
    ap = argparse.ArgumentParser(description="Train the AION BPE tokenizer")
    ap.add_argument("--corpus", nargs="+", required=True, help="corpus .txt file(s)")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = ap.parse_args()

    for f in args.corpus:
        if not os.path.isfile(f):
            print(f"missing corpus file: {f}", file=sys.stderr)
            sys.exit(2)

    tok = train_tokenizer(args.corpus, args.out, args.vocab_size)
    n = tok.get_vocab_size(with_added_tokens=True)
    print(f"saved {args.out}  vocab={n}  EOT id={tok.token_to_id(EOT)}")
    # sanity round-trip
    s = "Once upon a time there was a little girl named Lucy."
    enc = tok.encode(s)
    assert tok.decode(enc.ids) == s, "round-trip failed"
    print("round-trip OK:", enc.ids[:12], "...")


if __name__ == "__main__":
    main()
