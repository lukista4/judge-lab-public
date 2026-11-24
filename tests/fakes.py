"""Synthetic fixtures: a tiny word-or-char tokenizer, a small-context protocol, FakeEngine scripts. No real data."""
import math
import random
import re
from types import SimpleNamespace

from judgelab import protocol

CLOSE = 900
SPECIAL = {"<|endoftext|>": 901, "<|im_start|>": 902, "<|im_end|>": 903, "</think>": CLOSE}
EOS = SPECIAL["<|im_end|>"]
EOS_IDS = {EOS, SPECIAL["<|endoftext|>"]}
WORDS = ["NONE", "MINOR", "MAJOR", "FATAL"]
VOCAB = {ch: i for i, ch in enumerate([chr(c) for c in range(32, 127)] + ["\n", "\t", "[UNK]"] + WORDS)}
INV = {i: t for t, i in list(VOCAB.items()) + list(SPECIAL.items())}
PIECE = re.compile("|".join(re.escape(w) for w in sorted(WORDS, key=len, reverse=True)) + r"|[\s\S]")


class MiniTok:
    """tokenizers.Tokenizer-like surface: encode(text).ids, decode(ids), encode_special_tokens."""

    def __init__(self):
        self.encode_special_tokens = False

    def encode(self, text, add_special_tokens=False):
        split = "(" + "|".join(map(re.escape, SPECIAL)) + ")"
        parts = [text] if self.encode_special_tokens else re.split(split, text)
        ids = []
        for p in parts:
            if p in SPECIAL and not self.encode_special_tokens:
                ids.append(SPECIAL[p])
            else:
                ids += [VOCAB.get(m, VOCAB["[UNK]"]) for m in PIECE.findall(p)]
        return SimpleNamespace(ids=ids)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(INV[i] for i in ids if i in INV)


def plain(text):
    return [VOCAB[m] for m in PIECE.findall(text)]


class SmallProtocol(protocol.ExampleProtocol):
    CTX = 4096
    CLOSE_ID_PIN = CLOSE


PROTO = SmallProtocol()
VARIANTS = {"SEV": {w: [VOCAB[w]] for w in WORDS}, "AUDIT": {str(d): [VOCAB[str(d)]] for d in range(8)}}
ANSWERS = sorted({i for ids in VARIANTS["SEV"].values() for i in ids} | {i for ids in VARIANTS["AUDIT"].values() for i in ids})

THINKS = {  # think scenario -> ids the fake model emits (the engine stops at the first stop id)
    "natural": plain("I check each step.\nThe argument is complete.\n") + [CLOSE] + plain("\n\nverdict"),
    "literal": plain("I check each step.\n</think>\n\nThis text comes after a literal close.") + [EOS],
    "eos": plain("I check each step. It looks fine. Then") + [EOS] + plain("junk after eos"),
    "loop": plain("First I restate the claim with care. ") + plain("Wait, let me recheck this. ") * 60 + [EOS],
}


def read_row(prompt):
    """100 distinct finite logprobs (normalised), answer ids heavy; depends on the prompt length so reads differ."""
    ids = ANSWERS + [i for i in range(300) if i not in ANSWERS][:100 - len(ANSWERS)]
    w = [(1 + (i * 7 + len(prompt)) % 13 + i * 1e-3) * (10 if t in ANSWERS else 1) for i, t in enumerate(ids)]
    z = sum(w)
    return {t: math.log(v / z) for t, v in zip(ids, w)}


def make_script(scenario="natural", calls_log=None, limit_extra=50):
    def script(prompt_ids, req, lora=None):
        if calls_log is not None:
            calls_log.append((list(prompt_ids), dict(req), lora))
        if req.get("logprobs"):
            row = read_row(prompt_ids)
            return [max(row, key=row.get)], [row]
        if float(req.get("temperature", 1.0)) == 0.0:
            return plain("none\nmore") + [EOS], None
        if scenario == "limit":
            rng = random.Random(7)   # non-degenerate random letters, longer than the room (no '>', no loop)
            return [VOCAB[rng.choice("abcdefghijklmnopqrstuvwxyz      ")] for _ in range(int(req["max_tokens"]) + limit_extra)], None
        return list(THINKS[scenario]), None
    return script
