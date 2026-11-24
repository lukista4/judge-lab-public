"""Canary / planted-flaw control for LLM reviewers (stdlib only).

Every batch of reviews a lane (one model behind one access route, at one review stage) is asked to do also carries
controls: proofs with a planted fatal flaw ("invalid") and correct proofs ("valid"). The rules below decide whether a
lane's reviews may count at all.

  record per DISTINCT control   asking a lane the same control twice is not new evidence. A planted flaw counts as caught
                                only if it was caught every time it was shown; a valid control counts as accepted only if
                                it was accepted every time.
  batch void                    a lane that misses ANY planted flaw in a batch has every review of that batch voided.
  DISQUALIFIED                  caught < 75% of >= 4 distinct planted flaws: never asked again for that stage.
  LOW-YIELD                     rejects > 50% of >= 3 valid controls: flagged; its rejections still block a row.
  safety (needed to count)      >= MIN_INV distinct planted flaws seen, >= 90% caught over the whole record, at least one
                                planted flaw in this very batch, and the batch is not void.
  strict                        safety AND never missed a planted flaw AND never rejected a valid control (>= MIN_VAL seen).
  flag-aware catch              a planted flaw counts as caught if the reviewer set the fatal-flaw flag OR its score fell
                                at or below the rejection gate.
  second audit                  a new planted control enters the bank only after an open-book audit by another model
                                family; its four closing lines must agree, otherwise the verdict is UNSURE.

Also: `truncated()` builds a mechanical invalid control (the first ~55% of a real proof, cut at a paragraph or sentence
end), and `pick_canaries()` chooses controls per batch: one per six real reviews (at least two when there are four or
more), alternating invalid / valid, least-seen first, plus one truncation control built from a real job of the batch.
"""
import collections
import hashlib
import random
import re

MIN_INV, MIN_VAL = 6, 3


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class ControlRecord:
    """Canary outcomes per (lane, stage). Events: dict(lane, stage, batch, kid, valid: bool, passed: bool)."""

    def __init__(self, events=()):
        self.qual = collections.defaultdict(list)
        self.void = set()
        for e in events:
            self.add(e)

    def add(self, e):
        self.qual[(e["lane"], e["stage"])].append(e)
        if e["valid"] is False and e["passed"] is False:
            self.void.add((e["lane"], e["stage"], e["batch"]))

    def lane_stats(self, lane, stage):
        """(invalid seen, invalid caught, valid seen, valid accepted), counted per distinct control."""
        inv, val = collections.defaultdict(list), collections.defaultdict(list)
        for e in self.qual.get((lane, stage), []):
            (inv if e["valid"] is False else val)[e["kid"]].append(bool(e["passed"]))
        return len(inv), sum(1 for v in inv.values() if all(v)), len(val), sum(1 for v in val.values() if all(v))

    def disqualified(self, lane, stage):
        ni, ci, _, _ = self.lane_stats(lane, stage)
        return ni >= 4 and ci < 0.75 * ni

    def low_yield(self, lane, stage):
        _, _, nv, cv = self.lane_stats(lane, stage)
        return nv >= 3 and (nv - cv) > 0.5 * nv

    def batch_void(self, lane, stage, batch):
        return (lane, stage, batch) in self.void

    def g5(self, lane, stage, batch):
        """(safety, strict) for one review of `lane` at `stage` in `batch`."""
        ni, ci, nv, cv = self.lane_stats(lane, stage)
        inbatch = [e for e in self.qual.get((lane, stage), []) if e["batch"] == batch and e["valid"] is False]
        safety = ni >= MIN_INV and ci >= 0.9 * ni and bool(inbatch) and not self.batch_void(lane, stage, batch)
        strict = safety and ci == ni and nv >= MIN_VAL and cv == nv
        return safety, strict

    def review_counts(self, review):
        """A real review counts only if its lane is not disqualified and has canary safety in the review's batch."""
        lane, stage, batch = review["lane"], review["stage"], review["batch"]
        return (not self.disqualified(lane, stage)) and self.g5(lane, stage, batch)[0]


def flag_aware_caught(fatal_flaw, score, gate):
    """A planted flaw is caught if the fatal-flaw flag is set OR the score is at or below the rejection gate."""
    return bool(fatal_flaw) or (score is not None and score <= gate)


def truncated(proof):
    """Mechanical invalid control: the first ~55% of a proof, cut at a paragraph or sentence end (conclusion missing)."""
    cut = int(len(proof) * 0.55)
    j = max(proof.rfind("\n\n", 0, cut), proof.rfind(". ", 0, cut) + 1)
    return proof[:j if j > len(proof) * 0.3 else cut].rstrip()


def pick_canaries(record, bank, lane, stage, real_jobs, rnd=None, qualify=None):
    """Controls for one lane+stage in one batch. bank: dicts kid, valid, statement, proof. real_jobs: dicts cid, proof.
    qualify=(n_invalid, n_valid) asks for a fixed-size qualification set instead."""
    rnd = rnd or random.Random(1)
    seen = collections.Counter(e["kid"] for e in record.qual.get((lane, stage), []))
    inv = sorted([k for k in bank if not k["valid"]], key=lambda k: (seen[k["kid"]], sha(lane + stage + k["kid"])))
    val = sorted([k for k in bank if k["valid"]], key=lambda k: (seen[k["kid"]], sha(lane + stage + k["kid"])))
    if qualify:
        ni, nv = qualify
    else:
        n_real = len(real_jobs)
        n = 0 if n_real == 0 else max(2 if n_real >= 4 else 1, n_real // 6)
        ni, nv = (n + 1) // 2, n // 2
    jobs = [{"kind": "canary", "kid": k["kid"], "valid": k["valid"], "statement": k["statement"], "proof": k["proof"]}
            for k in inv[:ni] + val[:nv]]
    if real_jobs and not qualify:
        j = rnd.choice(real_jobs)
        if len(j["proof"]) > 600:
            jobs.append({"kind": "canary", "kid": "T" + sha("trunc:" + j["cid"])[:10], "valid": False,
                         "statement": j.get("statement", ""), "proof": truncated(j["proof"])})
    return jobs


AUDIT_FIELDS = ("FLAW_REAL", "REPAIRED_ELSEWHERE", "PROOF_STILL_COMPLETE")


def parse_second_audit(text):
    """Parse the four closing lines of an open-book flaw audit. A REAL-FATAL verdict whose lines disagree becomes UNSURE.
    Returns None when the reply cannot be parsed."""
    m = re.search(r"VERDICT:\s*\**\s*(REAL-FATAL|NOT-FATAL|UNSURE)", text or "")
    tail = (text or "")[-1500:]
    f = {}
    for x in AUDIT_FIELDS:
        mm = re.search(x + r":\s*\**\s*(yes|no)", tail, re.I)
        f[x] = mm.group(1).lower() if mm else None
    if not m or None in f.values():
        return None
    v = m.group(1)
    if v == "REAL-FATAL" and not (f["FLAW_REAL"] == "yes" and f["REPAIRED_ELSEWHERE"] == "no"
                                  and f["PROOF_STILL_COMPLETE"] == "no"):
        v = "UNSURE"
    return dict(f, verdict=v)


# ---------------------------------------------------------------------------------------------------- synthetic demo
def synthetic_bank():
    """A tiny control bank on elementary statements (written for this repository; no external text)."""
    stmt = "Prove that the sum of two odd integers is even."
    good = ("Let a = 2m + 1 and b = 2n + 1 for integers m, n.\n\nThen a + b = 2m + 2n + 2 = 2(m + n + 1).\n\n"
            "Since m + n + 1 is an integer, a + b is even.")
    bad_root = ("Let a = 2m + 1 and b = 2m + 1 for an integer m.\n\nThen a + b = 4m + 2 = 2(2m + 1), which is even.")
    bad_late = ("Let a = 2m + 1 and b = 2n + 1 for integers m, n.\n\nThen a + b = 2m + 2n + 2.\n\n"
                "Since 2m + 2n + 2 is greater than 2, it is even.")
    stmt2 = "Prove that n^2 - n is even for every integer n."
    good2 = "n^2 - n = n(n - 1) is a product of two consecutive integers, one of which is even; so it is even."
    bad2 = "For n = 1, 2, 3 the values are 0, 2, 6, all even. Hence n^2 - n is even for every integer n."
    bank = []
    for i, (s, p, v) in enumerate([(stmt, good, True), (stmt, bad_root, False), (stmt, bad_late, False),
                                   (stmt2, good2, True), (stmt2, bad2, False)]):
        bank.append({"kid": "K%02d" % i, "valid": v, "statement": s, "proof": p})
    return bank


def simulate_lane(bank, catch_rate, accept_rate, batches=4, seed=0, lane="L", stage="blind"):
    """Synthetic canary events for a lane that catches planted flaws with probability catch_rate and accepts valid
    controls with probability accept_rate (one decision per distinct control, repeated consistently)."""
    rng = random.Random(seed)
    decision = {k["kid"]: (rng.random() < (accept_rate if k["valid"] else catch_rate)) for k in bank}
    ev = []
    for b in range(batches):
        for k in bank:
            ev.append({"lane": lane, "stage": stage, "batch": "b%d" % b, "kid": k["kid"], "valid": k["valid"],
                       "passed": decision[k["kid"]]})
    return ev
