#!/usr/bin/env python3
"""judgelab.stage0_gate: the post-run Stage 0 verdict in one command (pre-checks + judgelab.equiv.stage0).

usage: python -m judgelab.stage0_gate ROWS_HFJ2 ROWS_HFJ3 OUT_HFJ2 OUT_HFJ3 REPORT.json [--gpu NAME] [--expect-gate-sha PREFIX]
  ROWS_* = STAGE0_ROWS_HFJ2.jsonl / STAGE0_ROWS_HFJ3.jsonl; the *_CALLS.jsonl beside each are found by name; OUT_* = the two batch-job
  logs; the HFJ3 replay's prefix-cache record STAGE0_ROWS_HFJ3_CACHE.json is found by name.
Checks, in order (any failure = verdict FAIL with the reason; never a silent pass):
  (1) both _CALLS files exist and are byte-identical (same calls, same order);
  (2) each log has the start line '== ... mode=<MODE> gpu=<GPU> ...' and an 'END rc=0' line (one GPU class, clean exit);
  (3) row files: HFJ2 rows only reader HFJ2 / cfg ref, HFJ3 rows only reader HFJ3 / cfg B1|B32|warm; row counts = calls x arms x cfgs;
      call ids of every row in the CALLS set;
  (4) _CACHE.json numeric bounds per arm (cold B1 hit share <= 0.90, warm >= 0.95, >= 4 decoy prompts of >= 8192 tokens, every decoy
      replayed in >= 1 B32 round). The bounds are fixed here and never read from the file;
  (5) judgelab.equiv.stage0 on the union of both row files.
The gate code's LF-normalised sha256 is bound into the report; --expect-gate-sha refuses to run if it does not start with PREFIX
(use it to pin the exact gate code that a pre-registration froze).
Writes REPORT.json (aggregates only) and prints the verdict. rc 0 PASS, 1 FAIL."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from . import equiv as eq

GATE_CODE = Path(eq.__file__).resolve()
CACHE_BOUNDS = dict(cold_max=0.90, warm_min=0.95, decoys_min=4, decoy_tokens_min=8192)   # never taken from the file


def lfsha(b):
    return hashlib.sha256(b.replace(bytes([13, 10]), bytes([10]))).hexdigest()


def cache_ok(c):
    """reasons a per-arm _CACHE entry fails the numeric bounds (empty list = ok)."""
    if not isinstance(c, dict):
        return ["entry missing"]
    bad = []
    num = lambda k: isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool)   # noqa: E731
    if not (num("cold_b1") and c["cold_b1"] <= CACHE_BOUNDS["cold_max"]):
        bad.append("cold_b1 %r > %.2f" % (c.get("cold_b1"), CACHE_BOUNDS["cold_max"]))
    if not (num("warm") and c["warm"] >= CACHE_BOUNDS["warm_min"]):
        bad.append("warm %r < %.2f" % (c.get("warm"), CACHE_BOUNDS["warm_min"]))
    if not (num("decoys") and c["decoys"] >= CACHE_BOUNDS["decoys_min"]):
        bad.append("decoys %r < %d" % (c.get("decoys"), CACHE_BOUNDS["decoys_min"]))
    if not (num("decoy_tokens") and c["decoy_tokens"] >= CACHE_BOUNDS["decoy_tokens_min"]):
        bad.append("decoy_tokens %r < %d" % (c.get("decoy_tokens"), CACHE_BOUNDS["decoy_tokens_min"]))
    r = c.get("b32_decoy_rounds")
    if not (isinstance(r, list) and num("decoys") and len(r) == c["decoys"] and all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in r)):
        bad.append("b32_decoy_rounds %r (want one int >= 1 per decoy)" % (r,))
    return bad


def jl(p):
    return [json.loads(x) for x in Path(p).read_text("utf-8").splitlines() if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("rows_hfj2", "rows_hfj3", "out_hfj2", "out_hfj3", "report"):
        ap.add_argument(name, type=Path)
    ap.add_argument("--gpu", default="NVIDIA L40S", help="the one GPU class both logs must name")
    ap.add_argument("--expect-gate-sha", help="refuse unless the LF-normalised sha256 of judgelab/equiv.py starts with this prefix")
    a = ap.parse_args(argv)
    r2p, r3p, o2p, o3p, rep = a.rows_hfj2, a.rows_hfj3, a.out_hfj2, a.out_hfj3, a.report
    if rep.exists():
        raise SystemExit("REFUSED report exists")
    gate_sha = lfsha(GATE_CODE.read_bytes())
    if a.expect_gate_sha and not gate_sha.startswith(a.expect_gate_sha.lower()):
        raise SystemExit("REFUSED gate code changed (LF-normalised sha %s, expected %s)" % (gate_sha[:12], a.expect_gate_sha))
    why = []
    c2, c3 = r2p.with_name(r2p.stem + "_CALLS.jsonl"), r3p.with_name(r3p.stem + "_CALLS.jsonl")
    if not (c2.exists() and c3.exists()):
        why.append("calls file missing")
    elif c2.read_bytes() != c3.read_bytes():
        why.append("CALLS differ between modes")
    for mode, op in (("HFJ2", o2p), ("HFJ3", o3p)):
        t = op.read_text("utf-8", errors="replace") if op.exists() else ""
        st = re.findall(r"^== \S+ job=\S+ mode=(\S+) gpu=(.+?) out=", t, re.M)
        if not st or st[-1][0] != mode or st[-1][1].strip() != a.gpu:
            why.append("%s .out start line: %s (want mode %s gpu %s)" % (mode, st[-1] if st else "none", mode, a.gpu))
        if not re.search(r"^== \S+ END rc=0\s*$", t, re.M):
            why.append("%s .out has no END rc=0" % mode)
    rows2, rows3 = jl(r2p), jl(r3p)
    calls = {json.loads(x)["call_id"] for x in c2.read_text("utf-8").splitlines() if x.strip()} if c2.exists() else set()
    if any((r.get("reader"), r.get("cfg")) != ("HFJ2", "ref") for r in rows2):
        why.append("HFJ2 file has non-ref rows")
    if any(r.get("reader") != "HFJ3" or r.get("cfg") not in ("B1", "B32", "warm") for r in rows3):
        why.append("HFJ3 file has foreign rows")
    want = {"HFJ2": 2 * len(calls), "HFJ3": 6 * len(calls)}
    if (len(rows2), len(rows3)) != (want["HFJ2"], want["HFJ3"]):
        why.append("row counts %d / %d, want %d / %d (calls x arms x cfgs)" % (len(rows2), len(rows3), want["HFJ2"], want["HFJ3"]))
    if any(r.get("call_id") not in calls for r in rows2 + rows3):
        why.append("row call_id outside the CALLS set")
    cj = r3p.with_name(r3p.stem + "_CACHE.json")
    cache = json.loads(cj.read_text("utf-8")) if cj.exists() else None
    if not isinstance(cache, dict):
        why.append("HFJ3 _CACHE.json missing or not an object (replay without the prefix-cache checks)")
    else:
        for arm in ("BASE", "ADAPTER"):
            for b in cache_ok(cache.get(arm)):
                why.append("_CACHE.json %s: %s" % (arm, b))
    try:
        st0 = eq.stage0(rows2 + rows3)
    except SystemExit as e:
        st0 = {"verdict": "FAIL", "gates": [], "refused": str(e)}
        why.append(str(e))
    verdict = "PASS" if not why and st0["verdict"] == "PASS" else "FAIL"
    out = {"verdict": verdict, "pre_checks_failed": why, "stage0": st0, "n_rows": [len(rows2), len(rows3)], "n_calls": len(calls),
           "gate_code": "judgelab/equiv.py %s (LF-normalised)" % gate_sha[:12], "replay_cache": cache, "expected_rows": want, "gpu_required": a.gpu,
           "cache_bounds": CACHE_BOUNDS,
           "inputs_sha256": {n: ({"raw": hashlib.sha256(p.read_bytes()).hexdigest(), "lf": lfsha(p.read_bytes())} if p.exists() else None)
                             for n, p in (("rows_hfj2", r2p), ("rows_hfj3", r3p), ("calls_hfj2", c2), ("calls_hfj3", c3), ("cache_hfj3", cj),
                                          ("out_hfj2", o2p), ("out_hfj3", o3p), ("gate_code", GATE_CODE), ("gate", Path(__file__).resolve()))}}
    rep.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("COMPUTED stage0 gate %s; pre-check failures %d; gates %s" % (verdict, len(why), {g["gate"]: g["pass"] for g in st0.get("gates", [])}))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
