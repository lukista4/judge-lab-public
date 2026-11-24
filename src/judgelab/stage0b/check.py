#!/usr/bin/env python3
"""judgelab.stage0b.check: Stage 0b check of the "one bf16 rounding flip" explanation for a cross-engine log-ratio shift.

Situation: on a few pinned calls the two readers' top lists differ by exactly one bf16 step on two candidate logits (log-ratio shift
+2 steps). Hypothesis: both engines compute nearly the same fp32 logit, the values sit on opposite sides of a bf16 rounding midpoint,
and the native bf16 output cast turns a sub-ulp difference into a full step. This checker tests that hypothesis on captured values.

Inputs: two capture JSON files (one per engine: HFJ2 = HF bf16 reader, HFJ3 = vLLM reader), written by judgelab.stage0b.capture, plus the
pin file. PASS iff, for EVERY pinned call:
  (C1) per logit, the HFJ2 and HFJ3 pre-cast fp32 values differ by < 1 bf16 step (step = the bf16 spacing at that value);
  (C2) per logit, the two pre-cast values straddle a rounding midpoint, in OPPOSITE directions for the two logits: the 'up' token's logit
       casts one step higher in HFJ3 than in HFJ2, the 'down' token's one step lower (the direction of the observed shift);
  (C3) one common bf16 cast (round-to-nearest-even, applied here to both engines' captured fp32 values) reproduces the observed shift
       exactly: [cast(up3) - cast(down3)] - [cast(up2) - cast(down2)] == SHIFT, and equals each engine's own native post-cast bf16
       values (otherwise the discrepancy lies in the cast or downstream of it).
Otherwise FAIL (rc 1). Malformed or unpinned input -> REFUSED (rc 4, no verdict). The report binds the sha256 of both inputs and this file.

Pin file (JSON): {"<call_id>": {"position": "SEV"|<severity>, "token_up": "<id>", "token_down": "<id>"}, ...}
Capture format (per engine file):
  {"engine": "HFJ2"|"HFJ3", "gpu_name": str, "calls": [{"call_id": str, "position": str, "token_up": str, "token_down": str,
   "pre_fp32_bits": {"up": "0x########", "down": "0x########"}, "post_bf16_bits": {"up": "0x####", "down": "0x####"}}, ...]}
Usage: python -m judgelab.stage0b.check HFJ2_CAPTURE.json HFJ3_CAPTURE.json REPORT.json --pin PIN.json [--shift 0.25]
"""
import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path

SHIFT = 0.25   # default observed log-ratio shift (HFJ3 B1 minus HFJ2 ref) of token_up vs token_down


class Refused(Exception):
    pass


def f32(bits):
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def bf16_rne(bits32):
    """fp32 bits -> bf16 bits, round to nearest, ties to even (finite inputs only)."""
    lsb = (bits32 >> 16) & 1
    return ((bits32 + 0x7FFF + lsb) >> 16) & 0xFFFF


def bf16_val(b16):
    return f32(b16 << 16)


def bf16_step(b16):
    """spacing between the bf16 value b16 and its upward neighbour in magnitude (same binade spacing)."""
    e = (b16 >> 7) & 0xFF
    return math.ldexp(1.0, (e if e else 1) - 127 - 7)


def hexbits(s, n, what):
    if not isinstance(s, str) or not re.fullmatch(r"0x[0-9a-fA-F]{%d}" % n, s):
        raise Refused("%s: want 0x + %d hex digits, got %r" % (what, n, s))
    v = int(s, 16)
    exp_mask = 0x7F800000 if n == 8 else 0x7F80
    if v & exp_mask == exp_mask:
        raise Refused("%s: inf/nan" % what)
    return v


def load_pin(path):
    try:
        pin = json.loads(Path(path).read_text("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise Refused("pin %s: unreadable JSON (%s)" % (path, e))
    if not isinstance(pin, dict) or not pin:
        raise Refused("pin: want a non-empty object")
    for cid, p in pin.items():
        if not isinstance(p, dict) or any(not isinstance(p.get(k), str) for k in ("position", "token_up", "token_down")):
            raise Refused("pin %s: want string position / token_up / token_down" % cid)
    return pin


def load(path, engine, pin):
    try:
        d = json.loads(Path(path).read_text("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise Refused("%s: unreadable JSON (%s)" % (path, e))
    if d.get("engine") != engine:
        raise Refused("%s: engine %r, want %r" % (path, d.get("engine"), engine))
    calls = d.get("calls")
    if not isinstance(calls, list) or sorted(c.get("call_id") for c in calls) != sorted(pin):
        raise Refused("%s: calls must be exactly the pinned calls" % path)
    out = {}
    for c in calls:
        p = pin[c["call_id"]]
        for k in ("position", "token_up", "token_down"):
            if str(c.get(k)) != p[k]:
                raise Refused("%s %s: %s %r, pinned %r" % (engine, c["call_id"], k, c.get(k), p[k]))
        pre = {t: hexbits((c.get("pre_fp32_bits") or {}).get(t), 8, "%s pre %s" % (engine, t)) for t in ("up", "down")}
        post = {t: hexbits((c.get("post_bf16_bits") or {}).get(t), 4, "%s post %s" % (engine, t)) for t in ("up", "down")}
        out[c["call_id"]] = (pre, post)
    return d.get("gpu_name"), out


def check_call(c2, c3, shift=SHIFT):
    (pre2, post2), (pre3, post3) = c2, c3
    det, ok = {}, {}
    cast2 = {t: bf16_rne(pre2[t]) for t in pre2}
    cast3 = {t: bf16_rne(pre3[t]) for t in pre3}
    c1 = True
    for t in ("up", "down"):
        step = min(bf16_step(cast2[t]), bf16_step(cast3[t]))
        diff = abs(f32(pre3[t]) - f32(pre2[t]))
        det["pre_diff_%s" % t], det["step_%s" % t] = diff, step
        c1 = c1 and diff < step
    ok["C1_lt_one_step"] = c1
    up_move = bf16_val(cast3["up"]) - bf16_val(cast2["up"])
    dn_move = bf16_val(cast3["down"]) - bf16_val(cast2["down"])
    det["cast_move_up"], det["cast_move_down"] = up_move, dn_move

    def straddles(t):
        a, b = sorted((bf16_val(cast2[t]), bf16_val(cast3[t])))
        if a == b:
            return False
        mid = (a + b) / 2.0
        lo, hi = sorted((f32(pre2[t]), f32(pre3[t])))
        return lo <= mid <= hi

    ok["C2_opposite_straddle"] = up_move > 0 and dn_move < 0 and straddles("up") and straddles("down")
    sh = (bf16_val(cast3["up"]) - bf16_val(cast3["down"])) - (bf16_val(cast2["up"]) - bf16_val(cast2["down"]))
    det["common_cast_shift"] = sh
    det["native_equals_common_cast"] = cast2 == post2 and cast3 == post3
    ok["C3_common_cast_reproduces"] = sh == shift and det["native_equals_common_cast"]
    det["pre_fp32"] = {"HFJ2": {t: f32(pre2[t]) for t in pre2}, "HFJ3": {t: f32(pre3[t]) for t in pre3}}
    return all(ok.values()), ok, det


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 0b bf16-flip mechanism check")
    ap.add_argument("hfj2_capture")
    ap.add_argument("hfj3_capture")
    ap.add_argument("report")
    ap.add_argument("--pin", required=True, help="JSON {call_id: {position, token_up, token_down}} fixed before the capture exists")
    ap.add_argument("--shift", type=float, default=SHIFT, help="the observed log-ratio shift to reproduce (default 0.25 = two bf16 steps at [16, 32))")
    a = ap.parse_args(argv)
    try:
        pin = load_pin(a.pin)
        g2, d2 = load(a.hfj2_capture, "HFJ2", pin)
        g3, d3 = load(a.hfj3_capture, "HFJ3", pin)
        if not g2 or g2 != g3:
            raise Refused("gpu_name must be the same card class for both engines: %r vs %r" % (g2, g3))
    except Refused as e:
        print("REFUSED", e)
        return 4
    calls = {}
    for c in sorted(pin):
        p, ok, det = check_call(d2[c], d3[c], a.shift)
        calls[c] = dict(passed=p, checks=ok, detail=det)
    verdict = "PASS" if all(v["passed"] for v in calls.values()) else "FAIL"
    report = dict(verdict=verdict, rule="STAGE 0b (C1 < 1 step per logit, C2 opposite midpoint straddle, C3 common cast reproduces the shift "
                  "and the native casts)", shift=a.shift, gpu_name=g2, calls=calls,
                  inputs_sha256={"hfj2_capture": sha(a.hfj2_capture), "hfj3_capture": sha(a.hfj3_capture), "pin": sha(a.pin), "checker": sha(__file__)})
    Path(a.report).write_text(json.dumps(report, indent=1), "utf-8")
    print(verdict, json.dumps({c: v["checks"] for c, v in calls.items()}))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
