"""Tests of judgelab.stage0b.check on synthetic captures (temp dir, checker run as a subprocess).
Positive control: pre-cast values (20.0624, 20.0626) vs (20.0626, 20.0624) -> one step up / one step down -> shift +0.25.
Negatives cover every PASS condition and every REFUSED path. Call ids and token ids are synthetic."""
import json
import struct
import subprocess
import sys

import pytest

from conftest import subprocess_env
from judgelab.stage0b import check as S

PIN = {"call-A:s0": {"position": "SEV", "token_up": "16", "token_down": "4680"},
       "call-B:s0": {"position": "MAJOR", "token_up": "17", "token_down": "21"}}
C1, C2 = sorted(PIN)
GOOD2 = {C1: (20.0624, 20.0626), C2: (20.0624, 20.0626)}   # HFJ2: up -> 20.0, down -> 20.125
GOOD3 = {C1: (20.0626, 20.0624), C2: (20.0626, 20.0624)}   # HFJ3: up -> 20.125, down -> 20.0 => shift +0.25


def b32(x):
    return struct.unpack(">I", struct.pack(">f", x))[0]


def hx32(x):
    return "0x%08x" % b32(x)


def hx16(x):
    return "0x%04x" % S.bf16_rne(b32(x))


def cap(engine, vals, gpu="NVIDIA L40S", post=None, drop=None, token_up=None):
    calls = []
    for c, p in PIN.items():
        if drop == c:
            continue
        up, dn = vals[c]
        e = dict(call_id=c, position=p["position"], token_up=token_up or p["token_up"], token_down=p["token_down"],
                 pre_fp32_bits={"up": hx32(up), "down": hx32(dn)}, post_bf16_bits={"up": hx16(up), "down": hx16(dn)})
        if post and c in post:
            e["post_bf16_bits"] = {"up": post[c][0], "down": post[c][1]}
        calls.append(e)
    return dict(engine=engine, gpu_name=gpu, calls=calls)


def run(tmp_path, c2, c3, raw2=None, pin=PIN):
    (tmp_path / "h2.json").write_text(raw2 if raw2 is not None else json.dumps(c2), "utf-8")
    (tmp_path / "h3.json").write_text(json.dumps(c3), "utf-8")
    (tmp_path / "pin.json").write_text(json.dumps(pin), "utf-8")
    rp = tmp_path / "R.json"
    if rp.exists():
        rp.unlink()
    p = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "judgelab.stage0b.check", str(tmp_path / "h2.json"), str(tmp_path / "h3.json"),
                        str(rp), "--pin", str(tmp_path / "pin.json")], capture_output=True, text=True, env=subprocess_env())
    rep = json.loads(rp.read_text("utf-8")) if rp.exists() else None
    return p.returncode, rep, p.stdout + p.stderr


def test_bf16_helpers():
    assert S.bf16_val(S.bf16_rne(b32(20.0624))) == 20.0 and S.bf16_val(S.bf16_rne(b32(20.0626))) == 20.125
    assert S.bf16_val(S.bf16_rne(b32(20.0625))) == 20.0   # midpoint ties to even
    assert S.bf16_step(S.bf16_rne(b32(20.0))) == 0.125 and S.bf16_step(S.bf16_rne(b32(12.0))) == 0.0625
    assert S.bf16_step(S.bf16_rne(b32(-20.0))) == 0.125


def test_positive_control(tmp_path):
    rc, rep, log = run(tmp_path, cap("HFJ2", GOOD2), cap("HFJ3", GOOD3))
    assert rc == 0 and rep["verdict"] == "PASS", log
    assert len(rep["inputs_sha256"]) == 4 and all(v["detail"]["common_cast_shift"] == 0.25 for v in rep["calls"].values())


def test_negative_logits(tmp_path):
    rc, _, log = run(tmp_path, cap("HFJ2", {C1: (-20.0626, -20.0624), C2: GOOD2[C2]}), cap("HFJ3", {C1: (-20.0624, -20.0626), C2: GOOD3[C2]}))
    assert rc == 0, log


NEG = [
    ("identical pre-cast values (shift 0)", GOOD2, GOOD2, "C3_common_cast_reproduces", {}),
    ("only the up logit crosses", GOOD2, {C1: (20.0626, 20.0626), C2: GOOD3[C2]}, "C2_opposite_straddle", {}),
    ("both logits cross in the same direction", GOOD2, {C1: (20.0626, 20.0628), C2: GOOD3[C2]}, "C2_opposite_straddle", {}),
    ("reversed direction (shift -0.25)", GOOD3, GOOD2, "C2_opposite_straddle", {}),
    ("up differs by > 1 step", {C1: (20.0624, 20.0626), C2: GOOD2[C2]}, {C1: (20.2, 20.0624), C2: GOOD3[C2]}, "C1_lt_one_step", {}),
    ("binade [8,16): two steps = 0.125, not 0.25", {C1: (12.03124, 12.03126), C2: GOOD2[C2]}, {C1: (12.03126, 12.03124), C2: GOOD3[C2]},
     "C3_common_cast_reproduces", {}),
    ("native HFJ3 post-cast differs from the common cast", GOOD2, GOOD3, "C3_common_cast_reproduces", {"post": {C1: ("0x41a2", "0x41a0")}}),
    ("one call passes, the other identical", GOOD2, {C1: GOOD3[C1], C2: GOOD2[C2]}, "C3_common_cast_reproduces", {}),
]


@pytest.mark.parametrize("name,v2,v3,want,k3", NEG, ids=[n[0] for n in NEG])
def test_fail_conditions(tmp_path, name, v2, v3, want, k3):
    rc, rep, _ = run(tmp_path, cap("HFJ2", v2), cap("HFJ3", v3, **k3))
    assert rc == 1 and rep["verdict"] == "FAIL" and any(not v["checks"][want] for v in rep["calls"].values())


def _nan():
    c = cap("HFJ2", GOOD2)
    c["calls"][0]["pre_fp32_bits"]["up"] = "0x7fc00000"
    return c


def _float():
    c = cap("HFJ2", GOOD2)
    c["calls"][0]["pre_fp32_bits"]["up"] = 20.0624
    return c


REF = [
    ("a pinned call missing", lambda: cap("HFJ2", GOOD2, drop=C2), None),
    ("token pair differs from the pin", lambda: cap("HFJ2", GOOD2, token_up="999"), None),
    ("engine label swapped", lambda: cap("HFJ3", GOOD2), None),
    ("NaN pre-cast bits", _nan, None),
    ("pre-cast given as a float, not bits", _float, None),
    ("unreadable JSON", lambda: None, "{not json"),
]


@pytest.mark.parametrize("name,make2,raw2", REF, ids=[r[0] for r in REF])
def test_refusals(tmp_path, name, make2, raw2):
    rc, rep, log = run(tmp_path, make2(), cap("HFJ3", GOOD3), raw2)
    assert rc == 4 and rep is None and "REFUSED" in log


def test_refuses_other_card_and_bad_pin(tmp_path):
    rc, rep, log = run(tmp_path, cap("HFJ2", GOOD2), cap("HFJ3", GOOD3, gpu="NVIDIA A100-SXM4-80GB"))
    assert rc == 4 and rep is None
    rc, rep, log = run(tmp_path, cap("HFJ2", GOOD2), cap("HFJ3", GOOD3), pin={"call-A:s0": {"position": "SEV"}})
    assert rc == 4 and "pin" in log
