"""Tests of judgelab.stage0_gate: synthetic Stage 0 inputs in a temp dir, the gate run as a subprocess."""
import json
import subprocess
import sys

import pytest

from conftest import subprocess_env
from judgelab import stage0_gate as G

GOOD_CACHE = {a: {"cold_b1": 0.02, "warm": 0.99, "cold_max": 0.90, "warm_min": 0.95, "decoys": 4, "decoy_tokens": 8192, "b32_decoy_rounds": [2, 2, 1, 3]}
              for a in ("BASE", "ADAPTER")}


def rows(n=24):
    r2, r3 = [], []
    for i in range(n):
        for arm in ("BASE", "ADAPTER"):
            for rd, cf in (("HFJ2", "ref"), ("HFJ3", "B1"), ("HFJ3", "B32"), ("HFJ3", "warm")):
                r = {"call_id": "c%02d" % i, "pid": "P%d" % (i // 3), "arm": arm, "reader": rd, "cfg": cf, "prompt_tokens": 40000 if i < 8 else 9000,
                     "E": 3.0 + 0.1 * i + ((0.5 + 0.02 * i) if arm == "ADAPTER" else 0.0),
                     "p_sev": {"NONE": 0.1, "MINOR": 0.6, "MAJOR": 0.3}, "sev_mass": 0.95,
                     "p_digits_by_sev": {"MINOR": {"3": 0.5, "4": 0.5}, "MAJOR": {"2": 1.0}}, "digit_mass_by_sev": {"MINOR": 0.9, "MAJOR": 0.9},
                     "tops": {"SEV": {"11": -0.5, "12": -1.0}, "MINOR": {"3": -0.69}}}
                (r2 if rd == "HFJ2" else r3).append(r)
    return r2, r3


def setup(td, cache=GOOD_CACHE, gpu3="NVIDIA L40S"):
    r2, r3 = rows()
    calls = "".join(json.dumps({"call_id": "c%02d" % i}) + "\n" for i in range(24))
    for name, rr in (("STAGE0_ROWS_HFJ2", r2), ("STAGE0_ROWS_HFJ3", r3)):
        (td / (name + ".jsonl")).write_text("".join(json.dumps(x) + "\n" for x in rr), "utf-8")
        (td / (name + "_CALLS.jsonl")).write_text(calls, "utf-8")
    if cache is not None:
        (td / "STAGE0_ROWS_HFJ3_CACHE.json").write_text(json.dumps(cache), "utf-8")
    for mode, gpu in (("HFJ2", "NVIDIA L40S"), ("HFJ3", gpu3)):
        (td / ("%s.out" % mode)).write_text("== x job=1 mode=%s gpu=%s out=/o\n...\n== x END rc=0\n" % (mode, gpu), "utf-8")


def run(td, extra=(), rep="R.json"):
    p = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "judgelab.stage0_gate", str(td / "STAGE0_ROWS_HFJ2.jsonl"),
                        str(td / "STAGE0_ROWS_HFJ3.jsonl"), str(td / "HFJ2.out"), str(td / "HFJ3.out"), str(td / rep)] + list(extra),
                       capture_output=True, text=True, env=subprocess_env())
    out = json.loads((td / rep).read_text("utf-8")) if (td / rep).exists() else None
    return p.returncode, out, p.stdout + p.stderr


def case(tmp_path, extra=(), **kw):
    setup(tmp_path, **kw)
    return run(tmp_path, extra)


def test_clean_pass_and_report(tmp_path):
    rc, out, log = case(tmp_path)
    assert rc == 0 and out["verdict"] == "PASS", log
    assert len(out["inputs_sha256"]) == 9 and all(v and len(v["lf"]) == 64 for v in out["inputs_sha256"].values())
    assert out["cache_bounds"]["warm_min"] == 0.95


def test_gate_code_pin(tmp_path):
    sha = G.lfsha(G.GATE_CODE.read_bytes())
    rc, out, _ = case(tmp_path, extra=["--expect-gate-sha", sha[:12]])
    assert rc == 0
    rc, out, log = run(tmp_path, ["--expect-gate-sha", "0" * 12], rep="R2.json")
    assert rc != 0 and "REFUSED gate code" in log and out is None


def test_lf_normalised_sha():
    b = b"a = 1\nb = 2\n"
    assert G.lfsha(b) == G.lfsha(b.replace(b"\n", b"\r\n")) and G.lfsha(b) != G.lfsha(b + b"# x\n")


BAD = [("cold_b1 0.99", {"cold_b1": 0.99}, "cold_b1"), ("warm 0.0", {"warm": 0.0}, "warm"),
       ("decoys 3", {"decoys": 3, "b32_decoy_rounds": [1, 1, 1]}, "decoys"), ("decoy_tokens 4096", {"decoy_tokens": 4096}, "decoy_tokens"),
       ("a decoy with 0 rounds", {"b32_decoy_rounds": [1, 0, 2, 1]}, "b32_decoy_rounds"),
       ("rounds list too short", {"b32_decoy_rounds": [1, 1]}, "b32_decoy_rounds"), ("warm as bool", {"warm": True}, "warm"),
       ("cold_b1 missing", {"cold_b1": None}, "cold_b1"), ("cold_max in file raised to 1.0 (ignored)", {"cold_b1": 0.95, "cold_max": 1.0}, "cold_b1")]


@pytest.mark.parametrize("name,change,want", BAD, ids=[b[0] for b in BAD])
def test_cache_bounds(tmp_path, name, change, want):
    c = json.loads(json.dumps(GOOD_CACHE))
    c["ADAPTER"].update(change)
    rc, out, _ = case(tmp_path, cache=c)
    assert rc == 1 and out["verdict"] == "FAIL" and any(want in w for w in out["pre_checks_failed"])


def test_cache_missing_entries(tmp_path):
    c = json.loads(json.dumps(GOOD_CACHE))
    del c["BASE"]
    rc, out, _ = case(tmp_path, cache=c)
    assert rc == 1 and any("BASE" in w for w in out["pre_checks_failed"])


def test_no_cache_file(tmp_path):
    rc, out, _ = case(tmp_path, cache=None)
    assert rc == 1 and any("_CACHE" in w for w in out["pre_checks_failed"])


def test_other_gpu_class(tmp_path):
    rc, out, _ = case(tmp_path, gpu3="NVIDIA L40")
    assert rc == 1 and any("start line" in w for w in out["pre_checks_failed"])
