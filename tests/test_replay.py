"""replay_full over VLLMProxy + FakeEngine: schema, E arithmetic, pruning, call selection, pooled loading, negatives."""
import hashlib
import json
import math

import pytest

import fakes as fx
from judgelab import engine as E
from judgelab import replay as RP
from judgelab.proxy import VLLMProxy

pytest.importorskip("torch")
KEYS = {"call_id", "pid", "arm", "reader", "cfg", "prompt_tokens", "E", "p_sev", "sev_mass", "p_digits_by_sev", "digit_mass_by_sev", "tops"}
STORED = {"call_id": "c-audit", "pid": "P1", "probe_ids": fx.plain("Judge this proof. I check each step.") + [fx.CLOSE],
          "first_error_raw": "Step 2 is not justified"}


def proxy(script):
    b = E.Broker(E.FakeEngine(script, ctx=fx.PROTO.CTX), 1 << 22, fake=True, tok=fx.MiniTok(), eos_ids=fx.EOS_IDS, ctx=fx.PROTO.CTX)
    return VLLMProxy(fx.MiniTok(), b, None, fx.EOS_IDS, proto=fx.PROTO)


def row(script=None):
    return RP.replay_full(proxy(script or fx.make_script()), STORED, fx.VARIANTS, arm="BASE", reader="HFJ3", cfg="B1", proto=fx.PROTO)


def test_schema_and_expected_score():
    r = row()
    p = fx.PROTO
    assert set(r) == KEYS and all(isinstance(k, str) for t in r["tops"].values() for k in t) and len(r["tops"]["SEV"]) == 100
    unpruned = {w for w, q in r["p_sev"].items() if q >= p.SEV_PRUNE}
    assert set(r["tops"]) == {"SEV"} | unpruned
    prefix = STORED["probe_ids"] + fx.plain(p.AJ_FE) + fx.plain(STORED["first_error_raw"]) + fx.plain(p.AJ_SEV)
    assert r["prompt_tokens"] == len(prefix)
    num = den = 0.0   # independent recomputation of E from the kept top lists
    for w in unpruned:
        t = {int(k): v for k, v in r["tops"][w].items()}
        dig = {d: math.exp(t[ids[0]]) for d, ids in fx.VARIANTS["AUDIT"].items() if ids[0] in t}
        z = sum(dig.values())
        num += r["p_sev"][w] * sum(int(d) * q for d, q in dig.items()) / z
        den += r["p_sev"][w]
    assert abs(r["E"] - num / den) < 1e-9 and abs(sum(r["p_sev"].values()) - 1) < 1e-9


def test_deterministic_and_corruption_visible():
    base = fx.make_script()

    def corrupt(prompt_ids, req, lora=None):
        toks, rows_ = base(prompt_ids, req, lora)
        if rows_:
            r0 = dict(rows_[0])
            k = fx.VARIANTS["SEV"]["MINOR"][0]
            r0[k] = r0[k] - 0.5   # one severity answer token shifted
            return toks, [r0]
        return toks, rows_
    a, b, c = row(), row(), row(corrupt)
    assert a == b
    assert a["tops"] != c["tops"] and (a["E"] != c["E"] or a["p_sev"] != c["p_sev"])


def test_select_calls_stratified():
    rows = [{"call_id": "c%03d" % i, "kind": "AUDIT", "status": "ok", "probe_ids": [1], "first_error_raw": "x",
             "p2_tokens": [8000, 25000, 40000, 17000][i % 4]} for i in range(80)]
    rows += [{"call_id": "bad1", "kind": "CHECK", "status": "ok", "probe_ids": [1], "first_error_raw": "x", "p2_tokens": 8000},
             {"call_id": "bad2", "kind": "AUDIT", "status": "infra", "probe_ids": [1], "first_error_raw": "x", "p2_tokens": 8000},
             {"call_id": "bad3", "kind": "AUDIT", "status": "ok", "first_error_raw": "x", "p2_tokens": 8000}]
    pick = RP.select_calls(rows)
    bands = [sum(lo <= r["p2_tokens"] < hi for r in pick) for lo, hi, _ in RP.BANDS]
    assert bands == [12, 6, 6]
    assert [r["call_id"] for r in RP.select_calls(list(reversed(rows)))] == [r["call_id"] for r in pick]
    assert not {"bad1", "bad2", "bad3"} & {r["call_id"] for r in pick}


def test_load_pool(tmp_path):
    s1, s2, held = tmp_path / "A_s00.jsonl", tmp_path / "A_s01.jsonl", tmp_path / "final_s02.jsonl"
    s1.write_text(json.dumps({"score": {"call_id": "x1", "a": "A1", "sample": 0}}) + "\n" + json.dumps({"call_id": "x2", "a": "A2", "sample": 0}) + "\n",
                  encoding="utf-8")
    s2.write_text(json.dumps({"score": {"call_id": "x3", "a": "A1", "sample": 0}}) + "\n" + json.dumps({"score": {"call_id": "x4", "a": "A1", "sample": 1}}) + "\n",
                  encoding="utf-8")
    held.write_text(s1.read_text("utf-8"), encoding="utf-8")
    h = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()   # noqa: E731
    pool = RP.load_pool([(s1, h(s1)), (s2, h(s2))])
    assert [r["call_id"] for r in pool] == ["x1", "x2", "x4"] and pool[0]["_src"] == str(s1)
    for srcs in ([(held, h(held))], [(s1, "0" * 64)]):
        with pytest.raises(SystemExit, match="REFUSED"):
            RP.load_pool(srcs)
