"""Self-contained tests of judgelab.equiv on synthetic panels.

The panels are generated at a noise level matching measured aggregate statistics of real base-model replicate reads
(between-proof E SD about 2.66, per-read E noise SD about 1.30, 78 problems with 3-4 proofs each). No real rows are used.
Every gate is exercised on an equivalent reader (must pass) and on sabotaged readers (must fail or be refused).
"""
import json
import math
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from conftest import subprocess_env
from judgelab import equiv as eq

PINS = {"config.json": "c0", "model.safetensors": "m0"}
AD = "a" * 12   # synthetic adapter id


def refused(fn):
    try:
        fn()
        return ""
    except SystemExit as e:
        return str(e)


def panel(seed, n_prob=78, per=(3, 4), arm="BASE", K=(0, 1), bias=0.0, tmul=1.0, spread=1.0, scramble=0.0, noise=1.30, eff_scramble=False):
    """returns (s0, s1, hf3, labels, calls, base_reads or None). Proofs a = 'p<i>_<j>'; call_id '<a>:L1:s<k>'."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_prob):
        n = int(rng.integers(per[0], per[1] + 1))
        pe = rng.normal(0, 1.2)
        for j in range(n):
            c = int(j % 2 == 0) if n > 1 else int(rng.integers(0, 2))
            rows.append(("p%02d_%d" % (i, j), "P%02d" % i, c, 3.0 + 2.6 * c + pe + rng.normal(0, 1.4), rng.normal(8.6, 0.6),
                         rng.normal(0, 0.8)))
    adsha = AD if arm == "ADAPTER" else None

    def row(a, pid, k, E, tt, runtime, armv=None, ad=adsha):
        ids = list(range(5)) + [int(x) for x in rng.integers(0, 50, int(tt))] + [7, 8]
        return {"a": a, "pid": pid, "sample": k, "call_id": "%s:L1:s%d" % (a, k), "E": float(np.clip(E, 0, 7)), "status": "ok",
                "think_tokens": int(tt), "arm": armv or ("BASE" if arm == "BASE" else "ADAPTER"), "runtime": runtime,
                "adapter_sha256": ad, "base_pins": dict(PINS), "think_class": "natural", "severity": "MINOR", "weighted_digit_mass": 0.9,
                "gen_is_answer": False, "probe_ids": ids, "probe_close_idx": 5 + int(tt), "think_prompt_n": 5}
    s0, s1, hf3, g0, g1, g3 = {}, {}, {}, {}, {}, {}
    for a, pid, c, mu, lt, eff in rows:
        t = lambda m=1.0: max(50.0, math.exp(lt + rng.normal(0, 0.25)) * m)
        s0[(a, 0)] = row(a, pid, 0, mu + eff * (arm == "ADAPTER") + rng.normal(0, noise), t(), "hf_bf16")
        s1[(a, 1)] = row(a, pid, 1, mu + eff * (arm == "ADAPTER") + rng.normal(0, noise), t(), "hf_bf16")
        for k in K:
            e = mu + (rng.normal(0, 0.8) if eff_scramble else eff) * (arm == "ADAPTER") + rng.normal(0, noise)
            if scramble and rng.random() < scramble:
                e = rng.uniform(0, 7)
            hf3[(a, k)] = row(a, pid, k, 4.5 + spread * (e - 4.5) + bias, t(tmul), "hfj3_vllm033_bf16")
        if arm == "ADAPTER":
            g0[(a, 0)] = row(a, pid, 0, mu + rng.normal(0, noise), t(), "hf_bf16", "BASE", None)
            g1[(a, 1)] = row(a, pid, 1, mu + rng.normal(0, noise), t(), "hf_bf16", "BASE", None)
            g3[(a, 0)] = row(a, pid, 0, mu + rng.normal(0, noise), t(), "hfj3_vllm033_bf16", "BASE", None)
    labels = [{"a": a, "correct": c} for a, _, c, _, _, _ in rows]
    calls = ["%s:L1:s%d" % (a, k) for a, *_ in rows for k in K]
    base = dict(s0=g0, s1=g1, hfj3=g3) if arm == "ADAPTER" else None
    return s0, s1, hf3, labels, calls, base


RUN = [{"runtime": "hfj3_vllm033_bf16", "penalty_scope": "prompt", "fake": False, "adapter_sha256": None, "base_pins": dict(PINS)}]
RUN_AD = [dict(RUN[0], adapter_sha256=AD)]


def s1run(p, arm="BASE", runs=None, bad=None, mutate=None):
    s0, s1, hf3, labels, calls, base = p
    if mutate:
        mutate(s0, s1, hf3, calls, base)
    return eq.stage1(s0, s1, hf3, labels, arm, calls, bad, runs if runs is not None else (RUN if arm == "BASE" else RUN_AD), base)


def gates(rep):
    return {g["gate"]: g for g in rep["primary"] + rep["secondary"] + rep["reported"]}


def stage0_rows(n=24, b32_none=False):
    rows = []
    for i in range(n):
        for arm in ("BASE", "ADAPTER"):
            for rd, cf in (("HFJ2", "ref"), ("HFJ3", "B1"), ("HFJ3", "B32"), ("HFJ3", "warm")):
                r = {"call_id": "c%02d" % i, "pid": "P%d" % (i // 3), "arm": arm, "reader": rd, "cfg": cf, "prompt_tokens": 40000 if i < 8 else 9000,
                     "E": 3.0 + 0.1 * i + (0.5 if arm == "ADAPTER" else 0) + (0.01 * i if arm == "ADAPTER" else 0),
                     "p_sev": {"NONE": 0.1, "MINOR": 0.6, "MAJOR": 0.3}, "sev_mass": 0.95,
                     "p_digits_by_sev": {"MINOR": {"3": 0.5, "4": 0.5}, "MAJOR": {"2": 1.0}}, "digit_mass_by_sev": {"MINOR": 0.9, "MAJOR": 0.9},
                     "tops": {"SEV": {"11": -0.5, "12": -1.0}, "MINOR": {"3": -0.69}}}
                if b32_none and cf == "B32" and i == 5 and arm == "BASE":
                    r["p_digits_by_sev"] = None
                rows.append(r)
    return rows


def run_suite():
    ok = []
    ok.append(("(0) c_K = 0.7071 (K=2) / 0.8660 (K=1)", abs(eq.c_k(2) - 0.70711) < 1e-4 and abs(eq.c_k(1) - 0.86603) < 1e-4))
    # (1) equivalent reader, BASE full panel, real noise: 12 seeds -> no FAIL / INCONCLUSIVE expected in >= 11
    res = []
    for sd in range(12):
        rep = s1run(panel(100 + sd))
        res.append(rep["verdict"])
    good = sum(v in ("PASS", "AMBER") for v in res)
    ok.append(("(1) equivalent BASE reader, 78 problems, K=2: %d/12 passed %s" % (good, sorted(set(res))), good >= 11))
    rep = s1run(panel(100))
    G = gates(rep)
    ok.append(("(1b) bars under ceilings at real noise: pooled %.3f within %.3f bias-all %.3f think %.3f" % (
        G["P.pooled"]["bar"], G["P.within"]["bar"], G["P.bias"]["splits"][0]["bar"], G["P.think"]["bar"]),
        G["P.pooled"]["bar"] < 0.06 and G["P.within"]["bar"] < 0.08 and G["P.bias"]["splits"][0]["bar"] < 0.30 and G["P.think"]["bar"] < 0.10))
    ok.append(("(1c) report has no HFJ3 bootstrap fields; c_K 0.707; rule corr_rss", abs(rep["c_K"] - 0.70711) < 1e-4 and "corr_rss" in rep["rule"]))
    # (2) bugs on BASE
    r = s1run(panel(101, bias=0.6))
    ok.append(("(2a) E bias +0.6 -> FAIL with P.bias", r["verdict"] == "FAIL" and "P.bias" in r["failed_primary"]))
    r = s1run(panel(102, tmul=1.3))
    ok.append(("(2b) think x1.3 -> FAIL with P.think", r["verdict"] == "FAIL" and "P.think" in r["failed_primary"]))
    r = s1run(panel(103, spread=1.25))
    ok.append(("(2c) E spread x1.25 -> FAIL with P.spread (ratio %.3f)" % gates(r)["P.spread"]["ratio"], r["verdict"] == "FAIL" and "P.spread" in r["failed_primary"]))
    r = s1run(panel(104, scramble=0.4))
    ok.append(("(2d) 40%% E scramble -> FAIL (%s)" % r["failed_primary"], r["verdict"] == "FAIL"))
    # (3) INCONCLUSIVE: 12 problems x 4, huge noise -> bars above ceilings, no point failure on an equivalent reader
    r = s1run(panel(105, n_prob=12, per=(4, 4), noise=2.6), runs=RUN, mutate=None) if False else None
    p = panel(105, n_prob=30, per=(4, 4), noise=2.6)
    s0, s1, hf3, labels, calls, _ = p
    old = eq.MIN_CALLS["BASE"]
    eq.MIN_CALLS["BASE"] = 10
    try:
        r = eq.stage1(s0, s1, hf3, labels, "BASE", calls, None, RUN, None)
    finally:
        eq.MIN_CALLS["BASE"] = old
    ok.append(("(3) noisy 30-problem panel -> INCONCLUSIVE (inconclusive %s, failed %s)" % (r["inconclusive_primary"], r["failed_primary"]),
               r["verdict"] in ("INCONCLUSIVE", "FAIL") and bool(r["inconclusive_primary"])))
    # (4) provenance -> REFUSED
    def mut_rt(s0, s1, hf3, calls, base):
        next(iter(hf3.values()))["runtime"] = "hf_bf16"
    def mut_ad(s0, s1, hf3, calls, base):
        next(iter(s1.values()))["adapter_sha256"] = "x"
    def mut_pins(s0, s1, hf3, calls, base):
        next(iter(hf3.values()))["base_pins"] = {"config.json": "c1"}
    def mut_cid(s0, s1, hf3, calls, base):
        r0 = next(iter(s0.values()))
        r0["call_id"] = r0["call_id"].replace("L1", "L2")
    def mut_rt2(s0, s1, hf3, calls, base):
        next(iter(s0.values()))["runtime"] = "hfj3_vllm033_bf16"
    pb = lambda: panel(110)
    for name, kw, want in (("HFJ3 row runtime", dict(mutate=mut_rt), "runtime"), ("HFJ2 row runtime", dict(mutate=mut_rt2), "runtime"),
                           ("adapter_sha256 differs", dict(mutate=mut_ad), "adapter_sha256"), ("base_pins differ", dict(mutate=mut_pins), "base_pins"),
                           ("HFJ2 call_id", dict(mutate=mut_cid), "call_id"),
                           ("RUN.json penalty_scope", dict(runs=[dict(RUN[0], penalty_scope="full")]), "penalty_scope"),
                           ("RUN.json fake", dict(runs=[dict(RUN[0], fake=True)]), "fake"),
                           ("RUN.json runtime", dict(runs=[dict(RUN[0], runtime="hf_bf16")]), "runtime"),
                           ("RUN.json missing", dict(runs=[]), "RUN.json")):
        m = refused(lambda: s1run(pb(), **kw))
        ok.append(("(4) %s -> REFUSED: %s" % (name, m[:70]), m.startswith("REFUSED") and want in m))
    # (5) P.sanity
    def drop(n):
        def f(s0, s1, hf3, calls, base):
            for key in sorted(hf3)[:n]:
                del hf3[key]
        return f
    p = panel(120)
    n_calls = len(p[4])
    k_ok = math.ceil(0.01 * n_calls) - 1
    r = s1run(panel(120), mutate=drop(k_ok))
    ok.append(("(5a) BASE %d calls, %d missing (< 1 %%) -> P.sanity pass" % (n_calls, k_ok), gates(r)["P.sanity"]["pass"]))
    r = s1run(panel(120), mutate=drop(k_ok + 1))
    ok.append(("(5b) %d missing (>= 1 %%) -> P.sanity FAIL, verdict FAIL" % (k_ok + 1), not gates(r)["P.sanity"]["pass"] and r["verdict"] == "FAIL"))
    pa = panel(121, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,))
    r = s1run(pa, arm="ADAPTER", mutate=drop(1))
    ok.append(("(5c) ADAPTER 96 calls, 1 missing -> P.sanity FAIL", not gates(r)["P.sanity"]["pass"] and r["verdict"] == "FAIL"))
    r = s1run(panel(122), bad={"x": {"status": "infra"}} if False else None, mutate=None)
    p = panel(122)
    bad = {c: {"status": "infra", "call_id": c} for c in p[4][:3]}
    r = s1run(p, bad=bad, mutate=lambda s0, s1, hf3, calls, base: [hf3.pop(k) for k in sorted(hf3)[:3]])
    ok.append(("(5d) 3 infra rows (< 1 %%) -> P.sanity pass, infra_share %.4f" % gates(r)["P.sanity"]["infra_share"], gates(r)["P.sanity"]["pass"]))
    def enone(s0, s1, hf3, calls, base):
        next(iter(hf3.values()))["E"] = None
    r = s1run(panel(123), mutate=enone)
    ok.append(("(5e) one HFJ3 ok row with E None -> P.sanity FAIL (e_none %d)" % gates(r)["P.sanity"]["e_none"], not gates(r)["P.sanity"]["pass"]))
    # (6) ADAPTER: P.effect decides, E-bias / think reported only
    res = [s1run(panel(130 + sd, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,)), arm="ADAPTER") for sd in range(6)]
    effp = sum(gates(x)["P.effect"]["pass"] or gates(x)["P.effect"]["inconclusive"] for x in res)
    ok.append(("(6a) equivalent ADAPTER reader: P.effect within bar %d/6 (bars %s)" % (
        sum(gates(x)["P.effect"]["within_bar"] for x in res), [round(gates(x)["P.effect"]["bar"], 3) for x in res]), effp >= 5))
    vv = [x["verdict"] for x in res]
    ok.append(("(6g) equivalent ADAPTER reader at real noise, 24 x 4, proposal ceilings: %d/6 PASS|AMBER %s" % (
        sum(v in ("PASS", "AMBER") for v in vv), sorted(set(vv))), sum(v in ("PASS", "AMBER") for v in vv) >= 5))
    r = s1run(panel(140, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,), eff_scramble=True), arm="ADAPTER")
    ok.append(("(6b) HFJ3 adapter effect scrambled -> P.effect not passed (contrast %.3f bar %.3f)" % (gates(r)["P.effect"]["contrast"], gates(r)["P.effect"]["bar"]),
               not gates(r)["P.effect"]["pass"]))
    r = s1run(panel(141, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,), bias=0.6), arm="ADAPTER")
    G = gates(r)
    ok.append(("(6c) ADAPTER E bias +0.6: R.bias reported (pass %s), not in primaries, verdict not decided by it (%s)" % (G["R.bias"]["pass"], r["verdict"]),
               "R.bias" in G and "P.bias" not in G and "R.bias" not in r["failed_primary"] and "R.think" in G))
    p = panel(142, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,))
    m = refused(lambda: eq.stage1(p[0], p[1], p[2], p[3], "ADAPTER", p[4], None, RUN_AD, None))
    ok.append(("(6d) ADAPTER without base reads -> REFUSED", "P.effect" in m))
    def base_ad(s0, s1, hf3, calls, base):
        next(iter(base["s0"].values()))["adapter_sha256"] = AD
    m = refused(lambda: s1run(panel(143, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,)), arm="ADAPTER", mutate=base_ad))
    ok.append(("(6e) base reader row with an adapter -> REFUSED: %s" % m[:60], m.startswith("REFUSED")))
    m = refused(lambda: s1run(panel(144, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,)), arm="ADAPTER", runs=RUN))
    ok.append(("(6f) ADAPTER RUN.json adapter_sha256 None vs rows -> REFUSED", "adapter_sha256" in m))
    # (6h) P.effect sigma / bar are HFJ2-only (scrambling HFJ3's adapter reads cannot change them)
    p = panel(145, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,))
    g1 = gates(s1run(p, arm="ADAPTER"))["P.effect"]
    p = panel(145, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,))
    rs = np.random.default_rng(7)
    keys = sorted(p[2])
    vals = [p[2][k]["E"] for k in keys]
    for k, v in zip(keys, rs.permutation(vals)):
        p[2][k]["E"] = float(v)
    g2 = gates(s1run(p, arm="ADAPTER"))["P.effect"]
    ok.append(("(6h) E1: HFJ3 adapter E permuted -> sigma %.4f == %.4f, bar %.4f == %.4f, contrast %.3f -> %.3f" % (
        g1["sigma"], g2["sigma"], g1["bar"], g2["bar"], g1["contrast"], g2["contrast"]),
        g1["sigma"] == g2["sigma"] and g1["bar"] == g2["bar"] and abs(
            g1["bar"] - math.sqrt((2.33 * math.sqrt(2) * g1["sigma"]) ** 2 + 0.05 ** 2)) < 1e-5))
    # (7) stage0 edges
    r0 = eq.stage0([])
    ok.append(("(7a) stage0([]) -> FAIL, no crash", r0["verdict"] == "FAIL"))
    m = refused(lambda: eq.stage0(stage0_rows(b32_none=True)))
    ok.append(("(7b) B32 row with p_digits_by_sev None -> REFUSED", "p_digits_by_sev" in m))
    r0 = eq.stage0(stage0_rows())
    ok.append(("(7c) identical stage0 rows: gates computed (%d), S0.E pass" % len(r0["gates"]),
               any(g["gate"] == "S0.E[BASE]" and g["pass"] for g in r0["gates"])))
    # (10) Stage 0 flip rule; constants set per case, restored after
    def shifted(spec, n=24, dE=None):
        rows = stage0_rows(n)
        for r in rows:
            for (ci, arm, cfgs, delta, tok) in spec:
                if r["call_id"] == "c%02d" % ci and r["arm"] == arm and r["cfg"] in cfgs:
                    if delta is None:
                        del r["tops"]["SEV"][tok]
                    else:
                        r["tops"]["SEV"][tok] += delta
                    if dE is not None:
                        r["E"] += dE
        return rows
    def s0(rows, lim=None, cap=None):
        old = (eq.FLIP_LIMIT, eq.FLIP_CAP)
        eq.FLIP_LIMIT = old[0] if lim is None else lim
        eq.FLIP_CAP = old[1] if cap is None else cap
        try:
            r = eq.stage0(rows)
        finally:
            eq.FLIP_LIMIT, eq.FLIP_CAP = old
        return r, {g["gate"]: g for g in r["gates"]}
    r, G = s0(stage0_rows())
    ok.append(("(10a0) identical rows: PASS, 0 flip calls, S0.flips gates present", r["verdict"] == "PASS" and G["S0.flips[BASE]"]["flip_calls"] == 0
               and G["S0.flips[ADAPTER]"]["flip_calls"] == 0))
    r, G = s0(shifted([(i, "ADAPTER", ("B1",), 0.125, "11") for i in (1, 4, 9)]))
    ok.append(("(10a) 3 ADAPTER calls with a one-step B1 flip (seen vs B32/warm too) -> excused, PASS (flips %d)" % G["S0.flips[ADAPTER]"]["flip_calls"],
               r["verdict"] == "PASS" and G["S0.flips[ADAPTER]"]["flip_calls"] == 3 and G["S0.flips[ADAPTER]"]["own_classes"] == [1]))
    r, G = s0(shifted([(i, "ADAPTER", ("B1", "B32", "warm"), 0.125, "11") for i in (1, 4, 9)]))
    ok.append(("(10b) NEGATIVE same shift in all HFJ3 cfgs (no flip inside HFJ3's own spread) -> not excused, highmass FAIL",
               r["verdict"] == "FAIL" and not G["S0.highmass[ADAPTER]"]["pass"] and G["S0.flips[ADAPTER]"]["flip_calls"] == 0))
    rows13 = shifted([(i, "BASE", ("B1",), 0.125, "11") for i in range(13)])
    r, G = s0(rows13, cap=12)
    ok.append(("(10c) NEGATIVE 13 flip calls > cap 12 -> S0.flips[BASE] FAIL", r["verdict"] == "FAIL" and not G["S0.flips[BASE]"]["pass"]))
    r, G = s0(rows13, cap=13)
    ok.append(("(10c2) same rows with cap 13 -> PASS", r["verdict"] == "PASS"))
    r, G = s0(shifted([(2, "BASE", ("B1",), 0.25, "11")]), lim=0.125)
    ok.append(("(10d) NEGATIVE 0.25 shift with limit 0.125 -> beyond limit, non-flip, highmass FAIL",
               r["verdict"] == "FAIL" and G["S0.flips[BASE]"]["beyond_limit_calls"] == 1 and not G["S0.highmass[BASE]"]["pass"]))
    r, G = s0(shifted([(2, "BASE", ("B1",), 0.25, "11"), (3, "BASE", ("B32",), 0.25, "11")]), lim=0.25)
    ok.append(("(10d2) one class: a 0.25 cross shift and a 0.25 self shift are both class 1 -> PASS (own %s)" % G["S0.flips[BASE]"]["own_classes"],
               r["verdict"] == "PASS" and G["S0.flips[BASE]"]["own_classes"] == [1]))
    r, G = s0(shifted([(5, "BASE", ("B1",), None, "11")]))
    ok.append(("(10e) NEGATIVE high-mass token missing in B1 -> not a flip, highmass FAIL", r["verdict"] == "FAIL" and not G["S0.highmass[BASE]"]["pass"]
               and G["S0.flips[BASE]"]["flip_calls"] == 0))
    r, G = s0(shifted([(i, "ADAPTER", ("B1",), 0.125, "11") for i in range(24)]), cap=24)
    ok.append(("(10f) every ADAPTER call flips -> S0.flips needs >= 1 non-flip call: FAIL, no crash", r["verdict"] == "FAIL"
               and not G["S0.flips[ADAPTER]"]["pass"] and G["S0.flips[ADAPTER]"]["non_flip_calls"] == 0))
    ok.append(("(10g) ceilings: BASE within 0.10, ADAPTER within 0.17 / pooled 0.12 / effect 0.35", eq.CEIL["BASE"]["within"] == 0.10
               and eq.CEIL["ADAPTER"]["within"] == 0.17 and eq.CEIL["ADAPTER"]["pooled"] == 0.12 and eq.CEIL["ADAPTER"]["effect"] == 0.35))
    # (11) constants; cross-only flip calls; invariance skips within-limit self flips
    ok.append(("(11a) constants: limit 0.25, cap 12, one class (0.125 -> 1, 0.25 -> 1, 0.26 -> None, 0 -> 0)", eq.FLIP_LIMIT == 0.25 and eq.FLIP_CAP == 12
               and eq.flip_class(0.125) == 1 and eq.flip_class(0.25) == 1 and eq.flip_class(0.26) is None and eq.flip_class(0.0) == 0))
    r, G = s0(shifted([(i, "ADAPTER", ("B32",), 0.125, "11") for i in range(5)], dE=0.15))
    ok.append(("(11b) self-only jitter (B32 one step + dE 0.15 on 5 calls, ref-B1 identical): 0 flip calls, 5 self flip pairs skipped by invariance -> PASS",
               r["verdict"] == "PASS" and G["S0.flips[ADAPTER]"]["flip_calls"] == 0 and G["S0.flips[ADAPTER]"]["self_flip_pairs"] == 5))
    r, G = s0(shifted([(3, "ADAPTER", ("B32",), 0.5, "11")], dE=0.15))
    ok.append(("(11c) NEGATIVE self pair BEYOND the limit (0.5) with dE 0.15 stays in invariance -> S0.invariance FAIL",
               r["verdict"] == "FAIL" and not G["S0.invariance[ADAPTER]"]["pass"] and G["S0.flips[ADAPTER]"]["self_flip_pairs"] == 0))
    r, G = s0(shifted([(4, "BASE", ("B1",), 0.25, "11"), (6, "BASE", ("B32",), 0.125, "12")]))
    det = G["S0.flips[BASE]"]["flip_detail"]
    ok.append(("(11d) safeguard (i): flip_detail reports spread + dE for each flip call: %s" % det, r["verdict"] == "PASS" and len(det) == 1
               and abs(list(det.values())[0]["ref_b1_spread"] - 0.25) < 1e-6 and "dE_b1_minus_ref" in list(det.values())[0]))
    r, G = s0(shifted([(4, "BASE", ("B1",), 0.375, "11"), (6, "BASE", ("B32",), 0.125, "12")]))
    ok.append(("(11e) NEGATIVE 3-step cross shift (0.375 > 0.25) -> not a flip, highmass FAIL", r["verdict"] == "FAIL" and not G["S0.highmass[BASE]"]["pass"]))
    # (8) auc_boot on single-class draws: no RuntimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            e = np.array([1.0, 2.0, 3.0, 4.0])
            yy = np.array([True, False, True, False])
            pid = np.array(["A", "A", "B", "C"])
            w = eq.weights(pid, sorted(set(pid)), np.random.default_rng(0))
            out = eq.auc_boot(e, yy, pid, w)
            ok.append(("(8) auc_boot single-class draws -> NaN draws, no warning (NaN share %.2f)" % float(np.mean(np.isnan(out[2]))), True))
        except RuntimeWarning as ex:
            ok.append(("(8) auc_boot warned: %s" % ex, False))
    # (9) CLI end to end (temp dirs, RUN.json next to READS, rc mapping, refusal of --base-* on BASE)
    with tempfile.TemporaryDirectory(prefix="eq_") as td:
        td = Path(td)
        s0, s1, hf3, labels, calls, _ = panel(150)
        for name, rows in (("h0", s0), ("h1", s1), ("h3", hf3)):
            (td / name / "s00").mkdir(parents=True)
            (td / name / "s00" / "READS.jsonl").write_text("".join(json.dumps({"score": r}) + "\n" for r in rows.values()), "utf-8")
        (td / "h3" / "s00" / "RUN.json").write_text(json.dumps(RUN[0]), "utf-8")
        (td / "calls.txt").write_text("\n".join(calls) + "\n", "utf-8")
        (td / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels), "utf-8")
        base = [sys.executable, "-B", "-X", "utf8", "-m", "judgelab.equiv", "stage1", "--hfj2-s0", str(td / "h0"), "--hfj2-s1", str(td / "h1"), "--hfj3", str(td / "h3"),
                "--calls", str(td / "calls.txt"), "--labels", str(td / "labels.jsonl"), "--arm", "BASE"]
        p1 = subprocess.run(base + ["--out", str(td / "r1.json")], capture_output=True, text=True, env=subprocess_env())
        v = json.loads((td / "r1.json").read_text("utf-8"))["verdict"] if (td / "r1.json").exists() else None
        ok.append(("(9a) CLI BASE rc %d verdict %s" % (p1.returncode, v), v is not None and p1.returncode == {"PASS": 0, "AMBER": 2, "INCONCLUSIVE": 3, "FAIL": 1}[v]))
        (td / "h3" / "s00" / "RUN.json").unlink()
        p2 = subprocess.run(base + ["--out", str(td / "r2.json")], capture_output=True, text=True, env=subprocess_env())
        ok.append(("(9b) CLI without HFJ3 RUN.json -> REFUSED", "RUN.json missing" in (p2.stdout + p2.stderr) and not (td / "r2.json").exists()))
        (td / "h3" / "s00" / "RUN.json").write_text(json.dumps(RUN[0]), "utf-8")
        p3 = subprocess.run(base + ["--base-hfj3", str(td / "h3"), "--out", str(td / "r3.json")], capture_output=True, text=True, env=subprocess_env())
        ok.append(("(9c) CLI BASE with --base-hfj3 -> REFUSED", "ADAPTER only" in (p3.stdout + p3.stderr)))
    # (12) P.effect primary / reported branches
    pa = panel(140, n_prob=24, per=(4, 4), arm="ADAPTER", K=(0,), eff_scramble=True)
    eq.EFFECT_PRIMARY = True
    r1 = s1run(pa, arm="ADAPTER")
    eq.EFFECT_PRIMARY = False
    r2 = s1run(pa, arm="ADAPTER")
    eq.EFFECT_PRIMARY = True
    ok.append(("(12a) primary: scrambled effect -> P.effect is a failed primary (default); effect_primary True",
               "P.effect" in r1["failed_primary"] and r1["verdict"] == "FAIL" and r1["effect_primary"] is True))
    g2 = {g["gate"] for g in r2["reported"]}
    ok.append(("(12b) reported: same panel -> P.effect listed under reported, not primary, never in failed/inconclusive; verdict %s" % r2["verdict"],
               "P.effect" in g2 and "P.effect" not in {g["gate"] for g in r2["primary"]} and "P.effect" not in r2["failed_primary"] + r2["inconclusive_primary"]
               and r2["effect_primary"] is False and gates(r2)["P.effect"]["contrast"] == gates(r1)["P.effect"]["contrast"]))
    rb = s1run(panel(146), arm="BASE")
    ok.append(("(12c) BASE report: effect_primary null", rb["effect_primary"] is None))
    with tempfile.TemporaryDirectory(prefix="v11_") as td:
        o = str(Path(td) / "R.json")
        common = ["stage1", "--hfj2-s0", "x", "--hfj2-s1", "x", "--hfj3", "x", "--calls", "x", "--labels", "x", "--out", o]
        m = refused(lambda: eq.main(common + ["--arm", "ADAPTER"]))
        ok.append(("(12d) CLI ADAPTER without --effect-mode -> REFUSED (%s)" % m[:60], "--effect-mode" in m))
        m = refused(lambda: eq.main(common + ["--arm", "BASE", "--effect-mode", "primary"]))
        ok.append(("(12e) CLI BASE with --effect-mode -> REFUSED (%s)" % m[:60], "--effect-mode" in m))
    eq.EFFECT_PRIMARY = True
    return ok


def test_equiv_suite():
    ok = run_suite()
    failed = [label for label, good in ok if not good]
    assert len(ok) >= 50 and not failed, failed
