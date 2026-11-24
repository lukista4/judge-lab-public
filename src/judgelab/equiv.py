#!/usr/bin/env python3
"""judgelab.equiv: reader-equivalence gates (Stage 0 replay + Stage 1 distributional test).

Two judge readers are compared: HFJ2 (the reference Hugging Face transformers bf16 reader) and HFJ3 (the in-process vLLM reader).
The verifier imports no reader or generator code (numpy + stdlib only). See docs/READER_EQUIVALENCE.md for the reasoning.

STAGE 0 (python -m judgelab.equiv stage0 --rows ROWS.jsonl [ROWS2.jsonl ...] --out REPORT.json)
  Input rows (one per call x arm x cfg; both readers in the SAME schema, produced by judgelab.replay.replay_full):
    {"call_id", "pid", "arm": "BASE"|"ADAPTER", "reader": "HFJ2"|"HFJ3", "cfg": "ref" (HFJ2) | "B1" | "B32" | "warm" (HFJ3),
     "prompt_tokens": int, "E": float (0-7), "p_sev": {NONE, MINOR, MAJOR, FATAL: p}, "sev_mass": float (raw mass),
     "p_digits_by_sev": {sev: {"0".."7": p}} and "digit_mass_by_sev": {sev: float} for every unpruned severity,
     "tops": {"SEV": {token_id(str): logp} top-100 at the severity position, sev: {token_id: logp} top-100 at that sev's score position}}
  B1 = one request at a time, B32 = 32 concurrent requests over the same engine, warm = B1 again with every prefix cached.
  Gates (HFJ3 B1 vs HFJ2 ref, per arm):
    S0.coverage   >= 24 calls per arm, >= 6 with prompt_tokens in [35000, 55000]; every (call, arm) has ref, B1, B32, warm.
    S0.flips      calls whose ref-vs-B1 top-list difference is a within-limit bf16 rounding flip of a class also seen inside HFJ3's own
                  B1/B32/warm spread leave the bars below (<= FLIP_CAP per arm, at least one non-flip call; every flip call's dE reported).
    S0.E          |dE| mean <= 0.015, p95 <= 0.04, max <= 0.08.
    S0.TVD        TVD(p_sev) and TVD(p_digits[sev]) for every live sev: mean <= 0.005, max <= 0.025.
    S0.highmass   tokens with p_HFJ2 >= 0.01: |dlogp| mean <= 0.02, max <= 0.12 (a missing high-mass token fails).
    S0.mass       |d sev_mass| and |d digit_mass|: mean <= 0.005, max <= 0.02.
    S0.adapter    dE = E(ADAPTER) - E(BASE) per call per reader: mean |dE_HFJ2| > 0.05, corr >= 0.995, ratio in [0.92, 1.08].
    S0.invariance HFJ3 B32 vs B1 and warm vs B1: max |dE| <= 0.015 (plus TVD / mass bounds).

STAGE 1 (python -m judgelab.equiv stage1 --hfj2-s0 DIR.. --hfj2-s1 DIR.. --hfj3 DIR.. --calls CALLS --labels LABELS.jsonl --arm BASE|ADAPTER
         [--effect-mode primary|reported] --out R.json)
  corr_rss rule. For every metric M:
    contrast  D   = | mean_k M(HFJ3_k) - (M(HFJ2_s0) + M(HFJ2_s1)) / 2 |          (point values)
    sigma_self    = SD over the paired pid-cluster bootstrap (B = 10,000, fixed seed) of M(HFJ2_s1) - M(HFJ2_s0)
    bar           = sqrt( (Z_RSS * c_K * sigma_self)^2 + m^2 ),  Z_RSS = 2.33, c_K = sqrt((1/K + 1/2) / 2), K = HFJ3 samples per proof
    pass iff valid draws >= 90 % and D <= bar;  INCONCLUSIVE iff it would pass but bar > CEIL[arm][M] (the panel is too noisy to certify).
  The noise scale comes from two independent HFJ2 replicates of the same proofs (s0, s1): the new reader must sit as close to the reference
  as the reference sits to itself. HFJ3 never enters sigma.
  Metrics: pooled AUC (m 0.02), within-problem AUC (m 0.025), E-bias on all / correct / wrong (m 0.10), log mean and log median think
  tokens (m log 1.05), spread ratio SD(E_HFJ3) / SD(E_HFJ2) in [0.88, 1.12], ADAPTER only: adapter-effect correlation (P.effect).
  P.sanity: e_none == 0, missing_share < 0.01, infra_rows / manifest_calls < 0.01.
  Provenance is REFUSED before any statistic: runtime strings, RUN.json penalty_scope / fake flags, one adapter_sha256 and one base_pins
  set across all readers, call ids equal to the manifest.
  Verdict: FAIL if a decisive primary has D > bar (or invalid draws); else INCONCLUSIVE if a decisive primary's bar > its ceiling; else AMBER
  if a secondary (S.shape, S.text, S.health) fails; else PASS. rc PASS 0, FAIL 1, AMBER 2, INCONCLUSIVE 3. SystemExit "REFUSED ..." = bad input.
  Labels are one JSON per line {"a": proof key, "correct": 0|1}; the output holds aggregates only.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

SEVS = ("NONE", "MINOR", "MAJOR", "FATAL")
B, SEED = 10000, 20261124
MIN_CALLS = {"BASE": 192, "ADAPTER": 96}   # pre-registered Stage 1 sizes (paired ok HFJ3 reads)
BARS = {"BASE": dict(pooled=(0.020, 0.050), within=(0.025, 0.060), bias=0.25, think=(0.90, 1.10)),
        "ADAPTER": dict(pooled=(0.025, 0.065), within=(0.030, 0.075), bias=0.35, think=(0.88, 1.12))}


def need(ok, why):
    if not ok:
        raise SystemExit("REFUSED " + why)


def jl(p):
    return [json.loads(x) for x in Path(p).read_text("utf-8").splitlines() if x.strip()]


def tvd(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys)


def gate(name, ok, **vals):
    return {"gate": name, "pass": bool(ok), **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in vals.items()}}


# ----------------------------------------------------------------------------- stage 0
FLIP_LIMIT, FLIP_CAP, FLIP_HM, FLIP_EPS, BF16_GRID = 0.25, 12, 0.01, 1e-6, 0.125   # Stage 0 flip rule (see docs/READER_EQUIVALENCE.md)


def tops_spread(x, y):
    """Max over shared top-100 positions of the spread of log-ratio shifts (y - x logprobs) over tokens with p >= FLIP_HM on either side.
    inf if a live position (SEV, or a sev with p_sev >= FLIP_HM) or a high-mass token is present on one side only."""
    m = 0.0
    for pos in set(x["tops"]) | set(y["tops"]):
        if pos not in x["tops"] or pos not in y["tops"]:
            if pos == "SEV" or max(float(x["p_sev"].get(pos, 0.0)), float(y["p_sev"].get(pos, 0.0))) >= FLIP_HM:
                return float("inf")
            continue
        tx, ty = x["tops"][pos], y["tops"][pos]
        hi = [t for t in set(tx) | set(ty) if max(math.exp(tx.get(t, -1e9)), math.exp(ty.get(t, -1e9))) >= FLIP_HM]
        if any(t not in tx or t not in ty for t in hi):
            return float("inf")
        d = [ty[t] - tx[t] for t in hi]
        if len(d) > 1:
            m = max(m, max(d) - min(d))
    return m


def flip_class(s):
    """0 = identical; k >= 1 = a k-step shift on the bf16 0.125 grid within FLIP_LIMIT; None = beyond the limit (not a flip)."""
    if s <= FLIP_EPS:
        return 0
    if s > FLIP_LIMIT + FLIP_EPS:
        return None
    return 1   # one flip class (each logit within one bf16 step)


def stage0(rows):
    """Per-arm gates; E None = FAIL; prompt_tokens equality; equal call sets; p_sev >= 0.01 branch rule."""
    CF = (("HFJ2", "ref"), ("HFJ3", "B1"), ("HFJ3", "B32"), ("HFJ3", "warm"))
    idx = {}
    for r in rows:
        key = (r["call_id"], r["arm"], r["reader"], r["cfg"])
        need(key not in idx, "duplicate stage0 row %s" % (key,))
        need(r["arm"] in ("BASE", "ADAPTER") and (r["reader"], r["cfg"]) in CF, "stage0 arm / reader / cfg")
        idx[key] = r
    gates = []
    calls = {a: sorted({k[0] for k in idx if k[1] == a}) for a in ("BASE", "ADAPTER")}
    complete = all((c, a, rd, cf) in idx for a in calls for c in calls[a] for rd, cf in CF)
    same = calls["BASE"] == calls["ADAPTER"]
    long_n = {a: sum(1 for c in calls[a] if 35000 <= (idx[(c, a, "HFJ2", "ref")].get("prompt_tokens") or 0) <= 55000) for a in calls} if complete else {}
    gates.append(gate("S0.coverage", complete and same and all(len(calls[a]) >= 24 for a in calls) and all(long_n.get(a, 0) >= 6 for a in calls),
                      n_base=len(calls["BASE"]), n_adapter=len(calls["ADAPTER"]), long_35_55k=long_n, complete=complete, same_calls=same))
    if not (complete and same and all(calls[a] for a in calls)):   # empty arm = FAIL, never a numpy crash
        return {"stage": 0, "gates": gates, "verdict": "FAIL"}
    for (c, a, rd, cf), r in idx.items():   # every cfg needs the digit fields
        need(r.get("p_digits_by_sev") is not None and r.get("digit_mass_by_sev") is not None and r.get("p_sev") is not None,
             "stage0 p_sev / p_digits_by_sev / digit_mass_by_sev missing for %s %s %s %s" % (c, a, rd, cf))
    enone = sum(1 for r in idx.values() if r.get("E") is None)
    gates.append(gate("S0.rows", enone == 0, e_none=enone))
    if enone:
        return {"stage": 0, "gates": gates, "verdict": "FAIL"}
    ptok = sum(1 for (c, a, rd, cf), r in idx.items() if r.get("prompt_tokens") != idx[(c, a, "HFJ2", "ref")].get("prompt_tokens"))
    gates.append(gate("S0.prompt", ptok == 0, prompt_tokens_mismatch=ptok))

    def live(h2, h3, s):
        return max(float(h2["p_sev"].get(s, 0.0)), float(h3["p_sev"].get(s, 0.0))) >= 0.01

    PR = ((("HFJ2", "ref"), ("HFJ3", "B1")), (("HFJ3", "B1"), ("HFJ3", "B32")), (("HFJ3", "B1"), ("HFJ3", "warm")))
    cls = {(c, a, i): flip_class(tops_spread(idx[(c, a) + p[0]], idx[(c, a) + p[1]])) for a in calls for c in calls[a] for i, p in enumerate(PR)}
    own = sorted({k for (c, a, i), k in cls.items() if i > 0 and k})   # flip classes inside HFJ3's own B1/B32/warm spread (panel-wide)
    flips = {a: [c for c in calls[a] if cls[(c, a, 0)] and cls[(c, a, 0)] in own] for a in calls}   # flip calls are defined by the cross pair ref-B1 only

    for arm in ("BASE", "ADAPTER"):
        nf = [c for c in calls[arm] if c not in flips[arm]]   # the original bars run on the non-flip calls
        gates.append(gate("S0.flips[%s]" % arm, len(flips[arm]) <= FLIP_CAP and len(nf) >= 1, flip_calls=len(flips[arm]), cap=FLIP_CAP,
                          limit=FLIP_LIMIT, non_flip_calls=len(nf), own_classes=own,
                          beyond_limit_calls=sum(1 for c in calls[arm] if any(cls[(c, arm, i)] is None for i in range(3))),
                          flip_call_ids=flips[arm], self_flip_pairs=sum(1 for c in calls[arm] for j in (1, 2) if cls[(c, arm, j)]),
                          flip_detail={c: {"ref_b1_spread": round(tops_spread(idx[(c, arm, "HFJ2", "ref")], idx[(c, arm, "HFJ3", "B1")]), 6),
                                           "dE_b1_minus_ref": round(idx[(c, arm, "HFJ3", "B1")]["E"] - idx[(c, arm, "HFJ2", "ref")]["E"], 6)}
                                       for c in flips[arm]}))   # safeguard: report the E difference of every flip call
        if not nf:
            continue
        pairs = [(idx[(c, arm, "HFJ2", "ref")], idx[(c, arm, "HFJ3", "B1")]) for c in nf]
        dE = np.array([abs(h3["E"] - h2["E"]) for h2, h3 in pairs])
        gates.append(gate("S0.E[%s]" % arm, dE.mean() <= 0.015 and np.percentile(dE, 95) <= 0.04 and dE.max() <= 0.08,
                          mean=float(dE.mean()), p95=float(np.percentile(dE, 95)), max=float(dE.max())))
        tv = []
        for h2, h3 in pairs:
            need(h2.get("p_digits_by_sev") is not None and h3.get("p_digits_by_sev") is not None, "stage0 p_digits_by_sev missing for %s" % h2["call_id"])
            tv.append(tvd(h2["p_sev"], h3["p_sev"]))
            for s in set(h2["p_digits_by_sev"]) & set(h3["p_digits_by_sev"]):
                if live(h2, h3, s):
                    tv.append(tvd(h2["p_digits_by_sev"][s], h3["p_digits_by_sev"][s]))
        tv = np.array(tv)
        gates.append(gate("S0.TVD[%s]" % arm, tv.mean() <= 0.005 and tv.max() <= 0.025, mean=float(tv.mean()), max=float(tv.max()), n=len(tv)))
        hm, missing, one_side = [], 0, 0
        for h2, h3 in pairs:
            need(h2.get("tops", {}).get("SEV") and h3.get("tops", {}).get("SEV"), "stage0 tops.SEV missing for %s" % h2["call_id"])
            for pos in set(h2["tops"]) | set(h3["tops"]):
                if pos not in h2["tops"] or pos not in h3["tops"]:   # a sev read by one reader only (SEV_PRUNE boundary)
                    one_side += 1
                    missing += int(pos == "SEV" or live(h2, h3, pos))
                    continue
                t2, t3 = h2["tops"][pos], h3["tops"][pos]
                for tok, lp in t2.items():
                    if math.exp(lp) >= 0.01:
                        if tok in t3:
                            hm.append(abs(t3[tok] - lp))
                        else:
                            missing += 1
        hm = np.array(hm) if hm else np.array([0.0])
        gates.append(gate("S0.highmass[%s]" % arm, missing == 0 and hm.mean() <= 0.02 and hm.max() <= 0.12, mean=float(hm.mean()),
                          max=float(hm.max()), missing_in_hfj3=missing, pruned_one_side=one_side))
        ms = np.array([abs(h3["sev_mass"] - h2["sev_mass"]) for h2, h3 in pairs] +
                      [abs(h3["digit_mass_by_sev"][s] - h2["digit_mass_by_sev"][s]) for h2, h3 in pairs
                       for s in set(h2["digit_mass_by_sev"]) & set(h3["digit_mass_by_sev"]) if live(h2, h3, s)])
        gates.append(gate("S0.mass[%s]" % arm, ms.mean() <= 0.005 and ms.max() <= 0.02, mean=float(ms.mean()), max=float(ms.max())))
        SP = [(c, cf) for c in calls[arm] for j, cf in ((1, "B32"), (2, "warm")) if not cls[(c, arm, j)]]   # self pairs minus within-limit self flips
        ie = np.array([abs(idx[(c, arm, "HFJ3", cf)]["E"] - idx[(c, arm, "HFJ3", "B1")]["E"]) for c, cf in SP] + [0.0])
        it = np.array([tvd(idx[(c, arm, "HFJ3", cf)]["p_sev"], idx[(c, arm, "HFJ3", "B1")]["p_sev"]) for c, cf in SP] + [0.0])
        idt, im, oneside = [0.0], [0.0], 0   # digits / mass / sev keys of B32 and warm vs B1
        for c in calls[arm]:
            b1 = idx[(c, arm, "HFJ3", "B1")]
            for cf in ("B32", "warm"):
                if (c, cf) not in SP:
                    continue
                r = idx[(c, arm, "HFJ3", cf)]
                im.append(abs(r["sev_mass"] - b1["sev_mass"]))
                for s in set(r["p_digits_by_sev"]) | set(b1["p_digits_by_sev"]):
                    if s not in r["p_digits_by_sev"] or s not in b1["p_digits_by_sev"]:
                        oneside += int(live(b1, r, s))
                    elif live(b1, r, s):
                        idt.append(tvd(r["p_digits_by_sev"][s], b1["p_digits_by_sev"][s]))
                        im.append(abs(r["digit_mass_by_sev"][s] - b1["digit_mass_by_sev"][s]))
        gates.append(gate("S0.invariance[%s]" % arm, ie.max() <= 0.015 and it.max() <= 0.025 and max(idt) <= 0.025 and max(im) <= 0.02 and oneside == 0,
                          max_dE=float(ie.max()), max_tvd_sev=float(it.max()), max_tvd_digits=float(max(idt)), max_mass=float(max(im)),
                          live_sev_one_side=oneside))
    d2 = np.array([idx[(c, "ADAPTER", "HFJ2", "ref")]["E"] - idx[(c, "BASE", "HFJ2", "ref")]["E"] for c in calls["BASE"]])
    d3 = np.array([idx[(c, "ADAPTER", "HFJ3", "B1")]["E"] - idx[(c, "BASE", "HFJ3", "B1")]["E"] for c in calls["BASE"]])
    if len(d2) >= 3 and np.abs(d2).mean() > 0.05 and d2.std() > 1e-9 and d3.std() > 1e-9:
        corr, ratio = float(np.corrcoef(d2, d3)[0, 1]), float(np.abs(d3).mean() / np.abs(d2).mean())
        gates.append(gate("S0.adapter", corr >= 0.995 and 0.92 <= ratio <= 1.08, corr=corr, ratio=ratio, mean_abs_dE_hfj2=float(np.abs(d2).mean()),
                          n=len(d2)))
    else:
        why = ("too few paired calls (n < 3)" if len(d2) < 3 else "adapter effect too small (needs mean|dE_HFJ2| > 0.05)"
               if np.abs(d2).mean() <= 0.05 else "dE constant on a reader, correlation undefined (std HFJ2 %.4g, HFJ3 %.4g)" % (d2.std(), d3.std()))
        gates.append(gate("S0.adapter", False, why=why, n=len(d2), mean_abs_dE_hfj2=float(np.abs(d2).mean()) if len(d2) else None))
    return {"stage": 0, "gates": gates, "verdict": "PASS" if all(g["pass"] for g in gates) else "FAIL"}


# ----------------------------------------------------------------------------- stage 1
def load_reads_full(dirs, want_sample=None):
    """READS.jsonl is a journal: per file the LAST line per call_id wins. Returns (ok, bad):
    ok = {(a, sample): score} for status-ok rows (sample-filtered), bad = {call_id: score} for every non-ok row (infra / over; no 'a' needed)."""
    ok, bad, owner = {}, {}, {}
    for d in dirs:
        files = sorted(Path(d).rglob("READS.jsonl")) if Path(d).is_dir() else [Path(d)]
        need(files, "no READS.jsonl under %s" % d)
        for f in files:
            last = {}
            for r in jl(f):
                s = r.get("score", r) if isinstance(r, dict) else None
                need(isinstance(s, dict) and s.get("call_id"), "READS row without call_id in %s" % f)
                last[s["call_id"]] = s
            for cid, s in last.items():
                need(cid not in owner, "call_id %s in two READS files (%s, %s)" % (cid, owner.get(cid), f))
                owner[cid] = str(f)
                if s.get("status", "ok") != "ok":
                    bad[cid] = s
                    continue
                if want_sample is not None and s.get("sample") != want_sample:
                    continue
                need("a" in s, "ok row without 'a': %s" % cid)
                key = (s["a"], s.get("sample"))
                need(key not in ok, "two ok calls for one (a, sample) %s (multi-lens / multi-kind input?)" % (key,))
                ok[key] = s
    arms = {v.get("arm") for v in ok.values()}
    need(len(arms) == 1, "one arm per READS set, got %s (pass the arm dir, e.g. <kit>/BASE, not the kit dir)" % sorted(map(str, arms)))
    return ok, bad


def load_reads(dirs, want_sample=None):
    return load_reads_full(dirs, want_sample)[0]


def weights(pids, uniq, rng):
    pos = {p: i for i, p in enumerate(uniq)}
    col = np.array([pos[p] for p in pids])
    draws = rng.integers(0, len(uniq), (B, len(uniq)))
    w = np.zeros((B, len(uniq)))
    for j in range(len(uniq)):
        w[:, j] = (draws == j).sum(1)
    return w[:, col]


def kern(e, y):
    d = e[y][:, None] - e[~y][None, :]
    return (d > 0) + 0.5 * (d == 0)


def auc_boot(e, y, pid, w):
    """point pooled AUC, point within AUC, bootstrap arrays of both (pid-cluster bootstrap weighting)."""
    k = kern(e, y)
    m = (pid[y][:, None] == pid[~y][None, :]).astype(float)
    pa = float(k.mean())
    pw = float((k * m).sum() / m.sum()) if m.sum() else float("nan")
    p, n = w[:, y], w[:, ~y]
    with np.errstate(invalid="ignore", divide="ignore"):   # single-class draws -> NaN, counted as invalid
        ba = ((p @ k) * n).sum(1) / (p.sum(1) * n.sum(1))
    r = np.sqrt(w)
    rp, rn = r[:, y], r[:, ~y]
    with np.errstate(invalid="ignore", divide="ignore"):
        bw = ((rp @ (k * m)) * rn).sum(1) / ((rp @ m) * rn).sum(1)
    return pa, pw, ba, bw


def emd(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.sum(np.abs(ca - cb)[:-1] * np.diff(grid)))


def distinct4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / len(g) if g else 0.0


def mwu_p(x, y):
    """two-sided Mann-Whitney U, normal approximation with tie correction."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    allv = np.concatenate([x, y])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n2 = len(x), len(y)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(allv, return_counts=True)
    nt = n1 + n2
    var = n1 * n2 / 12 * ((nt + 1) - (counts ** 3 - counts).sum() / (nt * (nt - 1)))
    if var <= 0:
        return 1.0
    z = (abs(u - n1 * n2 / 2) - 0.5) / math.sqrt(var)
    return float(math.erfc(max(z, 0) / math.sqrt(2)))


def think_ids(s):
    """the generated think tokens only: probe_ids = prompt (think_prompt_n) + think + close (real HFJ2 rows: len = n + think_tokens + 2,
    probe_close_idx = n + think_tokens). Missing layout fields -> REFUSED (never distinct-4 of the prompt)."""
    ids, c, n = s.get("probe_ids"), s.get("probe_close_idx"), s.get("think_prompt_n")
    need(isinstance(ids, list) and isinstance(c, int) and isinstance(n, int) and 0 <= n <= c <= len(ids), "probe_ids layout fields")
    return ids[n:c]


MARGIN = {"pooled": 0.02, "within": 0.025, "bias": 0.10, "think": 0.05, "shape": 0.10, "text": 0.02}
COVER, MIN_VALID, Z_SELF = 0.98, 0.90, 1.96
Z_RSS = 2.33   # corr_rss z (one-sided ~1 %)
EFFECT_PRIMARY = True   # False = P.effect is reported only on ADAPTER; the CLI sets it explicitly
CEIL = {"BASE": {"pooled": 0.06, "within": 0.10, "bias": 0.30, "bias_split": 0.40, "think": 0.10},   # INCONCLUSIVE ceilings (pre-registered)
        "ADAPTER": {"pooled": 0.12, "within": 0.17, "effect": 0.35}}
M_LOGTHINK, M_EFFECT, SPREAD = math.log(1.05), 0.05, (0.88, 1.12)
MISS_MAX, INFRA_MAX = 0.01, 0.01   # missing_share < 0.01 and infra rows < 1 % of the manifest (strict)
RUNTIME2, RUNTIME3 = "hf_bf16", "hfj3_vllm033_bf16"


def c_k(k):
    return math.sqrt((1.0 / k + 0.5) / 2.0)


def manifest(calls):
    man = {}
    for c in calls:
        m = re.fullmatch(r"(.+):([^:]+):s(\d+)", c)
        need(m is not None, "manifest call_id format <a>:<lens>:s<k>: %s" % c)
        key = (m.group(1), int(m.group(3)))
        need(key not in man, "manifest has two calls for %s" % (key,))
        man[key] = c
    return man


def hfj3_runs(dirs):
    """The RUN.json next to every HFJ3 READS.jsonl (penalty_scope / fake / runtime live there, not in the rows)."""
    runs = []
    for d in dirs:
        files = sorted(Path(d).rglob("READS.jsonl")) if Path(d).is_dir() else [Path(d)]
        for f in files:
            rj = f.parent / "RUN.json"
            need(rj.is_file(), "HFJ3 RUN.json missing next to %s" % f)
            runs.append(json.loads(rj.read_text("utf-8")))
    need(runs, "no HFJ3 RUN.json")
    return runs


def pins_key(r):
    return json.dumps(r.get("base_pins"), sort_keys=True)


def provenance(h0, h1, hf3, proofs, man, runs3, extra=()):
    """REFUSED (before any statistic) unless every reader row carries the expected runtime and ONE adapter_sha256 / base_pins set is used.
    extra = (rows, runtime, adapter_expected_None) triples for the adapter-effect base readers."""
    r2 = [h0[a] for a in proofs] + [h1[a] for a in proofs]
    r3 = list(hf3.values())
    need(all(r.get("runtime") == RUNTIME2 for r in r2), "HFJ2 rows: runtime != %s" % RUNTIME2)
    need(all(r.get("runtime") == RUNTIME3 for r in r3), "HFJ3 rows: runtime != %s" % RUNTIME3)
    need(runs3 is not None and len(runs3) > 0, "HFJ3 RUN.json provenance required")
    for rn in runs3:
        need(rn.get("runtime") == RUNTIME3, "HFJ3 RUN.json runtime %s" % rn.get("runtime"))
        need(rn.get("penalty_scope") == "prompt", "HFJ3 RUN.json penalty_scope %s != prompt" % rn.get("penalty_scope"))
        need(rn.get("fake") is False, "HFJ3 RUN.json fake is not False")
    ad = {json.dumps(r.get("adapter_sha256"), sort_keys=True) for r in r2 + r3} | {json.dumps(rn.get("adapter_sha256"), sort_keys=True) for rn in runs3}
    need(len(ad) == 1, "adapter_sha256 differs across HFJ2_s0 / HFJ2_s1 / HFJ3: %d values" % len(ad))
    bp = {pins_key(r) for r in r2 + r3} | {pins_key(rn) for rn in runs3 if "base_pins" in rn}
    for rows, rt, _ in extra:
        need(all(r.get("runtime") == rt for r in rows), "adapter-effect base reader rows: runtime != %s" % rt)
        need(all(r.get("adapter_sha256") is None for r in rows), "adapter-effect base reader rows carry an adapter")
        bp |= {pins_key(r) for r in rows}
    need(len(bp) == 1, "base_pins differ across readers: %d values" % len(bp))
    for a in proofs:
        for k, h in ((0, h0), (1, h1)):
            want = man.get((a, k)) or re.sub(r":s\d+$", ":s%d" % k, next(c for (b, _), c in man.items() if b == a))
            need(h[a].get("call_id") == want, "HFJ2_s%d call_id %s != %s" % (k, h[a].get("call_id"), want))


def auc_point(e, y, pid):
    k = kern(e, y)
    m = (pid[y][:, None] == pid[~y][None, :]).astype(float)
    return float(k.mean()), (float((k * m).sum() / m.sum()) if m.sum() else float("nan"))


def wmean(w, x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ x) / w.sum(1)


def wmedian(w, x):
    o = np.argsort(x, kind="mergesort")
    cs = np.cumsum(w[:, o], 1)
    idx = (cs < cs[:, -1:] / 2.0).sum(1)
    return x[o][np.minimum(idx, len(x) - 1)]


def wcorr(w, x, y):
    with np.errstate(invalid="ignore", divide="ignore"):
        sw = w.sum(1)
        mx, my = (w @ x) / sw, (w @ y) / sw
        cxy = (w @ (x * y)) / sw - mx * my
        vx, vy = (w @ (x * x)) / sw - mx * mx, (w @ (y * y)) / sw - my * my
        return cxy / np.sqrt(vx * vy)


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")


def stage1(s0, s1, hf3, labels, arm, calls, hf3_bad=None, runs3=None, base_reads=None):
    """corr_rss Stage 1 (module doc). s0 / s1 = load_reads(HFJ2 dirs, 0 / 1); hf3 / hf3_bad = load_reads_full(HFJ3 dirs); runs3 = hfj3_runs(HFJ3
    dirs); base_reads (ADAPTER only) = dict(s0=, s1=, hfj3=) BASE-arm reads of the same proofs by HFJ2 s0 / HFJ2 s1 / HFJ3 sample 0."""
    need(calls, "--calls manifest required")
    man = manifest(calls)
    K = sorted({k for _, k in man})
    proofs = sorted({a for a, _ in man})
    need(all((a, k) in man for a in proofs for k in K), "manifest not rectangular (every proof x every sample)")
    need(len(man) >= MIN_CALLS[arm], "manifest has %d calls < planned %d" % (len(man), MIN_CALLS[arm]))
    lab = {}
    for r in labels:
        need(r["correct"] in (0, 1) and r["a"] not in lab, "labels: correct 0|1, one row per key")
        lab[r["a"]] = bool(r["correct"])
    need(all(a in lab for a in proofs), "labels missing for manifest proofs")

    def usable(s):
        return s is not None and s.get("status", "ok") == "ok" and s.get("E") is not None
    h0 = {k[0]: v for k, v in s0.items() if k[1] == 0}
    h1 = {k[0]: v for k, v in s1.items() if k[1] == 1}
    miss2 = [a for a in proofs if not (usable(h0.get(a)) and usable(h1.get(a)))]
    need(not miss2, "HFJ2_s0 / HFJ2_s1 not usable on %d of %d manifest proofs (read them first)" % (len(miss2), len(proofs)))
    rarms = {v.get("arm") for v in [h0[a] for a in proofs] + [h1[a] for a in proofs] + list(hf3.values())}
    need(len(rarms) == 1, "READS arm differs across readers: %s" % sorted(map(str, rarms)))
    ra = next(iter(rarms))
    need((ra == "BASE") == (arm == "BASE"), "--arm %s does not match the READS arm %s" % (arm, ra))
    need(all(k in man for k in hf3), "HFJ3 rows outside the manifest: %d" % sum(1 for k in hf3 if k not in man))
    bad3 = dict(hf3_bad or {})
    need(all(c in set(calls) for c in bad3), "HFJ3 non-ok rows outside the manifest")
    for (a, k), r in hf3.items():
        need(r.get("call_id") == man[(a, k)], "HFJ3 call_id %s != manifest %s" % (r.get("call_id"), man[(a, k)]))
        need(r.get("pid") == h0[a].get("pid") == h1[a].get("pid"), "pid differs across readers for %s" % a)
    extra, eff = [], None
    if arm == "ADAPTER":
        need(base_reads is not None, "ADAPTER needs the BASE-arm reads of the same proofs (--base-hfj2-s0 / --base-hfj2-s1 / --base-hfj3) for P.effect")
        g0 = {k[0]: v for k, v in base_reads["s0"].items() if k[1] == 0}
        g1 = {k[0]: v for k, v in base_reads["s1"].items() if k[1] == 1}
        g3 = {k[0]: v for k, v in base_reads["hfj3"].items() if k[1] == 0}
        need(all(usable(g0.get(a)) and usable(g1.get(a)) for a in proofs), "BASE-arm HFJ2 s0 / s1 not usable on every ADAPTER proof")
        need(all(v.get("arm") == "BASE" for v in list(g0.values()) + list(g1.values()) + list(g3.values())), "base_reads arm != BASE")
        for a in proofs:
            need(g0[a].get("pid") == g1[a].get("pid") == h0[a].get("pid"), "pid differs in base_reads for %s" % a)
            need(g0[a].get("call_id") == h0[a].get("call_id") and g1[a].get("call_id") == h1[a].get("call_id"), "base_reads call_id != adapter call_id for %s" % a)
            if a in g3:
                need(g3[a].get("call_id") == h0[a].get("call_id"), "HFJ3 base read call_id != %s" % h0[a].get("call_id"))
        extra = [([g0[a] for a in proofs] + [g1[a] for a in proofs], RUNTIME2, True), (list(g3.values()), RUNTIME3, True)]
        eff = (g0, g1, g3)
    provenance(h0, h1, hf3, proofs, man, runs3, extra)
    ok_calls = [key for key in man if usable(hf3.get(key))]
    cover = len(ok_calls) / len(man)
    e_none = sum(1 for v in hf3.values() if v.get("status", "ok") == "ok" and v.get("E") is None)
    Q = [a for a in proofs if all(usable(hf3.get((a, k))) for k in K)]
    if eff is not None:
        Q = [a for a in Q if usable(eff[2].get(a))]
    need(len(Q) >= 10, "fewer than 10 proofs with every HFJ3 sample usable")
    pid = np.array([h0[a]["pid"] for a in Q])
    y = np.array([lab[a] for a in Q])
    need(y.any() and (~y).any(), "both classes needed")
    uniq = sorted(set(pid))
    mixed = sum(1 for p in uniq if len(set(y[pid == p])) == 2)
    rng = np.random.default_rng(SEED)
    w = weights(pid, uniq, rng)
    ck = c_k(len(K))
    C = CEIL[arm]
    decide_extra = arm == "BASE"   # pre-registered: ADAPTER E-bias / think are reported only
    out = {"stage": 1, "rule": "corr_rss (self-replication anchored)", "arm": arm, "reads_arm": ra, "K": K, "c_K": ck, "z": Z_RSS,
           "manifest_calls": len(man), "usable_hfj3_calls": len(ok_calls), "proofs_scored": len(Q), "problems": len(uniq), "mixed_problems": mixed,
           "B": B, "seed": SEED, "cluster": "pid", "sigma_self": "SD of paired bootstrap draws of M(s1)-M(s0)", "margins": MARGIN,
           "ceilings": C, "primary": [], "secondary": [], "reported": [],
           "effect_primary": bool(EFFECT_PRIMARY) if arm == "ADAPTER" else None}
    P, S, R = out["primary"], out["secondary"], out["reported"]
    miss = len(man) - len(ok_calls)
    P.append(gate("P.sanity", e_none == 0 and miss / len(man) < MISS_MAX and len(bad3) / len(man) < INFRA_MAX, e_none=e_none,
                  missing_calls=miss, missing_share=miss / len(man), infra_rows=len(bad3), infra_share=len(bad3) / len(man),
                  need="e_none 0, missing_share < %.2f, infra_share < %.2f" % (MISS_MAX, INFRA_MAX)))
    P.append(gate("P.coverage", cover >= COVER, usable_share=cover, need=COVER, missing_calls=miss, infra_rows=len(bad3)))

    def rss(name, b0, b1, p0, p1, p3, margin, ceil, where):
        """corr_rss gate. b0/b1 = bootstrap draws of M(s0)/M(s1); p* = point values (p3 one per HFJ3 sample)."""
        with np.errstate(invalid="ignore"):
            ds = b1 - b0
        valid = float(np.mean(~np.isnan(ds)))
        sig = float(np.nanstd(ds)) if valid > 0 else float("nan")
        d = float(abs(np.mean(p3) - (p0 + p1) / 2.0))
        bar = float(math.sqrt((Z_RSS * ck * sig) ** 2 + margin ** 2)) if math.isfinite(sig) else float("nan")
        within = valid >= MIN_VALID and math.isfinite(d) and math.isfinite(bar) and d <= bar
        inconc = bool(within and ceil is not None and bar > ceil)
        g = gate(name, within and not inconc, contrast=d, bar=bar, sigma_self=sig, margin=margin, ceiling=ceil, inconclusive=inconc,
                 within_bar=bool(within), valid_draws=valid, hfj2_s0=float(p0), hfj2_s1=float(p1), hfj3=[float(x) for x in p3])
        where.append(g)
        return g
    e0 = np.array([h0[a]["E"] for a in Q], float)
    e1 = np.array([h1[a]["E"] for a in Q], float)
    e3 = [np.array([hf3[(a, k)]["E"] for a in Q], float) for k in K]
    A0, A1 = auc_boot(e0, y, pid, w), auc_boot(e1, y, pid, w)
    A3 = [auc_point(e, y, pid) for e in e3]
    rss("P.pooled", A0[2], A1[2], A0[0], A1[0], [x[0] for x in A3], MARGIN["pooled"], C["pooled"], P)
    rss("P.within", A0[3], A1[3], A0[1], A1[1], [x[1] for x in A3], MARGIN["within"], C["within"], P)
    bias = []
    for split, mask in (("all", np.ones(len(Q), bool)), ("correct", y), ("wrong", ~y)):
        ww = w[:, mask]
        g = rss("bias." + split, wmean(ww, e0[mask]), wmean(ww, e1[mask]), float(e0[mask].mean()), float(e1[mask].mean()),
                [float(e[mask].mean()) for e in e3], MARGIN["bias"], (C["bias"] if split == "all" else C["bias_split"]) if decide_extra else None, bias)
    gb = gate("P.bias" if decide_extra else "R.bias", all(x["pass"] for x in bias), margin=MARGIN["bias"], splits=bias,
              inconclusive=any(x["inconclusive"] for x in bias), within_bar=all(x["within_bar"] for x in bias))
    (P if decide_extra else R).append(gb)
    t0 = np.array([h0[a].get("think_tokens") or 0 for a in Q], float)
    t1 = np.array([h1[a].get("think_tokens") or 0 for a in Q], float)
    t3 = [np.array([hf3[(a, k)].get("think_tokens") or 0 for a in Q], float) for k in K]
    need(t0.sum() > 0 and t1.sum() > 0, "HFJ2 think_tokens missing")
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = [np.log(wmean(w, t0)), np.log(wmean(w, t1))]
        lm = [np.log(wmedian(w, t0)), np.log(wmedian(w, t1))]
    lp3 = [math.log(t.mean()) if t.mean() > 0 else float("nan") for t in t3]
    mp3 = [math.log(np.median(t)) if np.median(t) > 0 else float("nan") for t in t3]
    tw = P if decide_extra else R
    rss("P.think" if decide_extra else "R.think", lt[0], lt[1], math.log(t0.mean()), math.log(t1.mean()), lp3, M_LOGTHINK,
        C["think"] if decide_extra else None, tw)
    tw[-1]["definition"] = "M = log(mean think tokens); contrast in log units (0.0488 = 5 %)"
    rss("P.think_median" if decide_extra else "R.think_median", lm[0], lm[1], math.log(max(np.median(t0), 1e-9)),
        math.log(max(np.median(t1), 1e-9)), mp3, M_LOGTHINK, None, tw)
    tw[-1]["definition"] = "M = log(median think tokens); no INCONCLUSIVE ceiling (design choice)"
    sd2 = (float(e0.std()) + float(e1.std())) / 2.0
    sd3 = float(np.mean([e.std() for e in e3]))
    ratio = sd3 / sd2 if sd2 > 0 else float("nan")
    P.append(gate("P.spread", math.isfinite(ratio) and SPREAD[0] <= ratio <= SPREAD[1], ratio=ratio, sd_hfj3=sd3, sd_hfj2=sd2, band=list(SPREAD),
                  self_ratio=float(e1.std() / e0.std()) if e0.std() > 0 else None))
    if eff is not None:
        g0, g1, g3 = eff
        d0 = e0 - np.array([g0[a]["E"] for a in Q], float)
        d1 = e1 - np.array([g1[a]["E"] for a in Q], float)
        d3 = e3[0] - np.array([g3[a]["E"] for a in Q], float)
        rself, r3 = corr(d0, d1), (corr(d3, d0) + corr(d3, d1)) / 2.0
        bself = wcorr(w, d0, d1)   # HFJ2-only draws; HFJ3 never enters sigma
        valid = float(np.mean(~np.isnan(bself)))
        sig = float(np.nanstd(bself)) if valid > 0 else float("nan")
        cst = max(0.0, rself - r3) if math.isfinite(rself) and math.isfinite(r3) else float("nan")
        bar = float(math.sqrt((Z_RSS * math.sqrt(2.0) * sig) ** 2 + M_EFFECT ** 2)) if math.isfinite(sig) else float("nan")
        within = valid >= MIN_VALID and math.isfinite(cst) and math.isfinite(bar) and cst <= bar
        inconc = bool(within and bar > C["effect"])
        (P if EFFECT_PRIMARY else R).append(gate("P.effect", within and not inconc, contrast=cst, corr_self=rself, corr_hfj3=r3, bar=bar, sigma=sig, margin=M_EFFECT,
                      ceiling=C["effect"], inconclusive=inconc, within_bar=bool(within), valid_draws=valid,
                      mean_abs_effect_hfj2=float((np.abs(d0).mean() + np.abs(d1).mean()) / 2), mean_abs_effect_hfj3=float(np.abs(d3).mean()),
                      definition="d = E(adapter) - E(base) per proof and reader; corr_self = corr(d_s0, d_s1); corr_hfj3 = mean_j corr(d_hfj3, d_sj);"
                                 " contrast = max(0, corr_self - corr_hfj3); sigma = bootstrap SD of corr_self (HFJ2 only); bar = sqrt((z sqrt2 sigma)^2 + m^2)"))
    rs = np.random.default_rng(SEED + 2)
    groups = {p: np.flatnonzero(pid == p) for p in uniq}
    be = []
    for _ in range(1000):
        ix = np.concatenate([groups[uniq[j]] for j in rs.integers(0, len(uniq), len(uniq))])
        be.append(emd(e1[ix], e0[ix]))
    be = np.array(be)
    em3 = float(np.mean([emd(e, e0) for e in e3]))
    S.append(gate("S.shape", em3 <= float(np.percentile(be, 95)) + MARGIN["shape"], emd_hfj3=em3, emd_self=emd(e1, e0),
                  emd_self_p95=float(np.percentile(be, 95)), margin=MARGIN["shape"], boot=1000))
    text = {}
    for L in (2048, 4096, 8192):
        J = [a for a in Q if len(think_ids(h0[a])) >= L and len(think_ids(h1[a])) >= L and all(len(think_ids(hf3[(a, k)])) >= L for k in K)]
        n0 = sum(1 for a in Q if len(think_ids(h0[a])) >= L)
        if len(J) >= 10:
            q0 = np.array([distinct4(think_ids(h0[a])[:L]) for a in J])
            q1 = np.array([distinct4(think_ids(h1[a])[:L]) for a in J])
            q3 = np.mean([[distinct4(think_ids(hf3[(a, k)])[:L]) for a in J] for k in K], axis=0)
            gap3, gaps = float(abs((q3 - q0).mean())), float(abs((q1 - q0).mean()))
            text[str(L)] = {"joint": len(J), "gap_hfj3": gap3, "gap_self": gaps, "ok": gap3 <= gaps + MARGIN["text"]}
        else:
            text[str(L)] = {"joint": len(J), "hfj2_s0_eligible": n0, "ok": n0 < 10}
    S.append(gate("S.text", all(v["ok"] for v in text.values()), distinct4=text, note="jointly eligible proofs, paired, self-relative"))
    R0, R3 = [h0[a] for a in Q], [hf3[(a, k)] for a in Q for k in K]

    def share(rows, f):
        return float(np.mean([1.0 if f(r) else 0.0 for r in rows]))
    h = dict(natural=share(R3, lambda r: r.get("think_class") == "natural") - share(R0, lambda r: r.get("think_class") == "natural"),
             forced_loop=share(R3, lambda r: r.get("think_class") == "forced_loop") - share(R0, lambda r: r.get("think_class") == "forced_loop"),
             wdm=float(np.mean([r.get("weighted_digit_mass") or 0 for r in R3]) - np.mean([r.get("weighted_digit_mass") or 0 for r in R0])),
             gen_is_answer=share(R3, lambda r: bool(r.get("gen_is_answer"))) - share(R0, lambda r: bool(r.get("gen_is_answer"))),
             sev_tvd=tvd({s: share(R3, lambda r, s=s: r.get("severity") == s) for s in SEVS},
                         {s: share(R0, lambda r, s=s: r.get("severity") == s) for s in SEVS}),
             e_none=e_none, infra_rows=len(bad3), missing_share=miss / len(man))
    S.append(gate("S.health", abs(h["natural"]) <= 0.05 and abs(h["forced_loop"]) <= 0.03 and abs(h["wdm"]) <= 0.02 and
                  abs(h["gen_is_answer"]) <= 0.03 and h["sev_tvd"] <= 0.08 and h["e_none"] == 0 and h["missing_share"] < MISS_MAX, **h))
    fail = [g["gate"] for g in P if not g["pass"] and not g.get("inconclusive")]
    inc = [g["gate"] for g in P if g.get("inconclusive")]
    out["verdict"] = "FAIL" if fail else ("INCONCLUSIVE" if inc else ("PASS" if all(g["pass"] for g in S) else "AMBER"))
    out["failed_primary"], out["inconclusive_primary"] = fail, inc
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a0 = sub.add_parser("stage0")
    a0.add_argument("--rows", nargs="+", required=True)
    a0.add_argument("--out", required=True)
    a1 = sub.add_parser("stage1")
    for k in ("--hfj2-s0", "--hfj2-s1", "--hfj3"):
        a1.add_argument(k, nargs="+", required=True)
    for k in ("--base-hfj2-s0", "--base-hfj2-s1", "--base-hfj3"):
        a1.add_argument(k, nargs="+", help="ADAPTER only: BASE-arm reads of the same proofs (P.effect)")
    a1.add_argument("--calls", required=True, help="HFJ3 CALLS manifest (call_id per line or JSONL with call_id)")
    a1.add_argument("--labels", required=True)
    a1.add_argument("--arm", choices=("BASE", "ADAPTER"), required=True)
    a1.add_argument("--effect-mode", choices=("primary", "reported"), help="ADAPTER only, REQUIRED: whether P.effect decides (primary) or is reported only")
    a1.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    need(not Path(a.out).exists(), "out exists")
    if a.cmd == "stage1":   # the P.effect mode must be stated for ADAPTER and is meaningless for BASE
        need((a.arm == "ADAPTER") == (a.effect_mode is not None), "--effect-mode is required for --arm ADAPTER and refused for BASE")
        global EFFECT_PRIMARY
        EFFECT_PRIMARY = a.effect_mode != "reported"
    if a.cmd == "stage0":
        rep = stage0([r for f in a.rows for r in jl(f)])
    else:
        ok3, bad3 = load_reads_full(a.hfj3)
        cl = [ln.strip() for ln in Path(a.calls).read_text("utf-8").splitlines() if ln.strip()]
        cl = [json.loads(x)["call_id"] if x.startswith("{") else x for x in cl]
        base = None
        if a.arm == "ADAPTER":
            need(a.base_hfj2_s0 and a.base_hfj2_s1 and a.base_hfj3, "ADAPTER needs --base-hfj2-s0 / --base-hfj2-s1 / --base-hfj3")
            base = dict(s0=load_reads(a.base_hfj2_s0, 0), s1=load_reads(a.base_hfj2_s1, 1), hfj3=load_reads(a.base_hfj3, 0))
        else:
            need(not (a.base_hfj2_s0 or a.base_hfj2_s1 or a.base_hfj3), "--base-* inputs are for --arm ADAPTER only")
        rep = stage1(load_reads(a.hfj2_s0, 0), load_reads(a.hfj2_s1, 1), ok3, jl(a.labels), a.arm, cl, bad3, hfj3_runs(a.hfj3), base)
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("COMPUTED %s verdict %s" % (a.cmd, rep["verdict"]))
    return {"PASS": 0, "AMBER": 2, "INCONCLUSIVE": 3}.get(rep["verdict"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
