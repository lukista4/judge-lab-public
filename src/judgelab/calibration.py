"""Human-anchored calibration of an LLM proof grader, and paired agreement between graders (stdlib only).

Method (used with 40 items of the public ProofBench dataset, Ma et al. 2025, expert 0-7 marks; see docs/GRADER_CALIBRATION.md):

  select_stratified   deterministic band-stratified selection: quotas per human grade, one item per problem where possible
                      (a second pass allows a repeated problem only if a grade cell is short), hash order with a fixed seed,
                      and ids re-assigned in a shuffled order so neither id nor position carries grade information.
  analyze             grader marks vs human marks: quadratic weighted kappa with a seeded bootstrap interval, mean bias
                      (grader minus human), MAE, exact share, within-1 share, Spearman rho, the 8x8 confusion matrix, and the
                      same statistics per human band (low 0-2 / mid 3-5 / high 6-7) and per proof generator.
                      Unmarked items are counted and listed, never dropped silently.
  copy_baseline       the "copy the sibling marks" baseline: predict round(mean of the other expert marks of the same problem)
                      without reading the proof. A grader that cannot beat it is not reading the proof.
  paired_agreement    paired agreement of a candidate grader with a reference grader (not with humans): mean difference,
                      exact, within-1, QWK, and precision / recall of "score >= hi" against the reference.

Nothing here contains data. `synthetic_items` / `synthetic_grader` produce demo data for tests and examples.
"""
import argparse
import hashlib
import io
import json
import math
import random
import sys
from pathlib import Path

K = 8  # grades 0..7
BANDS = ("low(0-2)", "mid(3-5)", "high(6-7)")


def band(g):
    return BANDS[0] if g <= 2 else (BANDS[1] if g <= 5 else BANDS[2])


def qwk(pairs, lo=0, hi=7):
    """pairs = [(human, grader)]; None when the expected-disagreement denominator is 0 (e.g. one grade only)."""
    n = len(pairs)
    if n == 0:
        return None
    k = hi - lo + 1
    O = [[0.0] * k for _ in range(k)]
    for h, c in pairs:
        O[h - lo][c - lo] += 1
    rh = [sum(O[i]) for i in range(k)]
    rc = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * O[i][j]
            den += w * rh[i] * rc[j] / n
    return None if den == 0 else 1 - num / den


def _rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
            j += 1
        for t in range(i, j + 1):
            r[s[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(pairs):
    if len(pairs) < 3:
        return None
    a, b = _rank([p[0] for p in pairs]), _rank([p[1] for p in pairs])
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return None if sa == 0 or sb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def stats(pairs):
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    d = [c - h for h, c in pairs]
    return {"n": n, "qwk": qwk(pairs), "mean_bias_grader_minus_human": sum(d) / n, "mae": sum(abs(x) for x in d) / n,
            "exact": sum(x == 0 for x in d) / n, "within1": sum(abs(x) <= 1 for x in d) / n, "spearman": spearman(pairs)}


def boot_qwk(pairs, B=2000, seed=1115):
    """Percentile 95% interval of QWK over item resamples; None if fewer than 100 replicates are defined."""
    rng = random.Random(seed)
    v = []
    for _ in range(B):
        q = qwk([pairs[rng.randrange(len(pairs))] for _ in pairs])
        if q is not None:
            v.append(q)
    if len(v) < 100:
        return None
    v.sort()
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]


def analyze(items, grades, B=2000, seed=1115):
    """items: {id: {'human': int 0..7, 'generator': str}}; grades: {id: int 0..7 or None}. Returns an aggregate dict."""
    unknown = sorted(i for i in grades if i not in items)
    missing, rows = [], []
    for i, it in items.items():
        c = grades.get(i)
        if c is None or not (isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 7):
            missing.append(i)
            continue
        rows.append((i, it, c))
    pairs = [(it["human"], c) for _, it, c in rows]
    res = {"n_items": len(items), "n_scored": len(pairs), "unmarked": sorted(missing), "unknown_ids": unknown,
           "overall": stats(pairs), "qwk_boot95": boot_qwk(pairs, B, seed) if pairs else None}
    res["by_band"] = {b: stats([(it["human"], c) for _, it, c in rows if band(it["human"]) == b]) for b in BANDS}
    gens = sorted({it.get("generator", "-") for _, it, _ in rows})
    res["by_generator"] = {g: stats([(it["human"], c) for _, it, c in rows if it.get("generator", "-") == g]) for g in gens}
    cm = [[0] * K for _ in range(K)]
    for h, c in pairs:
        cm[h][c] += 1
    res["confusion_human_rows_grader_cols"] = cm
    res["misses_abs_diff_ge_3"] = sum(1 for h, c in pairs if abs(h - c) >= 3)
    return res


# ---------------------------------------------------------------------------------------------------- selection
def _h(seed, *a):
    return hashlib.sha1("|".join(map(str, (seed,) + a)).encode()).hexdigest()


def select_stratified(rows, quota, seed="calib-demo", exclude_generators=(), max_year=None, min_len=600, max_len=30000):
    """rows: dicts with problem_id, response_id, grade (0..7), generator, year, proof. quota: {grade: count}.

    Filters (each documented in GRADER_CALIBRATION.md): year <= max_year (contamination), generator not in
    exclude_generators (do not calibrate a grader on its own family's proofs), proof length in [min_len, max_len]
    (drops stubs). Returns (chosen rows with new ids cal_01.., report dict)."""
    ok = [r for r in rows if (max_year is None or r["year"] <= max_year) and r["generator"] not in exclude_generators
          and min_len <= len(r["proof"]) <= max_len]
    chosen, used, short = [], set(), {}
    for g, n in sorted(quota.items()):
        pool = sorted((r for r in ok if r["grade"] == g), key=lambda r: _h(seed, r["problem_id"], r["response_id"]))
        got = 0
        for second in (False, True):
            for r in pool:
                if got >= n:
                    break
                if r in chosen or (not second and r["problem_id"] in used):
                    continue
                chosen.append(r)
                used.add(r["problem_id"])
                got += 1
        if got < n:
            short[g] = (got, n)
    chosen = sorted(chosen, key=lambda r: _h(seed, "order", r["problem_id"], r["response_id"]))
    out = [dict(r, id="cal_%02d" % i) for i, r in enumerate(chosen, 1)]
    rep = {"eligible": len(ok), "chosen": len(out), "short_cells": short,
           "distinct_problems": len({r["problem_id"] for r in out})}
    return out, rep


def copy_baseline(rows):
    """rows: dicts with problem_id and grade. Predict each grade as round-half-up(mean of the OTHER grades of the same
    problem). Returns aggregate stats (problems with a single proof are skipped)."""
    by = {}
    for r in rows:
        by.setdefault(r["problem_id"], []).append(r)
    pairs = []
    for prob, rs in by.items():
        for i, r in enumerate(rs):
            sib = [x["grade"] for j, x in enumerate(rs) if j != i]
            if sib:
                pairs.append((r["grade"], int(math.floor(sum(sib) / len(sib) + 0.5))))
    s = stats(pairs)
    return s


# ---------------------------------------------------------------------------------------------------- grader vs grader
def paired_agreement(cand, ref, hi=6):
    """cand, ref: {id: score 0..7}. Agreement of a candidate grader with a reference grader on shared ids.
    This is agreement, not accuracy: if the reference is biased against humans, high agreement inherits that bias."""
    ids = sorted(i for i in ref if i in cand and cand[i] is not None and ref[i] is not None)
    pairs = [(cand[i], ref[i]) for i in ids]
    n = len(pairs)
    if not n:
        return {"n": 0}
    tp = sum(x >= hi and y >= hi for x, y in pairs)
    fp = sum(x >= hi and y < hi for x, y in pairs)
    fn = sum(x < hi and y >= hi for x, y in pairs)
    return {"n": n, "mean_cand": sum(x for x, _ in pairs) / n, "mean_ref": sum(y for _, y in pairs) / n,
            "mean_diff_cand_minus_ref": sum(x - y for x, y in pairs) / n,
            "exact": sum(x == y for x, y in pairs) / n, "within1": sum(abs(x - y) <= 1 for x, y in pairs) / n,
            "qwk": qwk([(y, x) for x, y in pairs]),
            "precision_hi_vs_ref": tp / (tp + fp) if tp + fp else None,
            "recall_hi_vs_ref": tp / (tp + fn) if tp + fn else None,
            "hi_both": tp, "hi_cand_only": fp, "hi_ref_only": fn, "lo_both": n - tp - fp - fn}


# ---------------------------------------------------------------------------------------------------- synthetic demo
def synthetic_items(n_problems=60, per_problem=3, seed=3):
    """Synthetic expert-graded proofs with a bimodal grade distribution like public proof-grading sets."""
    rng = random.Random(seed)
    gens = ["gen_A", "gen_B", "gen_C"]
    weights = [157, 62, 62, 31, 20, 9, 16, 78]  # shape only: bimodal, many 0s and 7s
    rows = []
    for p in range(n_problems):
        diff = rng.random()
        for k in range(per_problem):
            g = rng.choices(range(8), weights=[w * (1.5 - diff if i < 4 else 0.5 + diff) for i, w in enumerate(weights)])[0]
            rows.append({"problem_id": "P%03d" % p, "response_id": k, "grade": g, "generator": gens[k % 3],
                         "year": 2020 + (p % 6), "proof": "x" * rng.randint(300, 4000)})
    return rows


def synthetic_grader(human, seed=0, harsh_top=1.2, generous_bottom=0.6, noise=1.0):
    """A synthetic grader with the error pattern found on real data: harsh on strong proofs, generous on weak ones."""
    rng = random.Random(seed)
    out = {}
    for i, h in human.items():
        shift = -harsh_top if h >= 6 else (generous_bottom if h <= 2 else -0.9)
        out[i] = max(0, min(7, int(round(h + shift + rng.gauss(0, noise)))))
    return out


def _fmt(x, nd=3):
    return "n/a" if x is None else "%.*f" % (nd, x)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grader marks vs human marks (aggregates only).")
    ap.add_argument("items", help="JSONL: id, human (0..7), generator")
    ap.add_argument("grades", help="JSONL: id, score (0..7 or null); last row per id wins")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    items = {}
    for ln in io.open(a.items, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            items[r["id"]] = {"human": int(r["human"]), "generator": r.get("generator", "-")}
    grades = {}
    for ln in io.open(a.grades, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            grades[r["id"]] = r.get("score")
    res = analyze(items, grades)
    o = res["overall"]
    print("n scored %d / %d (unmarked: %s)" % (res["n_scored"], res["n_items"], res["unmarked"] or "none"))
    if o["n"]:
        ci = res["qwk_boot95"]
        print("QWK %s [%s]  bias %+.2f  MAE %.2f  exact %.0f%%  within-1 %.0f%%" % (
            _fmt(o["qwk"]), "n/a" if not ci else "%.2f, %.2f" % tuple(ci), o["mean_bias_grader_minus_human"], o["mae"],
            100 * o["exact"], 100 * o["within1"]))
        for b, s in res["by_band"].items():
            if s["n"]:
                print("  %-10s n=%2d QWK %s bias %+.2f" % (b, s["n"], _fmt(s["qwk"]), s["mean_bias_grader_minus_human"]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
