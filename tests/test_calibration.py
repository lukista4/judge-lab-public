"""Grader calibration statistics and selection on synthetic data."""
import json
import subprocess
import sys

import pytest

from judgelab import calibration as C
from conftest import subprocess_env


def test_qwk_perfect_and_chance():
    assert C.qwk([(g, g) for g in range(8)]) == pytest.approx(1.0)
    assert C.qwk([(3, 3), (3, 3)]) is None                       # one grade only: undefined, not 1.0
    assert C.qwk([(0, 7), (7, 0), (0, 7), (7, 0)]) < 0


def test_qwk_matches_reference_formula():
    # hand-computed 3-item example on the 0..7 scale
    pairs = [(0, 1), (7, 6), (3, 3)]
    k = 8
    O = [[0] * k for _ in range(k)]
    for h, c in pairs:
        O[h][c] += 1
    rh = [sum(r) for r in O]
    rc = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = sum((i - j) ** 2 * O[i][j] for i in range(k) for j in range(k))
    den = sum((i - j) ** 2 * rh[i] * rc[j] / 3 for i in range(k) for j in range(k))
    assert C.qwk(pairs) == pytest.approx(1 - num / den)


def test_analyze_counts_unmarked_and_bands():
    items = {"a": {"human": 7, "generator": "g1"}, "b": {"human": 0, "generator": "g1"},
             "c": {"human": 4, "generator": "g2"}, "d": {"human": 6, "generator": "g2"}}
    res = C.analyze(items, {"a": 5, "b": 1, "c": None, "d": 9, "zz": 3}, B=200)
    assert res["n_scored"] == 2 and res["unmarked"] == ["c", "d"] and res["unknown_ids"] == ["zz"]
    assert res["by_band"]["high(6-7)"]["mean_bias_grader_minus_human"] == -2
    assert res["by_band"]["low(0-2)"]["mean_bias_grader_minus_human"] == 1
    assert sum(map(sum, res["confusion_human_rows_grader_cols"])) == 2


def test_harsh_top_pattern_shows_in_band_qwk():
    items = C.synthetic_items(n_problems=150, seed=11)
    human = {"%s/%d" % (r["problem_id"], r["response_id"]): r["grade"] for r in items}
    meta = {i: {"human": g, "generator": "x"} for i, g in human.items()}
    res = C.analyze(meta, C.synthetic_grader(human, seed=1), B=200)
    assert res["overall"]["qwk"] > 0.6                           # good overall agreement ...
    assert res["by_band"]["high(6-7)"]["mean_bias_grader_minus_human"] < -0.8   # ... from separating bands,
    assert res["by_band"]["low(0-2)"]["mean_bias_grader_minus_human"] > 0.3     # harsh at the top, generous below


def test_select_stratified_rules():
    rows = C.synthetic_items(n_problems=80, seed=4)
    quota = {0: 4, 1: 3, 2: 3, 3: 3, 6: 2, 7: 5}
    a, rep = C.select_stratified(rows, quota, seed="s1", exclude_generators=("gen_A",), max_year=2023)
    b, _ = C.select_stratified(rows, quota, seed="s1", exclude_generators=("gen_A",), max_year=2023)
    assert a == b                                                # deterministic
    assert all(r["generator"] != "gen_A" and r["year"] <= 2023 and 600 <= len(r["proof"]) <= 30000 for r in a)
    got = {g: sum(r["grade"] == g for r in a) for g in quota}
    for g in quota:   # every cell is filled, or reported short with the count actually found
        assert got[g] == (rep["short_cells"][g][0] if g in rep["short_cells"] else quota[g])
    assert [r["id"] for r in a] == ["cal_%02d" % i for i in range(1, len(a) + 1)]
    assert [r["grade"] for r in a] != sorted(r["grade"] for r in a)   # order carries no grade information


def test_copy_baseline():
    rows = [{"problem_id": "p", "grade": g} for g in (7, 7, 0)] + [{"problem_id": "q", "grade": 3}]
    s = C.copy_baseline(rows)
    assert s["n"] == 3                                           # q has no sibling
    # predictions: round(mean(7,0))=4 for both 7s, round(mean(7,7))=7 for the 0
    assert s["mae"] == pytest.approx((3 + 3 + 7) / 3)


def test_paired_agreement():
    ref = {i: g for i, g in enumerate([0, 2, 6, 7, 7, 3])}
    cand = {i: min(7, g + 1) for i, g in ref.items()}
    s = C.paired_agreement(cand, ref)
    assert s["n"] == 6 and s["mean_diff_cand_minus_ref"] == pytest.approx(4 / 6)
    assert s["hi_both"] == 3 and s["hi_cand_only"] == 0 and s["recall_hi_vs_ref"] == 1.0


def test_cli(tmp_path):
    items = tmp_path / "items.jsonl"
    grades = tmp_path / "grades.jsonl"
    items.write_text("\n".join(json.dumps({"id": i, "human": h, "generator": "g"}) for i, h in
                               [("a", 0), ("b", 3), ("c", 7), ("d", 6)]), encoding="utf-8")
    grades.write_text("\n".join(json.dumps({"id": i, "score": s}) for i, s in
                                [("a", 1), ("b", 2), ("c", 7), ("d", None)]), encoding="utf-8")
    out = tmp_path / "res.json"
    r = subprocess.run([sys.executable, "-m", "judgelab.calibration", str(items), str(grades), "--json", str(out)],
                       capture_output=True, text=True, env=subprocess_env())
    assert r.returncode == 0, r.stderr
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n_scored"] == 3 and res["unmarked"] == ["d"]
