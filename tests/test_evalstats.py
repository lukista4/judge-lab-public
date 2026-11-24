"""Evaluation-methodology functions on synthetic data (numpy only, CPU)."""
import math

import numpy as np
import pytest

from judgelab import evalstats as E


def test_auc_known_values_and_ties():
    assert E.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert E.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert E.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5       # all tied -> one half
    assert math.isnan(E.auc([0.1, 0.2], [1, 1]))                     # one class absent
    assert E.auc([0.1, 0.9, 0.3], [0, 1, np.nan]) == 1.0            # NaN label dropped


def test_auc_matches_brute_force():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 6, 80).astype(float)                         # many ties
    y = (rng.random(80) < 0.4).astype(float)
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert E.auc(s, y) == pytest.approx(brute, abs=1e-12)


def test_weighted_auc_equals_materialised_resample():
    rng = np.random.default_rng(2)
    s = rng.integers(0, 8, 60).astype(float)
    y = (rng.random(60) < 0.5).astype(float)
    idx = rng.integers(0, 60, 60)
    w = np.bincount(idx, minlength=60)
    assert E._weighted_auc(s, y, w) == pytest.approx(E.auc(s[idx], y[idx]), abs=1e-12)
    assert E._weighted_auc(s, y, np.ones(60, int)) == pytest.approx(E.auc(s, y), abs=1e-12)


def test_paired_bootstrap_detects_real_gain_and_null():
    d = E.synthetic_oof(n=400, seed=5)
    bs = E.paired_bootstrap({"arm": d["arm"], "base": d["base"]}, d["labels"], n_boot=150, seed=1)
    diff = E.summarize(bs["arm"] - bs["base"])
    assert diff["lo95"] > 0                                           # the stronger synthetic arm wins
    same = E.paired_bootstrap({"x": d["base"], "y": d["base"]}, d["labels"], n_boot=50, seed=1)
    assert np.all(same["x"] - same["y"] == 0)                         # paired: identical arms give exactly 0


def test_platt_recovers_slope():
    rng = np.random.default_rng(3)
    z_true = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < E.sigmoid(z_true)).astype(float)
    a, b = E.platt_fit(3.0 * z_true, y)                               # scores over-confident by 3x
    assert a == pytest.approx(1 / 3, abs=0.04) and abs(b) < 0.15


def test_crossfit_is_out_of_fold():
    rng = np.random.default_rng(4)
    z = rng.normal(0, 3, 500)
    y = (rng.random(500) < E.sigmoid(z / 3)).astype(float)
    folds = np.arange(500) % 5 + 1
    zc, fits = E.crossfit_platt(z, y, folds)
    for f, (a, b) in fits.items():
        tr = folds != f
        assert (a, b) == pytest.approx(E.platt_fit(z[tr], y[tr]))
        te = folds == f
        assert np.allclose(zc[te], a * z[te] + b)


def test_ece_null_flags_overconfidence_only():
    rng = np.random.default_rng(6)
    z = rng.normal(0, 1.5, 3000)
    y = (rng.random(3000) < E.sigmoid(z)).astype(float)
    folds = np.arange(3000) % 5 + 1
    good = E.calibration_report(z, y, folds, nsim=200)
    bad = E.calibration_report(4 * z, y, folds, nsim=200)
    assert good["verdict"] == "OK"
    assert bad["verdict"] == "RECALIBRATE"
    assert bad["ece_crossfit"] < bad["ece_raw"] and bad["logloss_crossfit"] < bad["logloss_raw"]


def test_wilson():
    lo, hi = E.wilson(81, 85)
    assert 0.88 < lo < 0.89 and 0.98 < hi < 0.99
    assert E.wilson(0, 0) == (None, None)


def test_predictive_band_wider_than_point_and_centred():
    d = E.synthetic_oof(n=400, seed=8)
    band = E.predictive_band(d["arm"], d["base"], d["labels"], k=80, n_sim=150, seed=2)
    assert band["lo95"] < band["point_full"] < band["hi95"]
    assert abs(band["mean"] - band["point_full"]) < 0.02
    assert band["hi95"] - band["lo95"] > 0.02
