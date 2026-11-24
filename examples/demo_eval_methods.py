"""Demo of the evaluation methodology on SYNTHETIC out-of-fold predictions (no real data).

    python examples/demo_eval_methods.py [--boot 500]

Prints: macro AUC of a baseline and a stronger arm, the paired bootstrap interval of the difference, a raw vs
cross-fitted calibration table with the ECE null band, and the predictive band for one future 120-item read.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judgelab import evalstats as E  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args(argv)
    d = E.synthetic_oof(n=a.n)
    L = d["labels"]
    m_base, per_base = E.macro_auc(d["base"], L)
    m_arm, per_arm = E.macro_auc(d["arm"], L)
    print("SYNTHETIC DATA. macro AUC base %.4f  arm %.4f  diff %+.4f" % (m_base, m_arm, m_arm - m_base))
    bs = E.paired_bootstrap({"arm": d["arm"], "base": d["base"]}, L, n_boot=a.boot)
    s = E.summarize(bs["arm"] - bs["base"])
    print("paired bootstrap (%d item resamples): diff 95%% [%+.4f, %+.4f]" % (a.boot, s["lo95"], s["hi95"]))
    print("\n%-8s %6s %9s %9s %11s %11s %9s %s" % ("route", "n", "ECE raw", "null p95", "ECE cf", "null p95 cf", "slope a",
                                                    "verdict"))
    for r in E.ROUTES:
        keep = ~np.isnan(L[r])
        rep = E.calibration_report(d["arm"][r][keep], L[r][keep], d["folds"][keep], nsim=300)
        print("%-8s %6d %9.4f %9.4f %11.4f %11.4f %9.3f %s" % (
            r, rep["n"], rep["ece_raw"], rep["ece_raw_null_p95"], rep["ece_crossfit"], rep["ece_crossfit_null_p95"],
            rep["platt_full"][0], rep["verdict"]))
    band = E.predictive_band(d["arm"], d["base"], L, k=120, n_sim=a.boot)
    print("\npredictive band for one 120-item read: %+.4f to %+.4f (point on all items %+.4f)" % (
        band["lo95"], band["hi95"], band["point_full"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
