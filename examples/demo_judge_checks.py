"""Demo of the judge-reliability checks on SYNTHETIC inputs (no real data, no model calls).

    python examples/demo_judge_checks.py

1. Known-answer swapped-order validity: builds the 30-pair / 60-order study on synthetic texts and scores three
   synthetic judges (an oracle, a position-biased judge, a length-biased judge).
2. Grader calibration: a synthetic grader that is harsh on strong proofs, scored against synthetic human marks.
3. Canary control: two synthetic reviewer lanes, one competent and one that misses planted flaws.
4. Garble detector on a few hand-written statements.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from judgelab import calibration as C  # noqa: E402
from judgelab import canary as K  # noqa: E402
from judgelab import garble as G  # noqa: E402
from judgelab import judge_validity as J  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")   # non-UTF-8 consoles
    m, orders = J.build_study()
    print("1. judge validity: %d pairs, %d orders, leakage problems: %d" % (len(m["pairs"]), len(orders),
                                                                         len(J.check_leakage(m, orders))))
    for name, v in (("oracle", J.oracle_judge(m, 3)), ("position", J.position_judge(orders)),
                    ("length", J.length_judge(orders))):
        s = J.summarize(J.evaluate(m, v))
        print("   %-9s %-11s %s" % (name, s["verdict"], " ".join("%s=%s" % (k, c["value"]) for k, c in s["checks"].items())))

    rows = C.synthetic_items(n_problems=120)
    human = {"%s/%d" % (r["problem_id"], r["response_id"]): r["grade"] for r in rows}
    res = C.analyze({i: {"human": g, "generator": "-"} for i, g in human.items()}, C.synthetic_grader(human), B=500)
    o = res["overall"]
    print("2. calibration (synthetic): QWK %.2f bias %+.2f;" % (o["qwk"], o["mean_bias_grader_minus_human"]),
          "; ".join("%s QWK %.2f bias %+.2f" % (b, s["qwk"] or 0, s["mean_bias_grader_minus_human"])
                    for b, s in res["by_band"].items()))
    print("   copy-the-siblings baseline MAE %.2f vs grader MAE %.2f" % (C.copy_baseline(rows)["mae"], o["mae"]))

    bank = [{"kid": "I%d" % i, "valid": False} for i in range(8)] + [{"kid": "V%d" % i, "valid": True} for i in range(4)]
    rec = K.ControlRecord(K.simulate_lane(bank, 1.0, 0.9, lane="good", seed=1) + K.simulate_lane(bank, 0.4, 1.0, lane="weak", seed=2))
    for lane in ("good", "weak"):
        print("3. lane %-5s stats %s disqualified %s safety(b0) %s" % (
            lane, rec.lane_stats(lane, "blind"), rec.disqualified(lane, "blind"), rec.g5(lane, "blind", "b0")[0]))

    for t in ("Let $n \\ge 2$. Prove that $2^n > n$.", "Show that the sum is at least n ` 1.", "Prove that x ≤ 2x − 1 for x ≥ 1."):
        s = G.signals(t)
        print("4. %-45s GARBLE=%s RISK=%s" % (t[:45], s["GARBLE"], s["RISK"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
