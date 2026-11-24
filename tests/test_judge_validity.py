"""Known-answer swapped-order judge validity on synthetic texts."""
import collections

import pytest

from judgelab import judge_validity as J


@pytest.fixture(scope="module")
def study():
    return J.build_study()


def test_structure(study):
    m, orders = study
    assert len(m["pairs"]) == 30 and len(orders) == 60
    assert collections.Counter(p["defect_type"] for p in m["pairs"]) == {t: 6 for t in J.ALL_TYPES}
    by_pair = collections.defaultdict(list)
    for o in m["orders"]:
        by_pair[o["pair_id"]].append(o["shown_a"])
    assert all(sorted(v) == ["intact", "variant"] for v in by_pair.values())   # each pair in both orders


def test_build_is_deterministic(study):
    m2, o2 = J.build_study()
    assert [o["prompt_sha256"] for o in study[1]] == [o["prompt_sha256"] for o in o2]


def test_edits_do_what_they_say(study):
    m, _ = study
    for p in m["pairs"]:
        t, a, b = p["defect_type"], p["intact"], p["variant"]
        if t == "e":
            assert a == b
        elif t == "a":
            assert p["edit"]["removed"] in a and p["edit"]["removed"] not in b and len(b) < len(a)
        elif t == "b":
            assert len(a) == len(b) and a.count("\\geq") + 1 == b.count("\\geq")
        elif t == "c":
            assert sorted(a) == sorted(b) and a != b
        elif t == "d":
            assert 5 <= p["edit"]["percent_added"] <= 12 and len(b) > len(a)


def test_no_leakage(study):
    assert J.check_leakage(*study) == []


def test_leakage_check_catches_a_leak(study):
    m, orders = study
    bad = [dict(o) for o in orders]
    bad[0]["prompt"] = bad[0]["prompt"].replace("=== END ===", "(this is the intact copy) === END ===")
    bad[1]["defect_type"] = "a"
    problems = J.check_leakage(m, bad)
    assert any("role word" in w for _, w in problems) and any("key fields" in w for _, w in problems)


def test_oracle_usable(study):
    m, _ = study
    for strength in (3, 4):
        s = J.summarize(J.evaluate(m, J.oracle_judge(m, strength)))
        assert s["verdict"] == "USABLE" and s["checks"]["R1"]["value"] == 24


def test_position_biased_judge_fails(study):
    m, orders = study
    s = J.summarize(J.evaluate(m, J.position_judge(orders)))
    assert s["verdict"] == "NOT USABLE" and s["checks"]["R5"]["value"] == 24
    assert J.position_bias(J.position_judge(orders))["share_a_among_decided"] == 1.0


def test_length_biased_judge_fails_on_padding(study):
    m, orders = study
    s = J.summarize(J.evaluate(m, J.length_judge(orders)))
    assert s["checks"]["R3"]["value"] == 0 and not s["checks"]["R3"]["passed"]
    assert s["checks"]["R2"]["value"] == 6 and s["verdict"] == "NOT USABLE"


def test_missing_order_is_incomplete(study):
    m, _ = study
    v = J.oracle_judge(m)
    v.pop("O01")
    assert J.summarize(J.evaluate(m, v))["verdict"] == "INCOMPLETE"


def test_threshold_edge(study):
    """R1 exactly at the threshold (20 of 24) still passes; 19 fails."""
    m, _ = study
    v = J.oracle_judge(m)
    defect = [p["pair_id"] for p in m["pairs"] if p["defect_type"] in ("a", "b", "c")]
    order_of = collections.defaultdict(list)
    for o in m["orders"]:
        order_of[o["pair_id"]].append(o["order_id"])

    def tie(pids):
        w = dict(v)
        for pid in pids:
            for oid in order_of[pid]:
                w[oid] = 2
        return J.summarize(J.evaluate(m, w))
    assert tie(defect[:4])["verdict"] == "USABLE"
    assert tie(defect[:5])["verdict"] == "NOT USABLE"


def test_favour_class_table():
    assert J.favour_class(4, 3) == "AF" and J.favour_class(3, 2) == "LF" and J.favour_class(2, 2) == "T"
    assert J.favour_class(1, 2) == "LS" and J.favour_class(0, 1) == "AS" and J.favour_class(4, 0) == "C"


def test_parse_verdict():
    assert J.parse_verdict('{"overall": 3}') == 3
    assert J.parse_verdict('```json\n{"overall": 0}\n```') == 0
    assert J.parse_verdict('{"overall": 5}') is None
    assert J.parse_verdict('{"overall": true}') is None
    assert J.parse_verdict("A is better") is None
