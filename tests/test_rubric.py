import pytest

from judgelab import rubric as R


def test_rescale_six_to_seven():
    assert [R.rescale(s, 6, 7) for s in range(7)] == [0, 1, 2, 4, 5, 6, 7]
    assert [R.rescale(m, 7, 6) for m in range(8)] == [0, 1, 2, 3, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        R.rescale(7, 6, 7)


def test_max_over_paths_vs_additive_chain():
    official = [{"A1": 3, "A2": 3}, {"B1": 2, "B2": 2, "B3": 2}]   # two alternative complete routes, max 6
    ai_chain = {"A1": 3, "A2": 3, "bare_answer": 0, "final": 1}    # one route written as a 7-point chain
    solvers = {"route_A": {"A1", "A2"}, "route_B": {"B1", "B2", "B3"}, "half_A": {"A1"}}
    out = R.audit_scheme(official, ai_chain, 6, 7, solvers)
    assert out["route_A"]["official_mapped"] == 7 and out["route_A"]["ai_chain"] == 6
    # a complete solution on the second path gets 7 officially but 0 on the AI chain
    assert out["route_B"] == {"official_mapped": 7, "ai_chain": 0, "ai_minus_official": -7}
    assert out["half_A"]["official_mapped"] == 4 and out["half_A"]["ai_chain"] == 3


def test_caps():
    assert R.additive_score({"a": 5, "b": 5}, {"a", "b"}, cap=7) == 7
    assert R.max_over_paths([], {"a"}) == 0
