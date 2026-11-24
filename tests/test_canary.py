"""Canary / planted-flaw control rules (synthetic events)."""
import random

from judgelab import canary as K


def ev(lane, batch, kid, valid, passed, stage="blind"):
    return {"lane": lane, "stage": stage, "batch": batch, "kid": kid, "valid": valid, "passed": passed}


def good_record(lane="L", n_inv=6, n_val=3, batch="b1"):
    return [ev(lane, batch, "I%d" % i, False, True) for i in range(n_inv)] + \
           [ev(lane, batch, "V%d" % i, True, True) for i in range(n_val)]


def test_safe_and_strict_lane():
    r = K.ControlRecord(good_record())
    assert r.lane_stats("L", "blind") == (6, 6, 3, 3)
    assert r.g5("L", "blind", "b1") == (True, True)
    assert r.review_counts({"lane": "L", "stage": "blind", "batch": "b1"})


def test_needs_planted_flaw_in_the_same_batch():
    r = K.ControlRecord(good_record())
    r.add(ev("L", "b2", "V0", True, True))                       # batch b2 had only a valid control
    assert r.g5("L", "blind", "b2")[0] is False


def test_one_miss_voids_the_batch_and_blocks_strict():
    events = good_record() + [ev("L", "b2", "I%d" % i, False, True) for i in range(6, 15)]
    events.append(ev("L", "b2", "I15", False, False))            # one miss in batch b2
    r = K.ControlRecord(events)
    assert r.batch_void("L", "blind", "b2")
    assert r.g5("L", "blind", "b2") == (False, False)
    safety, strict = r.g5("L", "blind", "b1")                   # 15/16 caught overall (>= 90%): b1 stays safe
    assert safety and not strict


def test_repeated_control_is_not_new_evidence():
    events = [ev("L", "b%d" % b, "I0", False, True) for b in range(10)]
    r = K.ControlRecord(events)
    assert r.lane_stats("L", "blind")[:2] == (1, 1)
    assert r.g5("L", "blind", "b0")[0] is False                  # 1 distinct planted flaw < MIN_INV


def test_caught_only_if_caught_every_time():
    events = [ev("L", "b1", "I0", False, True), ev("L", "b2", "I0", False, False)]
    assert K.ControlRecord(events).lane_stats("L", "blind")[:2] == (1, 0)


def test_disqualified_and_low_yield():
    bank = [{"kid": "I%d" % i, "valid": False} for i in range(6)] + [{"kid": "V%d" % i, "valid": True} for i in range(4)]
    weak = K.ControlRecord(K.simulate_lane(bank, catch_rate=0.0, accept_rate=1.0, lane="W"))
    assert weak.disqualified("W", "blind")
    picky = K.ControlRecord(K.simulate_lane(bank, catch_rate=1.0, accept_rate=0.0, lane="P"))
    assert picky.low_yield("P", "blind") and not picky.disqualified("P", "blind")
    # the reported real-run pattern: 2 of 6 and 3 of 6 planted flaws caught -> both disqualified
    for caught in (2, 3):
        evs = [ev("X", "b", "I%d" % i, False, i < caught) for i in range(6)]
        assert K.ControlRecord(evs).disqualified("X", "blind")


def test_flag_aware_catch():
    assert K.flag_aware_caught(True, 7, gate=4)
    assert K.flag_aware_caught(False, 3, gate=4)
    assert not K.flag_aware_caught(False, 6, gate=4)
    assert not K.flag_aware_caught(False, None, gate=4)


def test_truncated_cuts_at_boundary():
    proof = "Step one holds. " * 20 + "\n\n" + "Step two follows. " * 20 + "\n\nHence the claim."
    t = K.truncated(proof)
    assert 0.3 * len(proof) < len(t) <= 0.55 * len(proof)
    assert "Hence the claim" not in t and t.endswith(".")


def test_pick_canaries_counts_and_least_seen_first():
    bank = K.synthetic_bank()
    r = K.ControlRecord([ev("L", "b0", "K01", False, True)])
    real = [{"cid": "c%d" % i, "proof": "x. " * 400} for i in range(12)]
    jobs = K.pick_canaries(r, bank, "L", "blind", real, random.Random(0))
    planted = [j for j in jobs if not j["kid"].startswith("T")]
    assert len(planted) == 2                                     # 12 real reviews -> 2 controls (1 invalid, 1 valid)
    assert sum(not j["valid"] for j in planted) == 1
    assert planted[0]["kid"] != "K01"                            # least-seen invalid control first
    assert any(j["kid"].startswith("T") and j["valid"] is False for j in jobs)   # plus one truncation control
    assert K.pick_canaries(r, bank, "L", "blind", [], qualify=(2, 1)).__len__() == 3


def test_parse_second_audit():
    ok = "...\nFLAW_REAL: yes | the bound is used before it is proved\nREPAIRED_ELSEWHERE: no | -\nPROOF_STILL_COMPLETE: no\nVERDICT: REAL-FATAL"
    assert K.parse_second_audit(ok)["verdict"] == "REAL-FATAL"
    contradictory = ok.replace("PROOF_STILL_COMPLETE: no", "PROOF_STILL_COMPLETE: yes")
    assert K.parse_second_audit(contradictory)["verdict"] == "UNSURE"
    assert K.parse_second_audit("VERDICT: REAL-FATAL") is None
