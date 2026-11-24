"""CPU tests of judgelab.verdicts on a scripted fake Broker. Each greedy case is compared with a direct re-implementation of the reference
server's greedy rules; reads are checked against an independent top-k of the same log-softmax row."""
import random
from types import SimpleNamespace

import pytest

from judgelab import protocol
from judgelab import verdicts as V

VOCAB = {0: "x", 1: "y", 2: '"', 3: ",", 4: ' "severity"', 5: "}", 6: "\n", 7: "ab", 8: "a", 9: "q}r", 10: "<|im_end|>", 11: "b"}
EOS = {10}
P = protocol.current()
CTX = P.CTX


class Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCAB[i] for i in ids)


class Broker:
    """Scripted broker: greedy = vLLM semantics (stop at first stop_token_id INCLUDED, else max_tokens); reads = fixed row."""
    def __init__(self, script=None, row=None, override=None):
        self.tok, self.eos_ids, self.calls = Tok(), set(EOS), []
        self.script, self.row, self.override = script or [], row, override

    def submit(self, prompt_ids, req, lora=None):
        self.calls.append((list(prompt_ids), dict(req), lora))
        if self.override:
            return self.override
        if req["logprobs"]:
            return SimpleNamespace(token_ids=[0], finish_reason="length", stop_reason=None, logprobs=[self.row], prompt_n=7)
        out = (self.script + [0] * req["max_tokens"])[:req["max_tokens"]]
        cut = next((i + 1 for i, t in enumerate(out) if t in req["stop_token_ids"]), 0)
        out, finish = (out[:cut], "stop") if cut else (out, "length")
        return SimpleNamespace(token_ids=out, finish_reason=finish, stop_reason=None, logprobs=None, prompt_n=7)


def ref_greedy(prompt, script, stops, limit):
    """Direct re-implementation of the reference greedy rules on a scripted argmax sequence."""
    room = min(limit, CTX - len(prompt))
    result, tokens, content = {"timings": {"prompt_n": 7}}, [], ""
    for token in (script + [0] * room)[:room]:
        if token in EOS:
            return dict(result, content=content, tokens=tokens, stopped_eos=True)
        tokens.append(token)
        content = Tok().decode(tokens)
        matches = [(content.index(s), i, s) for i, s in enumerate(stops) if s in content]
        if matches:
            at, _, s = min(matches)
            return dict(result, content=content[:at], tokens=tokens, stopped_word=True, stopping_word=s)
    return dict(result, content=content, tokens=tokens, stopped_limit=True)


def gbody(prompt, n, stops, seed=11, **bad):
    return dict(P.greedy_body(seed, n, stops), prompt=prompt, cache_prompt=True, id_slot=0, return_tokens=True, stream=False, **bad)


def rbody(prompt, **bad):
    return dict(P.read_body(), prompt=prompt, cache_prompt=True, id_slot=0, return_tokens=True, stream=False, **bad)


def res(ids, finish="length"):
    return SimpleNamespace(token_ids=ids, finish_reason=finish, stop_reason=None, logprobs=None, prompt_n=1)


def refused(fn, *args):
    try:
        fn(*args)
    except (ValueError, RuntimeError):
        return True
    return False


GREEDY_CASES = [  # (name, script, stops, n_predict, prompt_len)
    ("first_stop_by_position", [0, 1, 9, 6, 2], ["\n", "}"], 384, 5),     # '}' (listed 2nd) cuts mid-token 'q}r'
    ("same_position_lower_index", [0, 7, 6], ["ab", "a"], 384, 5),         # both at 1 -> index 0 wins
    ("incremental_short_first", [0, 8, 11], ["ab", "a"], 384, 5),          # 'a' seen before 'ab' completes
    ("multi_token_stop", [0, 2, 4, 1], ['"' + ' "severity"'], 384, 5),
    ("eos", [0, 1, 10, 6], ["\n"], 384, 5),
    ("eos_first", [10], ["\n"], 384, 5),
    ("limit_n_predict", [0, 1, 0, 1, 0, 1], ["\n"], 4, 5),
    ("limit_ctx_room", [0, 1, 0, 1, 0, 1, 0, 1], ["\n"], 384, CTX - 3),    # room = 3
    ("stop_on_last_room_token", [0, 1, 6], ["\n"], 3, 5),
]


@pytest.mark.parametrize("name,script,stops,n,plen", GREEDY_CASES, ids=[c[0] for c in GREEDY_CASES])
def test_greedy_equal(name, script, stops, n, plen):
    prompt = [1] * plen
    b = Broker(script)
    got = V.complete_greedy(b, gbody(prompt, n, stops), "adapterA")
    assert got == ref_greedy(prompt, script, stops, n)
    room = min(n, CTX - plen)
    _, req, lora = b.calls[0]
    assert (req["max_tokens"], req["temperature"], req["stop"], req["stop_token_ids"], req["seed"], lora) == \
        (room, 0.0, [], sorted(EOS), 11, "adapterA")
    assert V.complete(Broker(script), gbody(prompt, n, stops), None) == got


def test_read_top100_and_selection():
    torch = pytest.importorskip("torch")
    for seed in range(12):
        g = torch.Generator().manual_seed(seed)
        lsm = torch.log_softmax(torch.randn(400, generator=g) * 3, dim=-1)
        values, ids = torch.topk(lsm, 150)
        pairs = list(zip(ids.tolist()[:100], values.tolist()[:100])) + [(ids[140].item(), values[140].item())]  # + a sampled id
        random.Random(seed).shuffle(pairs)
        b = Broker(row=dict(pairs))
        got = V.complete_read(b, rbody([3, 4, 5]), "adapterB")
        entries = got["completion_probabilities"][0]["top_logprobs"]
        assert [e["id"] for e in entries] == ids.tolist()[:100]            # the 100 largest, sorted desc; the extra id dropped
        assert got["completion_probabilities"][0]["id"] in ids.tolist()[:100]
        assert got == V.complete_read(Broker(row=dict(pairs)), rbody([3, 4, 5]), "adapterB")   # seeded selection is deterministic
        _, req, lora = b.calls[0]
        assert (req["max_tokens"], req["temperature"], req["logprobs"], req["seed"], lora) == (1, 1.0, 100, 0, "adapterB")


def test_read_refusals():
    torch = pytest.importorskip("torch")  # noqa: F841
    row = {i: -1.0 - i / 1000 for i in range(100)}
    assert not refused(V.complete_read, Broker(row=row), rbody([1]), None), "positive control"
    assert refused(V.complete_read, Broker(row=dict(list(row.items())[:99])), rbody([1]), None), "99 entries"
    for bad in (float("nan"), float("-inf"), float("inf")):
        assert refused(V.complete_read, Broker(row={**row, 5: bad}), rbody([1]), None), "nonfinite"
    for key, val in (("temperature", 0.7), ("top_k", 20), ("n_probs", 50), ("seed", 5), ("presence_penalty", 1.5),
                     ("n_predict", 2), ("repeat_last_n", 64)):
        assert refused(V.complete_read, Broker(row=row), dict(rbody([1]), **{key: val}), None), key
    assert refused(V.complete_read, Broker(row=row), {k: v for k, v in rbody([1]).items() if k != "min_p"}, None), "missing"
    assert refused(V.complete_read, Broker(override=res([0])), rbody([1]), None), "no logprobs"
    assert refused(V.complete_read, Broker(row=row), rbody([]), None), "empty prompt"
    assert refused(V.complete_read, Broker(row=row), rbody([1] * (CTX + 1)), None), "prompt over CTX"


def test_greedy_refusals():
    ok = gbody([1, 2], 8, ["\n"])
    for key, val in (("presence_penalty", 1.5), ("top_k", 20), ("temperature", 1.0), ("top_p", 0.95), ("repeat_last_n", 64)):
        assert refused(V.complete_greedy, Broker([6]), dict(ok, **{key: val}), None), key
    assert refused(V.complete_greedy, Broker([6]), gbody([1] * CTX, 8, ["\n"]), None), "room 0"
    assert refused(V.complete_greedy, Broker([6]), gbody([1, 2], 0, ["\n"]), None), "n_predict 0"
    assert not refused(V.complete_greedy, Broker([6]), gbody([1] * (CTX - 1), 8, ["\n"]), None), "room 1 accepted"
    for bad, why in ((res([0, 1], "stop"), "early stop without eos/word"), (res([0] * 9), "more than room"),
                     (res([6], "abort"), "abort")):
        assert refused(V.complete_greedy, Broker(override=bad), ok, None), why


def test_negative_controls():
    """The equality checks must be able to fail: a wrong stop rule (first-LISTED stop only) differs from the reference."""
    _, script, stops, n, plen = GREEDY_CASES[0]
    got = V.complete_greedy(Broker(script), gbody([1] * plen, n, stops), None)
    assert got != ref_greedy([1] * plen, script, stops[:1], n)
    assert got != ref_greedy([1] * plen, script, stops, 2)  # wrong room -> limit instead of word
    assert got["content"] == "xyq" and got["stopping_word"] == "}"
    eos = V.complete_greedy(Broker([0, 1, 10]), gbody([1], 9, ["\n"]), None)
    assert eos["tokens"] == [0, 1] and 10 not in eos["tokens"] and eos["stopped_eos"] is True
