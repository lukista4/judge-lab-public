"""Broker + FakeEngine (DELTA chunking, stop ids, abort, logprobs, admission, refusals) and VLLMProxy.stream_think stop rules."""
import threading
from types import SimpleNamespace

import pytest

import fakes as fx
from judgelab import engine as E
from judgelab.proxy import VLLMProxy


def broker(script, ctx=fx.PROTO.CTX, kv=1 << 22):
    return E.Broker(E.FakeEngine(script, ctx=ctx), kv, fake=True, tok=fx.MiniTok(), eos_ids=fx.EOS_IDS, ctx=ctx)


def req(**kw):
    r = dict(max_tokens=50, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=1, stop_token_ids=[], stop=[], logprobs=None,
             presence_penalty=0.0, penalty_scope="output")
    r.update(kw)
    return r


def test_submit_stop_id_included_and_chunked():
    eng = E.FakeEngine(lambda p, r, l: (list(range(1, 21)) + [99, 5, 6], None), ctx=4096)
    b = E.Broker(eng, 1 << 20, fake=True, ctx=4096)
    res = b.submit([1, 2, 3], req(stop_token_ids=[99]))
    assert res.token_ids == list(range(1, 21)) + [99] and res.finish_reason == "stop" and res.stop_reason == 99
    assert b.stats["steps"] >= 3   # 21 tokens at <= 7 per step
    assert res.prompt_n == 3


def test_length_and_logprobs():
    row = {t: -0.01 * t for t in range(1, 101)}
    b = broker(lambda p, r, l: ([7] * 10, [row] * 10))
    res = b.submit([1], req(max_tokens=4, logprobs=100))
    assert res.token_ids == [7] * 4 and res.finish_reason == "length" and len(res.logprobs) == 4 and res.logprobs[0] == row


def test_stream_abort_and_concurrency():
    b = broker(lambda p, r, l: (list(range(1, 200)), None))
    seen = []

    def cb(new):
        seen.extend(new)
        return len(seen) >= 10
    res = b.stream([1, 2], req(max_tokens=150), None, cb)
    assert res.finish_reason == "abort" and b.stats["aborted"] == 1 and len(res.token_ids) == len(seen) >= 10
    out = [None] * 16

    def work(i):
        out[i] = b.submit([i + 1] * (i + 1), req(max_tokens=30)).token_ids
    ts = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(o == list(range(1, 31)) for o in out)


def test_refusals():
    b = broker(lambda p, r, l: ([1], None), ctx=100)
    with pytest.raises(ValueError):
        b.submit([1] * 90, req(max_tokens=20))
    cfg = E.FakeEngine(None, ctx=4096).vllm_config
    assert E.assert_engine(cfg, 4096) == 8192 * 16
    for mutate, msg in ((lambda c: setattr(c.model_config, "max_model_len", 2048), "context"),
                        (lambda c: setattr(c.cache_config, "enable_prefix_caching", False), "prefix"),
                        (lambda c: setattr(c.cache_config, "mamba_ssm_cache_dtype", "bfloat16"), "float32"),
                        (lambda c: setattr(c.cache_config, "num_gpu_blocks", None), "capacity")):
        c = E.FakeEngine(None, ctx=4096).vllm_config
        mutate(c)
        with pytest.raises(ValueError, match=msg):
            E.assert_engine(c, 4096)


def test_engine_failure_wakes_waiters():
    class Boom(E.FakeEngine):
        def step(self):
            raise RuntimeError("boom")
    b = E.Broker(Boom(lambda p, r, l: ([1], None), ctx=4096), 1 << 20, fake=True, ctx=4096)
    with pytest.raises(RuntimeError, match="engine failure"):
        b.submit([1], req(max_tokens=2))


# ---------------------------------------------------------------- proxy
def think_body(prompt, seed=3, **over):
    p = fx.PROTO
    body = dict(p.home_sampling(seed), prompt=prompt, n_predict=p.CTX - len(prompt) - p.FE_MAX - 128, stop=[p.CLOSE])
    body.update(over)
    return body


def proxy(scenario, log=None):
    b = broker(fx.make_script(scenario, log))
    return VLLMProxy(fx.MiniTok(), b, None, fx.EOS_IDS, proto=fx.PROTO)


PROMPT = fx.plain("Judge this proof.")


def test_think_natural_close():
    log = []
    out = proxy("natural", log).stream_think(think_body(PROMPT))
    assert out["stopped_word"] and out["tokens"][-1] == fx.CLOSE and out["content"] == "I check each step.\nThe argument is complete.\n"
    _, r, _ = log[0]
    assert r["presence_penalty"] == 1.5 and r["penalty_scope"] == "prompt" and r["stop_token_ids"][0] == fx.CLOSE


def test_think_literal_close_retokenized():
    out = proxy("literal").stream_think(think_body(PROMPT))
    assert out["stopped_word"] and out["content"] == "I check each step.\n"
    assert out["tokens"] == fx.plain("I check each step.\n") + [fx.CLOSE]


def test_think_eos_not_in_tokens():
    out = proxy("eos").stream_think(think_body(PROMPT))
    assert out["stopped_eos"] and fx.EOS not in out["tokens"] and out["content"] == "I check each step. It looks fine. Then"


def test_think_limit_exactly_room():
    body = think_body(PROMPT)
    out = proxy("limit").stream_think(body)
    assert out["stopped_limit"] and len(out["tokens"]) == body["n_predict"]


def test_think_loop_cut_at_check_multiple():
    out = proxy("loop").stream_think(think_body(PROMPT))
    p = fx.PROTO
    assert out["loop"] is not None and not (out["stopped_word"] or out["stopped_eos"] or out["stopped_limit"])
    assert len(out["tokens"]) == p.LOOP_CHECK_TOKENS * p.LOOP_CONFIRM_CHECKS


def test_think_refuses_capped_or_altered_request():
    px = proxy("natural")
    for bad in (dict(n_predict=2000), dict(temperature=1.0), dict(stop=["\n"])):
        with pytest.raises(ValueError):
            px.stream_think(think_body(PROMPT, **bad))


def test_proxy_complete_dispatches_to_verdicts():
    pytest.importorskip("torch")
    px = proxy("natural")
    got = px.complete(dict(fx.PROTO.read_body(), prompt=PROMPT))
    assert len(got["completion_probabilities"][0]["top_logprobs"]) == 100
    g = px.complete(dict(fx.PROTO.greedy_body(5, 20, ["\n"]), prompt=PROMPT))
    assert g["stopped_word"] and g["content"] == "none" and g["stopping_word"] == "\n"


def test_proxy_requires_close_id():
    with pytest.raises(ValueError):
        VLLMProxy(fx.MiniTok(), SimpleNamespace(), None, fx.EOS_IDS)   # ExampleProtocol leaves CLOSE_ID_PIN unset
