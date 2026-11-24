"""judgelab.verdicts: the verdict sub-calls of a judge read (one top-100 logprob read, or one greedy extraction) over a Broker.

complete_read / complete_greedy reproduce the reference HF server's `complete` semantics over the judgelab.engine Broker:
  * the same request-body checks against protocol.read_body() / protocol.greedy_body(),
  * reads: the top-100 raw logprobs sorted desc, refused if any value is non-finite or fewer than 100 entries arrive;
    'selected' = multinomial over softmax(top values) with torch.Generator('cpu').manual_seed(body['seed']),
  * greedy: vLLM runs WITHOUT stop strings (stop_token_ids = eos ids only); content / stop are decided LOCALLY with
    the reference rule: after each token content = decode(tokens), first stop string by position -> content[:at];
    eos -> stopped_eos (eos NOT in tokens); else stopped_limit after room = min(n_predict, CTX - len(prompt)) tokens.
The broker supplies the tokenizer as broker.tok (decode(ids)) and the eos ids as broker.eos_ids.
torch is imported lazily (reads only); vLLM is never imported here.
"""
import math

from . import protocol

N_TOP = 100


def _req(**kw):
    """Full Broker req with neutral sampling; callers override what they need."""
    req = {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": None,
           "stop_token_ids": [], "stop": [], "logprobs": None, "presence_penalty": 0.0, "penalty_scope": "output"}
    req.update(kw)
    return req


def _prompt(body, ctx):
    prompt = body["prompt"]
    if not isinstance(prompt, list) or not prompt or not all(isinstance(t, int) for t in prompt):
        raise ValueError("empty prompt")
    if len(prompt) > ctx:
        raise ValueError("input outside the context")
    return prompt


def _result_ok(res):
    if getattr(res, "finish_reason", None) == "abort":
        raise RuntimeError("broker request aborted")
    return {"timings": {"prompt_n": res.prompt_n}}


def complete_read(broker, body, lora, proto=None):
    """One top-100 logprob read (reference `complete` with n_probs)."""
    p = proto or protocol.current()
    prompt = _prompt(body, p.CTX)
    if any(body.get(k) != v for k, v in p.read_body().items()):
        raise ValueError("unexpected logprob request")
    res = broker.submit(prompt, _req(max_tokens=1, temperature=1.0, logprobs=N_TOP, seed=body["seed"]), lora)
    result = _result_ok(res)
    rows = res.logprobs
    if not rows or not isinstance(rows[0], dict):
        raise ValueError("no logprob row returned")
    row = rows[0]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row.values()):
        raise ValueError("nonfinite top logprob")
    if len(row) < N_TOP:
        raise ValueError("fewer than 100 top logprobs")
    top = sorted(((int(t), float(v)) for t, v in row.items()), key=lambda q: (-q[1], q[0]))[:N_TOP]
    import torch
    values = torch.tensor([v for _, v in top], dtype=torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(body["seed"])
    selected = top[int(torch.multinomial(torch.softmax(values, dim=-1), 1, generator=gen).item())][0]
    entries = [{"id": token, "logprob": value} for token, value in top]
    return dict(result, completion_probabilities=[{"id": selected, "top_logprobs": entries}])


def complete_greedy(broker, body, lora, proto=None):
    """Greedy extraction (local stop-string rule; vLLM stops only on eos / max_tokens)."""
    p = proto or protocol.current()
    prompt = _prompt(body, p.CTX)
    stops, limit = body["stop"], body["n_predict"]
    if any(body.get(k) != v for k, v in p.greedy_body(body["seed"], limit, stops).items()):
        raise ValueError("unexpected greedy request")
    room = min(limit, p.CTX - len(prompt))
    if room < 1:
        raise ValueError("no context room for greedy extraction")
    eos_ids = set(broker.eos_ids)
    req = _req(max_tokens=room, temperature=0.0, top_k=0, seed=body["seed"], stop_token_ids=sorted(eos_ids), stop=[])
    res = broker.submit(prompt, req, lora)
    result = _result_ok(res)
    ids = list(res.token_ids)
    if len(ids) > room:
        raise RuntimeError("broker returned more tokens than room")
    tokens, content = [], ""
    for token in ids:
        if token in eos_ids:
            return dict(result, content=content, tokens=tokens, stopped_eos=True)
        tokens.append(token)
        content = broker.tok.decode(tokens)
        matches = [(content.index(stop), i, stop) for i, stop in enumerate(stops) if stop in content]
        if matches:
            at, _, stop = min(matches)
            return dict(result, content=content[:at], tokens=tokens, stopped_word=True, stopping_word=stop)
    if len(tokens) != room:
        raise RuntimeError("broker stopped early without eos or stop string")
    return dict(result, content=content, tokens=tokens, stopped_limit=True)


def complete(broker, body, lora, proto=None):
    """The reference dispatch: n_probs -> read, else greedy."""
    return (complete_read if body.get("n_probs") else complete_greedy)(broker, body, lora, proto)
