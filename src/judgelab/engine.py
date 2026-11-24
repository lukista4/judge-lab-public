"""judgelab.engine: one in-process vLLM engine shared by many judge-worker threads.

A Broker thread owns the engine (vllm.LLM(...).llm_engine, V1) and steps it; worker threads call the thread-safe submit() / stream().
That gives continuous batching across independent judge calls, prefix caching across calls that share a prompt, and DELTA outputs
that a per-token callback can inspect and abort (used by judgelab.proxy for the think-phase stop rules).

FakeEngine (CPU, scripted, deterministic) implements the subset of the engine API the Broker uses; every test runs on it.

Engine contract (asserted, refuse on mismatch): max_model_len == CTX, prefix caching enabled, Gated-DeltaNet (mamba) state cache in
float32, KV capacity known. KV admission: reserved prompt + generated tokens <= 0.85 * KV capacity; longest-prompt-first admission.
"""
import itertools
import json
import os
import queue
import sys
import threading
import traceback
from types import SimpleNamespace

from .protocol import CTX

ADMIT_FRACTION = 0.85
_ids = itertools.count()


class Result(SimpleNamespace):
    """token_ids, finish_reason ('stop'|'length'|'abort'), stop_reason, logprobs (list of {id: logprob}) or None, prompt_n."""


def assert_engine(cfg, want_ctx=CTX):
    mc, cc = cfg.model_config, cfg.cache_config
    if getattr(mc, "max_model_len", None) != want_ctx:
        raise ValueError("engine context differs from %d" % want_ctx)
    if not getattr(cc, "enable_prefix_caching", False):
        raise ValueError("prefix caching is not enabled")
    md = str(getattr(cc, "mamba_ssm_cache_dtype", "float32"))
    if "float32" not in md:
        raise ValueError("GDN state dtype is not float32: " + md)
    blocks, bs = getattr(cc, "num_gpu_blocks", None), getattr(cc, "block_size", None)
    if not blocks or not bs:
        raise ValueError("engine KV capacity unknown")
    return blocks * bs


def build_engine(base, adapter, max_num_seqs, fake, gmu=0.92, ctx=CTX):
    """-> (engine, kv_tokens, lora_request|None). `fake` = False, True (default script) or a script callable."""
    if fake:
        eng = FakeEngine(fake if callable(fake) else default_script, ctx=ctx)
        return eng, assert_engine(eng.vllm_config, ctx), ("fake-lora" if adapter is not None else None)
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise ValueError("VLLM_USE_FLASHINFER_SAMPLER must be 0")
    from vllm import LLM

    from .logits import make_processor_class
    kw = dict(model=str(base), dtype="bfloat16", tensor_parallel_size=1, max_model_len=ctx, max_num_seqs=max_num_seqs,
              gpu_memory_utilization=gmu, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, max_logprobs=100,
              mamba_ssm_cache_dtype="float32", logits_processors=[make_processor_class()], seed=0, enforce_eager=False)
    lora = None
    if adapter is not None:
        with open(os.path.join(str(adapter), "adapter_config.json"), "rb") as f:
            rank = int(json.loads(f.read())["r"])
        kw.update(enable_lora=True, max_loras=1, max_lora_rank=max(8, 1 << (rank - 1).bit_length()), lora_dtype="bfloat16")
    llm = LLM(**kw)
    eng = llm.llm_engine
    kv = assert_engine(eng.vllm_config, ctx)
    if adapter is not None:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest("arm", 1, str(adapter))
    return eng, kv, lora


def sampling_params(req):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    kw = dict(n=1, max_tokens=int(req["max_tokens"]), seed=req.get("seed"), stop_token_ids=list(req.get("stop_token_ids") or []),
              logprobs=req.get("logprobs"), output_kind=RequestOutputKind.DELTA, detokenize=False, skip_special_tokens=False)
    if float(req.get("temperature", 1.0)) == 0.0:
        kw.update(temperature=0.0)
    else:
        kw.update(temperature=float(req["temperature"]), top_p=float(req.get("top_p", 1.0)), top_k=int(req.get("top_k", 0)) or 0,
                  min_p=float(req.get("min_p", 0.0)))
    if req.get("presence_penalty"):
        kw["extra_args"] = {"hfj3_presence": float(req["presence_penalty"]), "hfj3_scope": req.get("penalty_scope", "prompt")}
    return SamplingParams(**kw)


class Broker:
    """One engine thread; thread-safe submit()/stream(). KV admission: reserved prompt+generated tokens <= 0.85 * kv_tokens; LJF order."""

    def __init__(self, engine, kv_tokens, fake=False, tok=None, eos_ids=(), ctx=CTX):
        self.eng, self.cap, self.fake, self.ctx = engine, int(kv_tokens * ADMIT_FRACTION), fake, ctx
        self.tok, self.eos_ids = tok, set(eos_ids)
        self.inbox, self.live, self.reserved = queue.Queue(), {}, 0
        self.fatal, self.stats = None, dict(requests=0, steps=0, gen_tokens=0, aborted=0, peak_live=0)
        self.thread = threading.Thread(target=self._loop, name="judgelab-engine", daemon=True)
        self.thread.start()

    def submit(self, prompt_ids, req, lora=None):
        return self.stream(prompt_ids, req, lora, None)

    def stream(self, prompt_ids, req, lora, on_delta):
        if len(prompt_ids) + int(req["max_tokens"]) > self.ctx:
            raise ValueError("request outside the %d context" % self.ctx)
        job = SimpleNamespace(rid="r%d" % next(_ids), prompt=list(prompt_ids), req=req, lora=lora, cb=on_delta, done=threading.Event(),
                              ids=[], lps=[] if req.get("logprobs") else None, finish=None, stop=None, prompt_n=None, error=None, cut=False)
        self.inbox.put(job)
        job.done.wait()
        if job.error is not None:
            raise RuntimeError("engine failure: " + job.error)
        return Result(token_ids=job.ids, finish_reason=job.finish, stop_reason=job.stop, logprobs=job.lps, prompt_n=job.prompt_n)

    def _admit(self, waiting):
        waiting.sort(key=lambda j: -len(j.prompt))   # LJF
        keep = []
        for job in waiting:
            need = len(job.prompt)
            if self.live and self.reserved + need > self.cap:
                keep.append(job)
                continue
            params = job.req if self.fake else sampling_params(job.req)
            self.eng.add_request(job.rid, {"prompt_token_ids": job.prompt}, params, lora_request=job.lora)
            self.live[job.rid] = job
            self.reserved += need
            self.stats["requests"] += 1
        self.stats["peak_live"] = max(self.stats["peak_live"], len(self.live))
        return keep

    def _finish(self, job, finish, stop):
        job.finish, job.stop = finish, stop
        self.live.pop(job.rid, None)
        self.reserved -= len(job.prompt) + len(job.ids)
        job.done.set()

    def _loop(self):
        waiting = []
        try:
            while True:
                block = not self.live and not waiting
                try:
                    waiting.append(self.inbox.get(timeout=0.05 if block else 0))
                    while True:
                        waiting.append(self.inbox.get_nowait())
                except queue.Empty:
                    pass
                waiting = self._admit(waiting)
                if not self.live:
                    continue
                outs = self.eng.step()
                self.stats["steps"] += 1
                for out in outs:
                    job = self.live.get(out.request_id)
                    if job is None:
                        continue
                    o = out.outputs[0]
                    new = list(o.token_ids)
                    if job.prompt_n is None:
                        job.prompt_n = len(job.prompt) - int(getattr(out, "num_cached_tokens", 0) or 0)
                    if job.cut:
                        continue
                    job.ids.extend(new)
                    self.reserved += len(new)
                    self.stats["gen_tokens"] += len(new)
                    if job.lps is not None and o.logprobs:
                        job.lps.extend({int(t): float(getattr(v, "logprob", v)) for t, v in row.items()} for row in o.logprobs[-len(new):])
                    if job.cb is not None and new and job.cb(new):
                        job.cut = True
                        self.eng.abort_request([job.rid])
                        self.stats["aborted"] += 1
                        self._finish(job, "abort", None)
                        continue
                    if out.finished:
                        self._finish(job, o.finish_reason, o.stop_reason)
        except BaseException as error:  # noqa: BLE001 - every waiter must wake up
            self.fatal = "%s: %s" % (type(error).__name__, error)
            traceback.print_exc(file=sys.stderr)
            for job in list(self.live.values()) + waiting:
                job.error = self.fatal
                job.done.set()
            while True:
                try:
                    job = self.inbox.get(timeout=1)
                except queue.Empty:
                    continue
                job.error = self.fatal
                job.done.set()


# ---------------------------------------------------------------- fake engine (CPU, scripted)
def default_script(prompt_ids, req, lora, close_id=248069):
    """Natural close for think requests, fixed verdict-like reads otherwise (ids only; tests pass their own scripts)."""
    if req.get("logprobs"):
        return [prompt_ids[-1] % 97 + 1], [{t: -0.01 * (t + 1) for t in range(1, 101)}]
    stops = list(req.get("stop_token_ids") or [])
    if req.get("temperature", 1.0) == 0.0:   # greedy extraction: ends with the first stop id (an eos), as a real engine would
        return [5, 6, 7] + stops[:1], None
    return [11, 12, 13, 10, close_id], None


class FakeEngine:
    """script(prompt_ids, req, lora) -> (token_ids, logprob_rows | None). Emits at most `chunk` tokens per step (exercises DELTA
    chunking), honours max_tokens / stop_token_ids (stop id included, finish 'stop') and records every request."""

    def __init__(self, script, ctx=CTX, prefix=True, chunk=7):
        self.script, self.chunk, self.reqs, self.live = script, chunk, [], {}
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=ctx),
            cache_config=SimpleNamespace(enable_prefix_caching=prefix, mamba_ssm_cache_dtype="float32", num_gpu_blocks=8192, block_size=16))

    def add_request(self, request_id, prompt, params, lora_request=None):
        ids = list(prompt["prompt_token_ids"])
        toks, lps = self.script(ids, params, lora_request)
        cut, finish, stop = list(toks), "length", None
        stops = set(params.get("stop_token_ids") or [])
        for i, t in enumerate(cut):
            if t in stops:
                cut, finish, stop = cut[:i + 1], "stop", t
                break
        if finish == "length" and len(cut) < params["max_tokens"]:
            finish = "stop"   # scripted natural end without a stop id
        cut = cut[:params["max_tokens"]]
        self.reqs.append(dict(prompt=ids, req=dict(params), lora=lora_request))
        self.live[request_id] = SimpleNamespace(toks=cut, lps=lps, pos=0, finish=finish, stop=stop)

    def has_unfinished_requests(self):
        return bool(self.live)

    def abort_request(self, ids):
        for i in ids:
            self.live.pop(i, None)

    def step(self):
        outs = []
        for rid, s in list(self.live.items()):
            new = s.toks[s.pos:s.pos + self.chunk]
            lp = [{k: SimpleNamespace(logprob=v) for k, v in row.items()} for row in (s.lps or [])[s.pos:s.pos + len(new)]] or None
            s.pos += len(new)
            done = s.pos >= len(s.toks)
            outs.append(SimpleNamespace(request_id=rid, finished=done, num_cached_tokens=0, outputs=[SimpleNamespace(
                token_ids=new, logprobs=lp, finish_reason=s.finish if done else None, stop_reason=s.stop if done else None)]))
            if done:
                del self.live[rid]
        return outs
