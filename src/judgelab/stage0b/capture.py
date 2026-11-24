"""judgelab.stage0b.capture: capture the two candidate logits of a pinned read before and after the native bf16 lm_head output cast.

For a pinned call, replay the Stage 0 read on one engine and capture, at the pinned read position, the logits of (token_up, token_down)
(a) immediately BEFORE the native bf16 output cast, as fp32 bits, and (b) AFTER it (the native bf16 logits, raw lm_head outputs, not
log-softmax). The native accumulation path is untouched (no fp32-output switch, no TF32 / reduced-precision flag changed on the GPU); the
hooks only COPY tensors.

  HF transformers reader: forward hook on core.lm_head (nn.Linear, bf16) -> its INPUT row (bf16 hidden state at the scored position) and
       OUTPUT (bf16 logits), i.e. the tensor before the reader's own .float().
  vLLM in-process reader: engine.apply_model(fn) runs fn(model) in-process; fn patches the model's ONE LogitsProcessor instance's
       _apply_head (the bf16 lm_head GEMM; also called by the LoRA wrapper) to copy (hidden_states, lm_head weight rows, output), and puts
       a forward hook on the outer 'logits_processor' module to check that the native values pass downstream unchanged.
  pre-cast fp32: the kernel's fp32 accumulator is not observable. Reconstruction from the captured bf16 hidden row h and bf16 weight row W[t]:
       (i) float64: math.fsum of the exact products (each bf16 x bf16 product is exact in binary64) = the correctly rounded exact dot;
       (ii) float32 (CPU torch.dot). pre_fp32_bits = fp32(round(i)). Per logit: distance of (i) to the nearest bf16 rounding midpoint vs
       |i - ii| and the bounds ceil(log2 n) * u * sum|h_i W_i| (tree order) and (n - 1) * u * sum|h_i W_i| (any order), u = 2^-24.
       midpoint_risk = distance <= 10 x max(tree bound, |i - ii|): then the kernel's own accumulation order could decide the cast and the
       reconstruction cannot.
  Sanity per engine: bf16_rne(pre_fp32_bits) == post_bf16_bits for both tokens; mismatches are written, never hidden.
  The read row is verified: the captured row's log_softmax must equal the read's returned top-100 logprobs (max |d| < 1e-3); B1: exactly
  one row per lm_head call and one live request.

This module holds the engine-independent capture machinery. Wiring it to a specific reader build (model loading, call selection, kit pins)
is deployment code and is not included; tests/test_stage0b_capture.py drives it with a tiny CPU model and a fake vLLM processor.
"""
import hashlib
import json
import math
import struct
import threading
from pathlib import Path

from .. import protocol

U32 = 2.0 ** -24
READ_MATCH_TOL = 1e-3
TOKEN_MATCH_TOL = 1e-4
RISK_FACTOR = 10.0
CFG = {"HFJ2": "ref", "HFJ3": "B1"}


class Refused(Exception):
    pass


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- bits
def f32_bits(x):
    return struct.unpack(">I", struct.pack(">f", x))[0]   # binary64 -> binary32, round to nearest even


def bits_f32(b):
    return struct.unpack(">f", struct.pack(">I", b))[0]


def bf16_rne(bits32):
    lsb = (bits32 >> 16) & 1
    return ((bits32 + 0x7FFF + lsb) >> 16) & 0xFFFF


def bf16_val(b16):
    return bits_f32(b16 << 16)


def bf16_up_down(b16):
    """(next larger value, next smaller value) bf16 bit patterns (finite, non-extreme)."""
    neg, mag = b16 & 0x8000, b16 & 0x7FFF
    if not neg:
        up = b16 + 1
        down = b16 - 1 if mag else 0x8001
    else:
        up = b16 - 1 if mag > 1 else 0x0000
        down = b16 + 1
    return up, down


def midpoint_distance(v, b16):
    up, down = bf16_up_down(b16)
    b = bf16_val(b16)
    m_up, m_dn = (b + bf16_val(up)) / 2.0, (b + bf16_val(down)) / 2.0
    return min(abs(v - m_up), abs(v - m_dn))


def reconstruct(h_row, w_row):
    """h_row, w_row: 1-D bf16 CPU tensors. -> dict (float64 exact-rounded dot, float32 dot, bits, bounds, midpoint distance)."""
    import torch
    hv, wv = h_row.float().tolist(), w_row.float().tolist()
    if len(hv) != len(wv) or not hv:
        raise Refused("hidden / weight length mismatch")
    prods = [a * b for a, b in zip(hv, wv)]          # exact in binary64 (8-bit x 8-bit significands)
    f64 = math.fsum(prods)                           # correctly rounded sum of exact products
    abs_sum = math.fsum(abs(p) for p in prods)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False    # CPU dot below; the GPU flag is restored (never changed for the read)
    try:
        f32 = float(torch.dot(h_row.to("cpu", torch.float32), w_row.to("cpu", torch.float32)).item())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    n = len(hv)
    bits = f32_bits(f64)
    cast = bf16_rne(bits)
    md = midpoint_distance(f64, cast)
    tree = math.ceil(math.log2(n)) * U32 * abs_sum
    worst = (n - 1) * U32 * abs_sum
    return {"f64": f64, "f32": f32, "f32_bits_of_f64": "0x%08x" % bits, "f32_bits_of_f32": "0x%08x" % f32_bits(f32),
            "abs_f64_minus_f32": abs(f64 - f32), "abs_sum": abs_sum, "n": n, "bound_tree": tree, "bound_worst": worst,
            "cast_bits": "0x%04x" % cast, "cast_of_f32_bits": "0x%04x" % bf16_rne(f32_bits(f32)), "midpoint_distance": md,
            "midpoint_over_tree_bound": (md / tree) if tree else None, "midpoint_risk": bool(md <= RISK_FACTOR * max(tree, abs(f64 - f32))),
            "safe_worstcase": bool(md > worst)}


def bf16_bits_of(t):
    import torch
    if t.dtype != torch.bfloat16:
        raise Refused("native output dtype %s is not bfloat16 (no native bf16 cast on this path)" % t.dtype)
    return int(t.reshape(1).view(torch.int16).item()) & 0xFFFF


# ---------------------------------------------------------------- recorder + hooks
class Recorder:
    """Copies (hidden rows, weight rows of `tokens`, full output rows) of every lm_head call while on; thread-safe."""

    def __init__(self, tokens, live_fn=None):
        self.tokens, self.live_fn = [int(t) for t in tokens], live_fn
        self.lock, self.on, self.cur, self.finals = threading.Lock(), False, None, None
        self.calls_total = 0

    def begin(self, on):
        with self.lock:
            self.on, self.cur, self.finals = bool(on), [], []

    def end(self):
        with self.lock:
            out, fin = self.cur, self.finals
            self.on, self.cur, self.finals = False, None, None
            return out or [], fin or []

    def head(self, hidden, weight, bias, out, where):
        with self.lock:
            self.calls_total += 1
            if not self.on:
                return
            h = hidden.detach().reshape(-1, hidden.shape[-1])
            o = out.detach().reshape(-1, out.shape[-1])
            self.cur.append(dict(h=h.to("cpu").clone(), out=o.to("cpu").clone(), w=weight.detach()[self.tokens].to("cpu").clone(), tokens=list(self.tokens),
                                 bias_none=bias is None, where=where, thread=threading.current_thread().name,
                                 live=(self.live_fn() if self.live_fn else None)))

    def final(self, out):
        with self.lock:
            if not self.on or out is None:
                return
            o = out.detach().reshape(-1, out.shape[-1])
            self.finals.append(dict(vals=o[:, self.tokens].to("cpu").clone(), shape=list(o.shape), dtype=str(o.dtype)))


def hook_hf(lm_head, rec):
    """HF reader: forward hook on core.lm_head (input row + bf16 output)."""
    def fn(mod, args, out):
        rec.head(args[0], mod.weight, getattr(mod, "bias", None), out, "hf_lm_head")
        rec.final(out)
    return lm_head.register_forward_hook(fn)


def hook_vllm(model, rec):
    """vLLM reader: patch the ONE LogitsProcessor's _apply_head (bf16 head GEMM) + forward hook on the outer 'logits_processor'."""
    lps = [m for m in model.modules() if type(m).__name__ == "LogitsProcessor"]
    if len(lps) != 1:
        raise Refused("want exactly one LogitsProcessor module, found %d" % len(lps))
    lp = lps[0]
    if not callable(getattr(lp, "_apply_head", None)):
        raise Refused("LogitsProcessor has no _apply_head (vLLM version?)")
    orig = lp._apply_head

    def _apply_head(*args, **kwargs):
        out = orig(*args, **kwargs)
        lm_head = args[0] if args else kwargs["lm_head"]
        hidden = args[1] if len(args) > 1 else kwargs["hidden_states"]
        bias = args[2] if len(args) > 2 else kwargs.get("embedding_bias")
        rec.head(hidden, lm_head.weight, bias, out, "vllm_apply_head")
        return out
    lp._apply_head = _apply_head
    outers = [(n, m) for n, m in model.named_modules() if n.rsplit(".", 1)[-1] == "logits_processor"]
    if len(outers) != 1:
        raise Refused("want exactly one module named logits_processor, found %d" % len(outers))
    outers[0][1].register_forward_hook(lambda mod, args, out: rec.final(out))
    return {"logits_processor_class": type(lp).__name__, "outer_name": outers[0][0], "outer_class": type(outers[0][1]).__name__,
            "head_dtype": str(getattr(lp, "head_dtype", None)), "scale": getattr(lp, "scale", None), "soft_cap": getattr(lp, "soft_cap", None),
            "logits_as_input": getattr(lp, "logits_as_input", None), "model_class": type(model).__name__}


# ---------------------------------------------------------------- read instrumentation + analysis
def instrument(server, rec, labels, capture_labels, stats_fn=None):
    """Wrap server.complete (instance attribute): every read is labelled by its exact prompt (labels: tuple(prompt) -> position); the
    recorder is on only for labels in capture_labels. Returns the list the reads are appended to. `del server.complete` undoes it."""
    orig, reads = server.complete, []

    def complete(body):
        prompt = tuple(body["prompt"])
        label = labels.get(prompt)
        s0 = dict(stats_fn()) if stats_fn else None
        rec.begin(label in capture_labels)
        try:
            res = orig(body)
        finally:
            recs, fins = rec.end()
        s1 = dict(stats_fn()) if stats_fn else None
        cached = (s1["cached_tokens"] - s0["cached_tokens"]) if s0 else None
        reads.append(dict(label=label, prompt_len=len(prompt), records=recs, finals=fins, result=res, cached_tokens=cached,
                          server_logits=(getattr(server, "logits", None).detach().to("cpu").clone()
                                         if getattr(server, "logits", None) is not None and label in capture_labels else None)))
        return res
    server.complete = complete
    return reads


def read_labels(plain, stored, proto=None):
    p = proto or protocol.current()
    prefix = list(stored["probe_ids"]) + plain(p.AJ_FE) + plain(stored["first_error_raw"]) + plain(p.AJ_SEV)
    labels = {tuple(prefix): "SEV"}
    for w in p.SEV_WORDS:
        labels[tuple(prefix + plain(w + p.AJ_SCORE))] = w
    return labels


def top_of(result):
    e = result["completion_probabilities"][0]
    return {int(t["id"]): float(t["logprob"]) for t in e["top_logprobs"]}


def analyse(read, up, down, engine):
    """-> per-call capture entry (checker keys + diagnostics) for the pinned read `read`."""
    import torch
    recs, fins = read["records"], read["finals"]
    if not recs:
        raise Refused("no lm_head call captured during the pinned read (hook did not fire)")
    top = top_of(read["result"])

    def dmax_of(r):
        if r["out"].shape[0] != 1:
            return float("inf")
        lsm = torch.log_softmax(r["out"][0].float(), dim=-1)
        return max(abs(float(lsm[t]) - v) for t, v in top.items())
    # the scored call = the LAST lm_head call of the read whose row reproduces the read's returned top-100 logprobs (chunked prefill
    # calls the head once per chunk; any later call would be a different position). Expected: the last call itself.
    d_all = [dmax_of(r) for r in recs]
    match = [i for i, d in enumerate(d_all) if d < READ_MATCH_TOL]
    if not match:
        if recs[-1]["h"].shape[0] != 1:
            raise Refused("B1: the last lm_head call of the read has %d rows (want 1)" % recs[-1]["h"].shape[0])
        raise Refused("captured row does not reproduce the read's returned logprobs (max |d| %.3g): wrong position" % d_all[-1])
    rec, dmax = recs[match[-1]], d_all[match[-1]]
    if rec["h"].shape[0] != 1:
        raise Refused("B1: the scored lm_head call has %d rows (want 1)" % rec["h"].shape[0])
    if rec["live"] is not None and rec["live"] != 1:
        raise Refused("B1: %d live requests at the scored lm_head call (want 1)" % rec["live"])
    if not rec["bias_none"]:
        raise Refused("lm_head has a bias")
    multi = [r["h"].shape[0] for r in recs if r["h"].shape[0] != 1]
    tok = {"up": int(up), "down": int(down)}
    idx = {t: i for i, t in enumerate(rec["tokens"])}
    pre, post, recon, ok = {}, {}, {}, {}
    for k, t in tok.items():
        native = rec["out"][0, t]
        post_bits = bf16_bits_of(native)
        r = reconstruct(rec["h"][0], rec["w"][idx[t]])
        pre[k], post[k], recon[k] = r["f32_bits_of_f64"], "0x%04x" % post_bits, r
        ok[k] = r["cast_bits"] == post[k]
    fin_eq = None
    if len(fins) == len(recs):
        f = fins[match[-1]]
        fin_eq = all(bf16_bits_of(f["vals"][0, idx[t]]) == int(post[k], 16) for k, t in tok.items()) if f["dtype"] == "torch.bfloat16" else False
    srv_eq = None
    if read.get("server_logits") is not None:
        srv_eq = bool(torch.equal(read["server_logits"].reshape(-1), rec["out"][0].float()))
    up_v, dn_v = bf16_val(int(post["up"], 16)), bf16_val(int(post["down"], 16))
    entry = {"position": None, "token_up": str(up), "token_down": str(down), "pre_fp32_bits": pre, "post_bf16_bits": post,
             "pre_fp32": {k: bits_f32(int(pre[k], 16)) for k in pre}, "post_bf16": {"up": up_v, "down": dn_v},
             "recon": recon, "sanity_cast_equals_native": ok, "sanity_ok": all(ok.values()),
             "final_logits_equal_native": fin_eq, "hf_server_logits_equal_capture": srv_eq,
             "read_match_max_abs_dlogp": dmax, "lm_head_calls_in_read": len(recs), "rows_per_call_not_1": multi,
             "scored_index_from_end": match[-1] - len(recs), "matching_calls": len(match),
             "live_requests_at_capture": rec["live"], "capture_where": rec["where"], "capture_thread": rec["thread"],
             "prompt_tokens": read["prompt_len"], "cached_tokens": read["cached_tokens"],
             "tokens_in_returned_top": {k: (t in top) for k, t in tok.items()},
             "logratio": {"post_bf16_logit_diff": up_v - dn_v,
                          "returned_logprob_diff": (top[tok["up"]] - top[tok["down"]]) if tok["up"] in top and tok["down"] in top else None,
                          "pre_fp32_diff": bits_f32(int(pre["up"], 16)) - bits_f32(int(pre["down"], 16))}}
    h16 = rec["h"][0].contiguous().view(torch.int16).numpy().astype("<i2").tobytes()
    entry["hidden_row"] = {"n": int(rec["h"].shape[1]), "dtype": str(rec["h"].dtype), "sha256": sha256_bytes(h16), "bf16_le_hex": h16.hex()}
    entry["weight_rows_sha256"] = {k: sha256_bytes(rec["w"][idx[t]].contiguous().view(torch.int16).numpy().astype("<i2").tobytes())
                                   for k, t in tok.items()}   # same checkpoint rows on both engines (cross-engine check)
    rd = entry["logratio"]["returned_logprob_diff"]
    if rd is not None and abs(rd - (up_v - dn_v)) > TOKEN_MATCH_TOL:   # logprob diff == logit diff up to fp32 rounding (~1e-6)
        raise Refused("wrong token: captured logit diff %.6g != returned logprob diff %.6g" % (up_v - dn_v, rd))
    return entry


def ref_logratio(path, sha, call_id, engine, pos, up, down):
    """The log-ratio of (up, down) in a stored Stage 0 row of this engine (to check that the replay reproduced Stage 0)."""
    blob = Path(path).read_bytes()
    if sha256_bytes(blob) != sha:
        raise Refused("ref rows sha mismatch")
    for line in blob.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("call_id") == call_id and r.get("arm") == "BASE" and r.get("reader") == engine and r.get("cfg") == CFG[engine]:
            t = (r.get("tops") or {}).get(pos) or {}
            if up in t and down in t:
                return {"ref_up": t[up], "ref_down": t[down], "ref_diff": t[up] - t[down], "ref_prompt_tokens": r.get("prompt_tokens")}
            return {"ref_missing_tokens": True}
    return None


def build_capture(engine, gpu_name, results, pin, meta, ref_rows=None, ref_rows_sha256=None):
    """results: {call_id: (reads, replay_row)} for every pinned call -> (capture dict in the checker format, sanity_ok)."""
    calls, sane = [], True
    for cid in sorted(pin):
        reads, row = results[cid]
        p = pin[cid]
        hits = [r for r in reads if r["label"] == p["position"] and r["records"]]
        if len(hits) != 1:
            raise Refused("%s: %d captured reads at position %s (want 1; pruned or unlabelled)" % (cid, len(hits), p["position"]))
        e = analyse(hits[0], p["token_up"], p["token_down"], engine)
        e["position"] = p["position"]
        e = dict({"call_id": cid}, **e)
        row_top = (row.get("tops") or {}).get(p["position"]) or {}
        e["replay_row_diff"] = (row_top[p["token_up"]] - row_top[p["token_down"]]) if p["token_up"] in row_top and p["token_down"] in row_top else None
        e["replay_row_prompt_tokens"] = row.get("prompt_tokens")
        if ref_rows:
            ref = ref_logratio(ref_rows, ref_rows_sha256, cid, engine, p["position"], p["token_up"], p["token_down"])
            e["ref"] = ref
            if ref and "ref_diff" in ref and e["replay_row_diff"] is not None:
                e["replay_minus_ref"] = e["replay_row_diff"] - ref["ref_diff"]
        sane = sane and e["sanity_ok"]
        calls.append(e)
    return {"engine": engine, "gpu_name": gpu_name, "calls": calls, "sanity_ok": sane, "meta": meta}, sane
