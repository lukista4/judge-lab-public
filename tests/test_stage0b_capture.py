"""CPU tests of judgelab.stage0b.capture. A tiny torch model with a bf16 Linear head is read through a chunked-prefill reference server
(head called once per chunk) and judgelab.replay.replay_full, so the hook sees a realistic read path; the vLLM path is tested with a fake
LogitsProcessor (+ a fake LoRA wrapper) and a fake engine.apply_model. The Stage 0b checker is run as a subprocess on our outputs."""
import copy
import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
from conftest import subprocess_env  # noqa: E402
from judgelab import replay as RP  # noqa: E402
from judgelab.stage0b import capture as C  # noqa: E402

V, H = 4800, 64
PIN = {"call-A:s0": {"position": "SEV", "token_up": "16", "token_down": "4680"},
       "call-B:s0": {"position": "MAJOR", "token_up": "17", "token_down": "21"}}
SEV_IDS = {"NONE": [30], "MINOR": [31], "MAJOR": [32], "FATAL": [33]}
DIG = {str(d): [40 + d] for d in range(8)}
DESIGNATED = [16, 17, 21, 4680] + [30, 31, 32, 33] + [40 + d for d in range(8)]
VARIANTS = {"SEV": SEV_IDS, "AUDIT": DIG}
TOKS = sorted({int(PIN[c][k]) for c in PIN for k in ("token_up", "token_down")})


class TinyCore(nn.Module):
    """hidden(t) = bf16(mean of integer embeddings up to t), dim 0 forced to 1: exact and independent of chunking (cache = (sum, count))."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(V, H)
        self.embed.weight.data = torch.randint(-2, 3, (V, H), generator=g).to(torch.bfloat16)
        self.lm_head = nn.Linear(H, V, bias=False)
        w = torch.randn(V, H, generator=g) * 0.5
        w[:, 0] = 0.0
        w[DESIGNATED, 0] = 4.0
        self.lm_head.weight.data = w.to(torch.bfloat16)

    def body(self, ids, cache=None):
        x = self.embed(torch.tensor(ids)).float()
        s0, n0 = cache if cache is not None else (torch.zeros(H), 0)
        cs = torch.cumsum(x, 0) + s0
        h = cs / torch.arange(n0 + 1, n0 + 1 + x.shape[0], dtype=torch.float32)[:, None]
        h[:, 0] = 1.0
        return h.to(torch.bfloat16), (cs[-1], n0 + x.shape[0])


class TinyServer:
    """Reference-server surface: tokenize + complete(read body). Chunked prefill: the head runs once per chunk (last row of the chunk)."""

    def __init__(self, seed=0, chunk=7):
        self.core, self.chunk, self.logits = TinyCore(seed).eval(), chunk, None

    def tokenize(self, text, special=True):
        return [100 + (ord(c) % 997) for c in text]

    def complete(self, body):
        ids, cache = body["prompt"], None
        with torch.inference_mode():
            for i in range(0, len(ids), self.chunk):
                h, cache = self.core.body(ids[i:i + self.chunk], cache)
                out = self.core.lm_head(h[-1:])
            self.logits = out.float()[0]
            lsm = torch.log_softmax(self.logits, -1)
            v, t = torch.topk(lsm, 100)
        return {"timings": {"prompt_n": len(ids)},
                "completion_probabilities": [{"id": int(t[0]), "top_logprobs": [{"id": int(a), "logprob": float(b)} for a, b in zip(t, v)]}]}


def stored(cid, n, seed):
    g = torch.Generator().manual_seed(seed)
    return {"call_id": cid, "pid": "p", "probe_ids": torch.randint(100, 1000, (n,), generator=g).tolist(), "first_error_raw": "step 3"}


def tiny_results(seed=0):
    out, srv = {}, TinyServer(seed)
    rec = C.Recorder(TOKS)
    C.hook_hf(srv.core.lm_head, rec)
    for i, cid in enumerate(sorted(PIN)):
        st = stored(cid, 30 + 9 * i, 1 + i)
        labels = C.read_labels(lambda s: srv.tokenize(s, special=False), st)
        reads = C.instrument(srv, rec, labels, {PIN[cid]["position"]})
        row = RP.replay_full(srv, st, VARIANTS, arm="BASE", reader="HFJ2", cfg="ref")
        del srv.complete
        out[cid] = (reads, row)
    return out, srv


def write_caps(d, seed2=0, seed3=0, gpu="CPU-TEST"):
    paths = {}
    for eng, seed in (("HFJ2", seed2), ("HFJ3", seed3)):
        res, _ = tiny_results(seed)
        cap, _ = C.build_capture(eng, gpu, res, PIN, {"test": True})
        p = d / ("CAPTURE_%s.json" % eng)
        p.write_text(json.dumps(cap), encoding="utf-8")
        paths[eng] = p
    (d / "pin.json").write_text(json.dumps(PIN), encoding="utf-8")
    return paths


def checker(p2, p3, d):
    r = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", "judgelab.stage0b.check", str(p2), str(p3), str(d / "REPORT.json"),
                        "--pin", str(d / "pin.json")], capture_output=True, text=True, env=subprocess_env())
    return r.returncode, r.stdout


def test_bits():
    assert C.bf16_rne(C.f32_bits(1.0)) == 0x3F80
    assert C.bf16_rne(C.f32_bits(1.0 + 2 ** -8)) == 0x3F80            # exact midpoint ties to even
    assert C.bf16_rne(C.f32_bits(1.0 + 2 ** -8 + 2 ** -20)) == 0x3F81
    assert C.bf16_rne(C.f32_bits(-1.0 - 2 ** -8 - 2 ** -20)) == 0xBF81
    assert abs(C.midpoint_distance(1.0 + 2 ** -8 + 2 ** -20, 0x3F81) - 2 ** -20) < 1e-15
    up, down = C.bf16_up_down(0xC000)   # -2.0
    assert C.bf16_val(up) > -2.0 > C.bf16_val(down)


def test_hook_captures_right_row():
    res, srv = tiny_results(0)
    for i, cid in enumerate(sorted(PIN)):
        reads, _ = res[cid]
        pos = PIN[cid]["position"]
        hit = [r for r in reads if r["label"] == pos and r["records"]]
        assert len(hit) == 1
        st = stored(cid, 30 + 9 * i, 1 + i)
        prompt = [k for k, v in C.read_labels(lambda s: srv.tokenize(s, special=False), st).items() if v == pos][0]
        want = srv.core.body(list(prompt))[0][-1]   # independent no-cache forward of the full read prompt
        assert torch.equal(hit[0]["records"][-1]["h"][0], want)
        assert torch.equal(hit[0]["records"][-1]["out"][0], srv.core.lm_head(want[None])[0])
        assert len(hit[0]["records"]) > 1   # chunked prefill: several head calls, the last one is scored
        e = C.analyse(hit[0], PIN[cid]["token_up"], PIN[cid]["token_down"], "HFJ2")
        assert e["hf_server_logits_equal_capture"] is True and e["final_logits_equal_native"] is True
        assert e["read_match_max_abs_dlogp"] < 1e-5 and e["scored_index_from_end"] == -1


def test_reconstruction_equals_native_cast():
    res, _ = tiny_results(0)
    rec = [r for r in res[sorted(PIN)[0]][0] if r["records"]][0]["records"][-1]
    W = TinyCore(0).lm_head.weight.data
    n_safe = n_eq = 0
    for t in range(0, V, 37):
        r = C.reconstruct(rec["h"][0], W[t])
        if not r["midpoint_risk"]:
            n_safe += 1
            n_eq += int(r["cast_bits"], 16) == C.bf16_bits_of(rec["out"][0, t])
    assert n_safe >= 100 and n_eq == n_safe, (n_eq, n_safe)


def test_negative_wrong_position():
    res, _ = tiny_results(0)
    reads, _ = res["call-B:s0"]
    major = [r for r in reads if r["label"] == "MAJOR" and r["records"]][0]
    other = res["call-A:s0"][0]
    other_rec = [r for r in other if r["records"]][0]["records"][-1]
    for bad in (dict(major, records=major["records"][:-1]), dict(major, records=[other_rec])):
        with pytest.raises(C.Refused, match="wrong position"):
            C.analyse(bad, "17", "21", "HFJ2")
    assert [r for r in reads if r["label"] == "SEV"][0]["records"] == []   # only the pinned position is captured


def test_negative_wrong_token(tmp_path):
    res, _ = tiny_results(0)
    major = [r for r in res["call-B:s0"][0] if r["label"] == "MAJOR" and r["records"]][0]
    C.analyse(major, "17", "21", "HFJ2")
    bad = copy.deepcopy(major)
    bad["records"][-1]["tokens"] = [17 if t == 21 else (21 if t == 17 else t) for t in bad["records"][-1]["tokens"]]
    bad["records"][-1]["out"][0, 17], bad["records"][-1]["out"][0, 21] = major["records"][-1]["out"][0, 21], major["records"][-1]["out"][0, 17]
    with pytest.raises(C.Refused, match="wrong"):
        C.analyse(bad, "17", "21", "HFJ2")
    p = write_caps(tmp_path)
    cap = json.loads(p["HFJ3"].read_text("utf-8"))
    cap["calls"][0]["token_up"] = "18"
    p["HFJ3"].write_text(json.dumps(cap), encoding="utf-8")
    rc, out = checker(p["HFJ2"], p["HFJ3"], tmp_path)
    assert rc == 4 and out.startswith("REFUSED")


def test_checker_accepts_format(tmp_path):
    p = write_caps(tmp_path)
    rc, out = checker(p["HFJ2"], p["HFJ3"], tmp_path)
    assert rc == 1 and out.split()[0] == "FAIL", (rc, out)
    rep = json.loads((tmp_path / "REPORT.json").read_text("utf-8"))
    # identical engines: C1 holds, no shift -> FAIL (C2/C3), and the native values equal the common cast
    assert all(c["checks"]["C1_lt_one_step"] for c in rep["calls"].values())
    assert all(c["detail"]["native_equals_common_cast"] for c in rep["calls"].values())


MUTS = [
    lambda c: c.update(engine="HFJ2"),
    lambda c: c["calls"][0]["pre_fp32_bits"].update(up=int(c["calls"][0]["pre_fp32_bits"]["up"], 16)),
    lambda c: c["calls"][0]["post_bf16_bits"].update(down="0x" + "0" * 8),
    lambda c: c["calls"].pop(),
    lambda c: c["calls"][1].update(position="SEV" if c["calls"][1]["position"] != "SEV" else "MAJOR"),
    lambda c: c.update(gpu_name="NVIDIA A100"),
]


def test_checker_refuses_wrong_format(tmp_path):
    p = write_caps(tmp_path)
    good2, good3 = p["HFJ2"].read_text("utf-8"), p["HFJ3"].read_text("utf-8")
    for i, m in enumerate(MUTS):
        cap = json.loads(good3)
        m(cap)
        p["HFJ3"].write_text(json.dumps(cap), encoding="utf-8")
        p["HFJ2"].write_text(good2, encoding="utf-8")
        rc, out = checker(p["HFJ2"], p["HFJ3"], tmp_path)
        assert rc == 4, (i, rc, out)


# ---------------------------------------------------------------- fake vLLM
class LogitsProcessor(nn.Module):   # the class name matters: hook_vllm finds the ONE module of this class name
    def __init__(self, vocab):
        super().__init__()
        self.vocab, self.head_dtype, self.scale, self.soft_cap, self.logits_as_input = vocab, torch.bfloat16, 1.0, None, False

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        return torch.nn.functional.linear(hidden_states, lm_head.weight, embedding_bias)

    def forward(self, lm_head, hidden_states, embedding_bias=None):
        return self._apply_head(lm_head, hidden_states, embedding_bias)[..., :self.vocab]


class LogitsProcessorWithLoRA(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer

    def forward(self, lm_head, hidden_states, embedding_bias=None):
        logits = self.base_layer._apply_head(lm_head, hidden_states, embedding_bias)
        logits.add_(torch.zeros_like(logits))   # BASE request: no LoRA delta
        return logits[:, :self.base_layer.vocab]


class FakeLM(nn.Module):
    def __init__(self, lora=False):
        super().__init__()
        self.lm_head = TinyCore(0).lm_head
        lp = LogitsProcessor(V)
        self.logits_processor = LogitsProcessorWithLoRA(lp) if lora else lp

    def compute_logits(self, h):
        return self.logits_processor(self.lm_head, h)


class FakeVLLMModel(nn.Module):
    def __init__(self, lora=False):
        super().__init__()
        self.language_model = FakeLM(lora)

    def compute_logits(self, h):
        return self.language_model.compute_logits(h)


def apply_model(model, fn):   # == engine.apply_model -> worker.apply_model -> fn(get_model()), in-process
    return [fn(model)]


@pytest.mark.parametrize("lora", [False, True])
def test_vllm_fake_hook(lora):
    model = FakeVLLMModel(lora).requires_grad_(False)
    rec = C.Recorder([16, 17, 21, 4680], live_fn=lambda: 1)
    info = apply_model(model, lambda m: C.hook_vllm(m, rec))[0]
    assert info["outer_class"] == ("LogitsProcessorWithLoRA" if lora else "LogitsProcessor")
    hs = torch.randn(3, H, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    rec.begin(True)
    for k in range(3):   # one compute_logits per step; the last one is the scored row
        logits = model.compute_logits(hs[k:k + 1])
    recs, fins = rec.end()
    assert len(recs) == 3 and len(fins) == 3 and torch.equal(recs[-1]["h"][0], hs[2])
    lsm = torch.log_softmax(logits[0].float(), -1)
    top_ids = torch.topk(lsm, 100).indices.tolist()
    read = dict(records=recs, finals=fins, label="SEV", prompt_len=9, cached_tokens=0, server_logits=None,
                result={"completion_probabilities": [{"id": top_ids[0], "top_logprobs": [{"id": t, "logprob": float(lsm[t])} for t in top_ids]}]})
    e = C.analyse(read, "16", "4680", "HFJ3")
    assert e["final_logits_equal_native"] is True and e["live_requests_at_capture"] == 1 and e["capture_where"] == "vllm_apply_head"
    for k in ("up", "down"):
        assert e["recon"][k]["midpoint_risk"] or e["sanity_cast_equals_native"][k]
    rec.begin(True)   # NEGATIVE: two rows in the scored call (not B1) -> refused
    model.compute_logits(hs[:2])
    recs2, fins2 = rec.end()
    with pytest.raises(C.Refused, match="B1"):
        C.analyse(dict(read, records=recs2, finals=fins2), "16", "4680", "HFJ3")
    model.compute_logits(hs[:1])   # recorder off -> nothing captured
    assert rec.cur is None


def test_vllm_fake_refuses_two_processors():
    m = FakeVLLMModel()
    m.extra = LogitsProcessor(V)
    with pytest.raises(C.Refused):
        C.hook_vllm(m, C.Recorder([16]))


def test_native_dtype_refused():
    with pytest.raises(C.Refused, match="not bfloat16"):
        C.bf16_bits_of(torch.tensor(1.0, dtype=torch.float32))


def test_build_capture_needs_one_hit():
    res, _ = tiny_results(0)
    reads, row = res["call-A:s0"]
    with pytest.raises(C.Refused, match="captured reads"):
        C.build_capture("HFJ2", "CPU", {"call-A:s0": ([r for r in reads if r["label"] != "SEV"], row)},
                        {"call-A:s0": PIN["call-A:s0"]}, {})
