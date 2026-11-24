"""judgelab.logits: a prompt-aware presence penalty for vLLM's V1 logits-processor interface.

penalize(logits, seen, penalty)  reference rule: logits[seen] -= penalty (each distinct id once), as the HF reference reader samples.
make_processor_class()           -> PromptAwarePresencePenalty, a vLLM V1 LogitsProcessor (vllm imported lazily). Per request,
  SamplingParams.extra_args {'hfj3_presence': p, 'hfj3_scope': 'prompt'|'output'}: subtract p once from every distinct id in
  set(prompt_ids) | set(output_ids) ('prompt' = the reference reader's rule: prompt ids count as seen, then every sampled id) or
  set(output_ids) ('output' = stock vLLM presence_penalty). Requests without the key are untouched. output_ids is vLLM's LIVE list
  (it grows each step).

Why it exists: stock vLLM applies presence_penalty to generated tokens only. The reference reader also penalises prompt tokens, which
changes think length and wording on long prompts. Matching it is required before two readers can be compared at all.
"""

SCOPES = ("prompt", "output")


def penalize(logits, seen, penalty):
    import torch
    ids = torch.tensor(sorted(set(int(i) for i in seen)), dtype=torch.long, device=logits.device)
    out = logits.clone()
    if ids.numel():
        out[..., ids] -= penalty
    return out


class _State:
    """Row bookkeeping shared by the real processor and the tests (no vllm import)."""

    def __init__(self):
        self.rows = {}   # batch index -> (penalty, scope, prompt_ids set, live output list)

    def update(self, batch_update):
        if batch_update is None:
            return
        for added in batch_update.added:
            index, params, prompt_ids, output_ids = added[0], added[1], added[2], added[3]
            ea = getattr(params, "extra_args", None) or {}
            if "hfj3_presence" in ea:
                scope = ea.get("hfj3_scope", "prompt")
                if scope not in SCOPES:
                    raise ValueError("hfj3_scope must be prompt or output")
                self.rows[index] = (float(ea["hfj3_presence"]), scope, frozenset(prompt_ids or ()), output_ids)
            else:
                self.rows.pop(index, None)
        for index in batch_update.removed:
            self.rows.pop(index, None)
        for a, b, direction in batch_update.moved:
            ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
            if ra is not None:
                self.rows[b] = ra
            if rb is not None and "SWAP" in str(direction).upper():
                self.rows[a] = rb

    def apply(self, logits):
        import torch
        for index, (penalty, scope, prompt, output) in self.rows.items():
            if index >= logits.shape[0] or not penalty:
                continue
            seen = set(output) | (prompt if scope == "prompt" else set())
            if seen:
                ids = torch.tensor(sorted(seen), dtype=torch.long, device=logits.device)
                logits[index, ids] -= penalty
        return logits


def make_processor_class(base=None):
    """-> PromptAwarePresencePenalty subclassing vllm's V1 LogitsProcessor (or `base`, for tests)."""
    if base is None:
        from vllm.v1.sample.logits_processor import LogitsProcessor as base

    class PromptAwarePresencePenalty(base):
        def __init__(self, vllm_config=None, device=None, is_pin_memory=False):
            self.state = _State()

        def is_argmax_invariant(self):
            return False

        def update_state(self, batch_update):
            self.state.update(batch_update)

        def apply(self, logits):
            return self.state.apply(logits)

    return PromptAwarePresencePenalty
