"""judgelab.proxy: an HTTP-judge-server-compatible proxy over the shared Broker (one proxy per worker thread).

The judge client talks to a server only through `tokenize(text, special)`, `stream_think(body)` and `complete(body)`. VLLMProxy offers
exactly these, with the reference server's request-body checks and result keys, so the client logic runs unchanged over vLLM.

stream_think applies the reference per-token stop rules, in order, on every streamed token (via the Broker's on_delta callback):
  close id -> stop 'word' (close kept); eos -> stop 'eos'; a literal close string spelled by plain tokens -> content cut there, tokens
  re-tokenized + close id, stop 'word'; room exhausted -> 'limit'; every LOOP_CHECK_TOKENS tokens a degenerate-loop check, confirmed after
  LOOP_CONFIRM_CHECKS consecutive hits -> 'loop'. The think is never capped below the context: a capped request is refused.
"""
from types import SimpleNamespace

from . import protocol


class Tokenizer:
    """Thin adapter over a `tokenizers.Tokenizer`-like backend: tokenize(text, special) / decode(ids)."""

    def __init__(self, backend):
        self.backend = backend

    def tokenize(self, text, special=True):
        prev = getattr(self.backend, "encode_special_tokens", False)
        self.backend.encode_special_tokens = not special   # special=False: special-token strings are encoded as plain text
        try:
            return list(self.backend.encode(text, add_special_tokens=False).ids)
        finally:
            self.backend.encode_special_tokens = prev

    def decode(self, ids):
        return self.backend.decode(list(ids), skip_special_tokens=False)


class VLLMProxy(Tokenizer):
    def __init__(self, backend, broker, lora, eos_ids, penalty_scope="prompt", proto=None):
        super().__init__(backend)
        self.p = proto or protocol.current()
        if self.p.CLOSE_ID_PIN is None:
            raise ValueError("protocol CLOSE_ID_PIN is not set")
        self.broker, self.lora, self.eos_ids, self.scope = broker, lora, set(eos_ids), penalty_scope
        self.close = self.p.CLOSE_ID_PIN
        self._piece = {}

    def piece(self, token):
        if token not in self._piece:
            self._piece[token] = self.decode([token])
        return self._piece[token]

    def reset(self):
        """No transport state: vLLM prefix caching replaces the reference server's incremental cache."""

    def stream_think(self, body):
        p = self.p
        prompt, room = body["prompt"], body["n_predict"]
        sampling = p.home_sampling(body["seed"])
        if (room != p.CTX - len(prompt) - p.FE_MAX - 128 or room < p.MIN_THINK_ROOM
                or body["stop"] != [p.CLOSE] or any(body[k] != v for k, v in sampling.items())):
            raise ValueError("unexpected or capped think request")
        st = SimpleNamespace(tokens=[], hits=0, loop=None, stop=None, content=None)

        def on_delta(new):   # the reference per-token rules, in order, on each streamed token
            for token in new:
                if token == self.close:
                    st.tokens.append(token)
                    st.stop = "word"
                    return True
                if token in self.eos_ids:
                    st.stop = "eos"
                    return True
                st.tokens.append(token)
                if ">" in self.piece(token) and p.CLOSE in self.decode(st.tokens[-len(p.CLOSE):]):
                    st.content = self.decode(st.tokens).split(p.CLOSE, 1)[0]
                    st.tokens = self.tokenize(st.content, special=False) + [self.close]
                    st.stop = "word"
                    return True
                if len(st.tokens) == room:
                    st.stop = "limit"
                    return True
                if len(st.tokens) % p.LOOP_CHECK_TOKENS == 0:
                    signal_ = p.degenerate_signal(self.decode(st.tokens))
                    st.hits = st.hits + 1 if signal_ else 0
                    if st.hits >= p.LOOP_CONFIRM_CHECKS:
                        st.loop, st.stop = signal_, "loop"
                        return True
            return False

        req = dict(max_tokens=room, temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=sampling["top_k"],
                   min_p=sampling["min_p"], seed=sampling["seed"], stop_token_ids=[self.close] + sorted(self.eos_ids),
                   presence_penalty=sampling["presence_penalty"], penalty_scope=self.scope)
        res = self.broker.stream(prompt, req, self.lora, on_delta)
        if st.stop is None:   # the engine ended the request without a per-token rule firing
            if res.finish_reason == "length":
                st.stop = "limit"
            elif res.finish_reason == "stop" and res.stop_reason in self.eos_ids:
                st.stop = "eos"
            elif res.finish_reason == "stop" and res.stop_reason == self.close:
                st.tokens.append(self.close)
                st.stop = "word"
            else:
                raise RuntimeError("think ended without a classified stop: %s %s" % (res.finish_reason, res.stop_reason))
        stop = None if st.stop == "loop" else st.stop
        content = st.content
        if content is None:
            content = self.decode(st.tokens[:-1] if stop == "word" else st.tokens)
        return {"content": content, "tokens": st.tokens, "loop": st.loop, "timings": {"prompt_n": len(prompt)},
                "stopped_word": stop == "word", "stopped_eos": stop == "eos", "stopped_limit": stop == "limit"}

    def complete(self, body):
        from . import verdicts
        return verdicts.complete(self.broker, body, self.lora, self.p)
