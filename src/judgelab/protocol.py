"""judgelab.protocol: the judge-client contract the reader and the replay depend on.

The reader (judgelab.engine / judgelab.proxy / judgelab.verdicts) is transport code: it reproduces, over an in-process vLLM engine, the
request semantics of a llama.cpp-style HTTP judge server (`tokenize`, `stream_think`, `complete`). Which request bodies are legal, where
the verdict prefix fragments go and how a top-100 row is turned into answer probabilities is defined by the judge client.

The production judge client (prompts, rubric, verdict format) is NOT part of this repository. `ExampleProtocol` below is a small,
self-consistent stand-in so the transport and the replay can be exercised end to end on CPU. Its constants are illustrative, not the
values of any real run. Swap in your own client with `use(obj)`; any object exposing the same attributes works.
"""
import math

CTX = 65536   # one judge slot: prompt + uncapped thinking + verdict


class ExampleProtocol:
    CTX = CTX
    CLOSE = "</think>"
    CLOSE_ID_PIN = None          # the tokenizer id of CLOSE; set by the caller (tokenizer-specific)
    FE_MAX = 384                 # room reserved for the first-error extraction after the think
    MIN_THINK_ROOM = 1024
    LOOP_CHECK_TOKENS = 256      # degenerate-loop check cadence (tokens)
    LOOP_CONFIRM_CHECKS = 2      # consecutive positive checks before a loop is declared
    SEV_WORDS = ("NONE", "MINOR", "MAJOR", "FATAL")
    SEV_PRUNE = 0.01             # severities below this share are not expanded into digit reads
    DIGITS = tuple(str(d) for d in range(8))
    AJ_FE = '\n{"first_error": "'               # verdict-prefix fragments (illustrative JSON-shaped verdict)
    AJ_SEV = '", "severity": "'
    AJ_SCORE = '", "score": '

    def read_body(self):
        """Body keys of a one-token top-100 logprob read (everything except the prompt)."""
        return dict(n_predict=1, n_probs=100, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, seed=0, presence_penalty=0.0,
                    repeat_last_n=0)

    def greedy_body(self, seed, n_predict, stop):
        """Body keys of a greedy extraction (local stop-string rule)."""
        return dict(n_predict=n_predict, stop=list(stop), temperature=0.0, top_k=0, top_p=1.0, min_p=0.0, seed=seed, presence_penalty=0.0,
                    repeat_last_n=0)

    def home_sampling(self, seed):
        """Sampling of the think phase (illustrative values in the range recommended for thinking-mode chat models)."""
        return dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, seed=seed)

    def degenerate_signal(self, text, window=400, min_repeats=4):
        """A repeated-suffix loop detector: returns a short label if the last `window` chars are one phrase repeated, else None."""
        tail = text[-window:]
        for size in range(8, window // min_repeats + 1):
            unit = tail[-size:]
            if unit.strip() and tail.endswith(unit * min_repeats):
                return "repeat:%d" % size
        return None

    def top_list(self, result):
        """complete() result of a read -> (selected id, {token id: logprob})."""
        entry = result["completion_probabilities"][0]
        return entry["id"], {int(t["id"]): float(t["logprob"]) for t in entry["top_logprobs"]}

    def answer_logps(self, top, kind, variants):
        """Digit log-probabilities from a top-100 row. variants[kind][digit] = token ids spelling that digit.
        Returns ({digit: logp}, flags); a digit with no variant in the top list gets the floor (min logp of the row) and flag 'floor_<d>'."""
        floor = min(top.values()) if top else -1e9
        lp, flags = {}, []
        for d, ids in variants[kind].items():
            vals = [top[t] for t in ids if t in top]
            if vals:
                m = max(vals)
                lp[d] = m + math.log(sum(math.exp(v - m) for v in vals))
            else:
                lp[d] = floor
                flags.append("floor_" + d)
        return lp, flags


_CURRENT = ExampleProtocol()


def current():
    return _CURRENT


def use(obj):
    """Install a judge-client object (must expose the ExampleProtocol attributes). Returns the previous one."""
    global _CURRENT
    prev, _CURRENT = _CURRENT, obj
    return prev
