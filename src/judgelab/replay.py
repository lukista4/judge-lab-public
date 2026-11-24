"""judgelab.replay: ONE shared teacher-forced replay for both readers (the Stage 0 input of judgelab.equiv).

replay_full(server, stored, variants, ...) re-reads a stored judge call's verdict from its stored prefix (probe_ids + AJ_FE +
first_error_raw + AJ_SEV): one top-100 read at the severity position, then one digit read per severity whose share is >= SEV_PRUNE. It
keeps what the Stage 0 gates need: {call_id, pid, arm, reader, cfg, prompt_tokens, E, p_sev, sev_mass, p_digits_by_sev,
digit_mass_by_sev, tops{SEV, <sev>...}}. E = expected score = sum over kept severities of p(sev) * E[digit | sev].
server = anything with tokenize(text, special) + complete(read body): the reference HF reader or judgelab.proxy.VLLMProxy. Running the
SAME function on both readers is what makes positions and pruning comparable.

select_calls: the Stage 0 call set = AUDIT rows with status ok + probe_ids + first_error_raw, stratified by verdict-prefix length
(12 in [5k, 15k), 6 in [20k, 35k), 6 in [35k, 55k] tokens), deterministic by sha256(call_id).
"""
import hashlib
import json
import math
import threading
from pathlib import Path

from . import protocol

BANDS = ((5000, 15000, 12), (20000, 35000, 6), (35000, 55001, 6))
FINAL_MARKERS = ("final", "lockbox", "sealed", "heldout")   # inputs whose path names a held-out set are refused


def replay_full(server, stored, variants, *, arm, reader, cfg, proto=None):
    p = proto or protocol.current()

    def plain(text):
        return server.tokenize(text, special=False)

    def read(ids):
        return p.top_list(server.complete(dict(p.read_body(), prompt=ids)))[1]

    prefix = stored["probe_ids"] + plain(p.AJ_FE) + plain(stored["first_error_raw"]) + plain(p.AJ_SEV)
    top = read(prefix)
    mass = {w: sum(math.exp(top[t]) for t in variants["SEV"][w] if t in top) for w in p.SEV_WORDS}
    total = sum(mass.values())
    p_sev = {w: (mass[w] / total if total else 0.0) for w in p.SEV_WORDS}
    tops, p_dig, d_mass = {"SEV": {str(k): v for k, v in top.items()}}, {}, {}
    num = den = 0.0
    for w, value in mass.items():
        if not total or value / total < p.SEV_PRUNE:
            continue
        t2 = read(prefix + plain(w + p.AJ_SCORE))
        tops[w] = {str(k): v for k, v in t2.items()}
        lp, flags = p.answer_logps(t2, "AUDIT", variants)
        digits = {d: math.exp(v) for d, v in lp.items() if "floor_" + d not in flags}
        d_mass[w] = round(sum(digits.values()), 6)
        if digits:
            z = sum(digits.values())
            p_dig[w] = {d: q / z for d, q in digits.items()}
            num += (value / total) * sum(int(d) * q for d, q in digits.items()) / z
            den += value / total
    return {"call_id": stored["call_id"], "pid": stored.get("pid"), "arm": arm, "reader": reader, "cfg": cfg, "prompt_tokens": len(prefix),
            "E": (num / den) if den else None, "p_sev": p_sev, "sev_mass": round(total, 6), "p_digits_by_sev": p_dig,
            "digit_mass_by_sev": d_mass, "tops": tops}


def load_pool(srcs):
    """READS.jsonl shards (journal entries {'score': row, ...} or bare rows), sha-checked; held-out paths refused;
    pooled and deduplicated by (a, sample) then call_id (first source wins)."""
    pool, seen = [], set()
    for path, sha in srcs:
        if any(m in str(path).lower() for m in FINAL_MARKERS):
            raise SystemExit("REFUSED held-out path in Stage 0 input: %s" % path)
        blob = Path(path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != sha:
            raise SystemExit("REFUSED stored sha %s" % path)
        for line in blob.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row = dict(row.get("score", row), _src=str(path))
            key = (row.get("a"), row.get("sample"), row.get("call_id"))
            if (key[0], key[1]) in seen or key[2] in seen:
                continue
            seen.update({(key[0], key[1]), key[2]})
            pool.append(row)
    return pool


def select_calls(rows, n_total=24):
    ok = [r for r in rows if r.get("kind") == "AUDIT" and r.get("status") == "ok" and r.get("probe_ids") and r.get("first_error_raw") is not None]
    key = lambda r: hashlib.sha256(r["call_id"].encode()).hexdigest()   # noqa: E731
    pick = []
    for lo, hi, k in BANDS:
        band = sorted((r for r in ok if lo <= r.get("p2_tokens", 0) < hi), key=key)
        pick += band[:k * n_total // 24]
    return pick


def run_rows(servers, stored, variants, arm, reader, cfg, threads=1, proto=None):
    """servers: one server per thread (proxies share one engine); returns rows in stored order."""
    out, lock, it = [None] * len(stored), threading.Lock(), iter(range(len(stored)))

    def work(srv):
        while True:
            with lock:
                i = next(it, None)
            if i is None:
                return
            out[i] = replay_full(srv, stored[i], variants, arm=arm, reader=reader, cfg=cfg, proto=proto)

    ts = [threading.Thread(target=work, args=(servers[j % len(servers)],)) for j in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if any(r is None for r in out):
        raise RuntimeError("a replay thread failed")
    return out
