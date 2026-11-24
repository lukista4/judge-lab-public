"""Statement-garbling detector and generator <-> marker statement contract (stdlib only).

Problem statements scraped from PDFs often lose structure: exponents flatten (2^n - 3 -> 2n-3), fractions collapse into
columns, a cube root becomes '√3 n', '+' becomes a backtick. A model then solves (and a grader then marks) a different
problem. This module flags such damage heuristically, without a clean reference, and checks that the text the generator
saw is the text the marker saw.

Signals of `signals(text)`:
  LIG        a typographic ligature (U+FB00..U+FB06) - the text came out of a PDF text layer
  HYPH       a line-break hyphen left inside a word ('co- efficients', soft hyphen)
  LAYOUT     a run of >= 4 spaces inside the statement (stacked fractions / limits flattened into columns);
             not counted when the text has LaTeX or a markdown table
  ENC        font-encoding substitutes between math tokens ('n ` 1' for n+1, '“' for =, 'ă' for <, 't1, ..., nu' for {1..n})
  ROOTIDX    '√<digit> <var>' (an index-n root flattened)
  PLAINMATH  unicode math operators with NO LaTeX and NO '^' anywhere: exponents / fractions cannot be represented
GARBLE = LIG or HYPH or LAYOUT or ROOTIDX or ENC (direct evidence). RISK = PLAINMATH without GARBLE (indirect).
The detector says nothing about meaning: a GARBLE row may still read fine, and an exponent lost without any other
trace is invisible to it (that needs a clean reference; see `orphan_numbers` and `find_clean_source`).

Contract (`stratum`): the generator statement and EVERY marker statement of the problem must be equal after NFKC and
whitespace normalisation. Strata, conservative:
  corrupted  contract DIFF, or a structural garble (LAYOUT / ROOTIDX / ENC), or orphan numbers together with any signal
  ambiguous  any other signal: ligature / hyphen only, RISK, orphan numbers alone, or no reference
  clean      SAME, no signal, no orphan numbers, a reference is present
  unjoined   no marker statement found
An ORPHAN NUMBER is an integer token (>= 2 digits) of the statement that occurs in none of the reference's digit runs
(e.g. '331992' when the reference writes 33^{1992}): evidence of a digit merge.
"""
import collections
import re
import unicodedata

LIG = re.compile("[ﬀ-ﬆ]")
HYPH = re.compile("[a-z][-­‐‑] [a-z]{2,}")
LAYOUT = re.compile(r"\S {4,}\S")
ENC = re.compile(r"[A-Za-z0-9)] ` [A-Za-z0-9(¨]|[A-Za-z0-9)] [¨ăą] ?[A-Za-z0-9(]|[{\s]t[0-9a-z], ")
ENCQ = re.compile(r"[A-Za-z0-9)] “ [A-Za-z0-9(]")
ROOTIDX = re.compile(r"√\s?[2-9]\s+[A-Za-z(]")
MATHOPS = re.compile("[−√≤≥·×∈∑∏⌊⌋≡≠∞]")
LATEX = re.compile(r"\$|\\[A-Za-z(\[]")
WRAP = re.compile(r"^.*?Problem:\s*", re.S)


def strip_wrapper(t):
    """Drop a chat wrapper that precedes 'Problem:' within the first 400 characters."""
    return WRAP.sub("", t, count=1) if "Problem:" in t[:400] else t


def signals(t):
    t = strip_wrapper(t or "")
    s = {k: bool(rx.search(t)) for k, rx in (("LIG", LIG), ("HYPH", HYPH), ("ROOTIDX", ROOTIDX), ("ENC", ENC))}
    s["ENC"] = s["ENC"] or (bool(ENCQ.search(t)) and "”" not in t)
    s["LATEX"] = bool(LATEX.search(t))
    s["LAYOUT"] = bool(LAYOUT.search(t)) and not s["LATEX"] and "|" not in t
    s["PLAINMATH"] = bool(MATHOPS.search(t)) and not s["LATEX"] and "^" not in t
    s["GARBLE"] = s["LIG"] or s["HYPH"] or s["LAYOUT"] or s["ROOTIDX"] or s["ENC"]
    s["RISK"] = s["PLAINMATH"] and not s["GARBLE"]
    return s


def audit(texts):
    """Counts of each signal over an iterable of statements."""
    c = collections.Counter()
    for t in texts:
        s = signals(t)
        c["n"] += 1
        for k in ("GARBLE", "RISK", "LIG", "HYPH", "LAYOUT", "ROOTIDX", "ENC", "LATEX"):
            c[k] += s[k]
    return dict(c)


def norm(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t or "")).strip()


BLOCK = re.compile(r"=== PROBLEM ===\n(.*?)\n=== OFFICIAL SOLUTION[^\n]*===\n(.*?)\n=== MARKING GUIDANCE", re.S)


def marker_blocks(prompt_text):
    """(statement, reference) from a grading prompt laid out as PROBLEM / OFFICIAL SOLUTION / MARKING GUIDANCE blocks."""
    m = BLOCK.search(prompt_text or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def orphan_numbers(stmt, ref):
    runs = set(re.findall(r"\d+", ref or ""))
    out = []
    for n in sorted(set(re.findall(r"(?<![\d.])\d{2,}(?!\d)(?!\.\d)", stmt or ""))):
        if n not in runs and not any(n in r for r in runs if len(r) > len(n)):
            out.append(n)
    return out


def stratum(gen, markers, ref):
    """gen: generator statement; markers: marker statements (may be empty); ref: reference text or None.
    Returns (stratum, reasons)."""
    if not markers:
        return "unjoined", ["no marker statement"]
    why = []
    diff = [i for i, m in enumerate(markers) if norm(m) != norm(gen)]
    if diff:
        why.append("contract DIFF in %d of %d marker statements" % (len(diff), len(markers)))
    s = signals(gen)
    struct = [k for k in ("LAYOUT", "ROOTIDX", "ENC") if s[k]]
    soft = [k for k in ("LIG", "HYPH") if s[k]] + (["RISK"] if s["RISK"] or (s["PLAINMATH"] and s["GARBLE"]) else [])
    orph = orphan_numbers(gen, ref) if ref else []
    if struct:
        why.append("structural garble " + "+".join(struct))
    if soft:
        why.append("signal " + "+".join(soft))
    if orph:
        why.append("orphan numbers " + ",".join(orph[:5]))
    if not ref:
        why.append("no reference")
    if diff or struct or (orph and (soft or s["GARBLE"])):
        return "corrupted", why
    if soft or orph or not ref:
        return "ambiguous", why
    return "clean", why or ["SAME, no signal"]


def grams(t, n=5):
    """Prose word n-grams (alphabetic words of length >= 3), robust to garbled math tokens."""
    w = [x for x in re.findall(r"[a-z]+", (t or "").lower()) if len(x) >= 3]
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def find_clean_source(statement, candidates, min_shared=6, min_containment=0.6):
    """Candidates (iterable of texts) that carry LaTeX and contain >= min_containment of the statement's prose grams.
    Returns a list of (index, containment)."""
    g = grams(statement)
    out = []
    if not g:
        return out
    for i, c in enumerate(candidates):
        if not LATEX.search(c or ""):
            continue
        m = len(g & grams(c))
        if m >= min_shared and m / len(g) >= min_containment:
            out.append((i, round(m / len(g), 3)))
    return out
