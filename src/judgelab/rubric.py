"""Rubric audit helpers: compare an AI-written additive point scheme with an official marking scheme (stdlib only).

Findings this encodes (docs/RUBRIC_AUDIT.md): AI-written 0-7 schemes were compared with official competition marking
rules. Three structural losses recurred:
  1. scale mismatch      the official full mark was 6, the AI scheme's 7; comparisons need an explicit monotone map;
  2. max-over-paths      official rules often give the maximum over alternative solution paths; an additive chain of
                         checkpoints written along one path under-credits a solver who took another valid path;
  3. bare-answer credit  official rules sometimes give a point for the correct answer or a stated conjecture; the AI
                         scheme's zero-credit field said the opposite.
"""


def round_half_up(x):
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def rescale(s, from_max, to_max):
    """Monotone map of an integer mark between scales, e.g. official 0..6 -> 0..7: round_half_up(7 s / 6).
    Not injective in general; differences of one point after mapping may be rounding."""
    if not 0 <= s <= from_max:
        raise ValueError("mark %r outside 0..%d" % (s, from_max))
    return round_half_up(to_max * s / from_max)


def additive_score(checkpoints, earned, cap=None):
    """checkpoints: {id: points}; earned: set of ids. Sum of points earned, optionally capped."""
    total = sum(p for k, p in checkpoints.items() if k in earned)
    return min(total, cap) if cap is not None else total


def max_over_paths(paths, earned, cap=None):
    """paths: list of {id: points} (alternative solution routes); the mark is the best single path."""
    best = max(additive_score(p, earned) for p in paths) if paths else 0
    return min(best, cap) if cap is not None else best


def audit_scheme(official_paths, ai_chain, official_max, ai_max, solver_paths):
    """Compare the two schemes on hypothetical solvers. solver_paths: {solver: set of earned checkpoint ids}.
    Returns per-solver official mark (mapped to the AI scale), AI-chain mark, and the difference."""
    out = {}
    for name, earned in solver_paths.items():
        off = rescale(max_over_paths(official_paths, earned, official_max), official_max, ai_max)
        ai = additive_score(ai_chain, earned, ai_max)
        out[name] = {"official_mapped": off, "ai_chain": ai, "ai_minus_official": ai - off}
    return out
