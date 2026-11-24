"""Garble detector and statement contract: positive and negative cases (all strings written for this repository)."""
import pytest

from judgelab import garble as G

SIGNAL_CASES = [  # (text, GARBLE, RISK, why)
    ("Find all real ﬁxed points of the map g.", True, False, "ligature"),
    ("Show that a(1 − b) ≤ 4 c for all a, b, c in the unit interval.", False, True, "plain-text math: RISK"),
    ("Evaluate:  1        1          1 √ ≥√ +√ p   q", True, False, "flattened fractions (LAYOUT)"),
    ("Let b = √3 m where m is a positive integer.", True, False, "flattened root index (ROOTIDX)"),
    ("Show that the board needs at least k ` 2 colours.", True, False, "font-encoding + (ENC)"),
    ("Let B be a subset of t1, . . . , mu and k “ b1 ` b2.", True, False, "font-encoding braces and = (ENC)"),
    ("the points are lo­ cated on one line", True, False, "soft-hyphen line break (HYPH)"),
    ("Problem:\nLet $a^2+b^2=2$. Find the maximum of $a+b$.", False, False, "NEG clean LaTeX"),
    (r"\[\begin{array}{c} 1 &    &    & 1 \end{array}\] Find the next row.", False, False, "NEG LaTeX array spacing"),
    ("She wrote “ done ” and stopped; find k.", False, False, "NEG real quotation is not ENC"),
    ("Use the helper `h` to compute h(2).", False, False, "NEG markdown code backticks"),
    ("| x | $1$ | $2$ |\n|---|----|----| Find the mean.", False, False, "NEG markdown table"),
    ("Find all integers m such that m divides 3 to the power m minus 1.", False, False, "NEG plain prose"),
    ("Prove that u ≤ v whenever u^2 ≤ v^2 and u, v ≥ 0.", False, False, "NEG unicode ops but '^' present"),
    ("Let t be a non-negative integer.", False, False, "NEG ordinary hyphen inside a word"),
]


@pytest.mark.parametrize("text,garble,risk,why", SIGNAL_CASES, ids=[c[3] for c in SIGNAL_CASES])
def test_signals(text, garble, risk, why):
    s = G.signals(text)
    assert (s["GARBLE"], s["RISK"]) == (garble, risk), {k: v for k, v in s.items() if v}


def test_audit_counts():
    c = G.audit([t for t, *_ in SIGNAL_CASES])
    assert c["n"] == 15 and c["GARBLE"] == 6 and c["RISK"] == 1


GOOD = "Let $m \\ge 2$ be an integer. Prove that $3^m - 2$ is odd."
REF = "Since $3^m$ is odd for every $m$, $3^m - 2$ is odd minus even, hence odd."
CONTRACT = [
    ("clean: identical statements, reference present", GOOD, [GOOD, GOOD + "  \n"], REF, "clean"),
    ("NEG DIFF: generator lost the exponent", GOOD.replace("$3^m - 2$", "$3m - 2$"), [GOOD], REF, "corrupted"),
    ("NEG DIFF in one of two marker statements", GOOD, [GOOD, GOOD.replace("2", "4")], REF, "corrupted"),
    ("NEG digit merge with plain-text math", "Let c satisfy c3 − c = 251990. Show that g(c)3 − g(c) = 251990.",
     ["Let c satisfy c3 − c = 251990. Show that g(c)3 − g(c) = 251990."], "Since $c^3-c=25^{1990}$ we get ...", "corrupted"),
    ("NEG LAYOUT even when both sides agree", "Evaluate:  1        1          1 √ ≥√ +√ p   q",
     ["Evaluate:  1        1          1 √ ≥√ +√ p   q"], "Since 1/sqrt(p) >= ...", "corrupted"),
    ("ambiguous: ligature only", "Prove there are inﬁnitely many primes $p$ with $p \\equiv 3 \\pmod 4$.",
     ["Prove there are infinitely many primes $p$ with $p \\equiv 3 \\pmod 4$."], "Take $N = 4 p_1 \\cdots p_k - 1$ ...", "ambiguous"),
    ("ambiguous: plain-text math", "Prove that for all n ≥ 3 we have n ≤ 2n − 3.",
     ["Prove that for all n ≥ 3 we have n ≤ 2n − 3."], "For n = 3 equality holds.", "ambiguous"),
    ("ambiguous: orphan number alone", "Prove that some piece has area at most $12 S$.",
     ["Prove that some piece has area at most $12 S$."], "Each piece has area at most $S/2$ by averaging.", "ambiguous"),
    ("NEG no marker statement", GOOD, [], REF, "unjoined"),
    ("NEG no reference", GOOD, [GOOD], None, "ambiguous"),
]


@pytest.mark.parametrize("name,gen,markers,ref,want", CONTRACT, ids=[c[0] for c in CONTRACT])
def test_stratum(name, gen, markers, ref, want):
    got, why = G.stratum(gen, markers, ref)
    assert got == want, why


def test_marker_blocks():
    p = "x\n=== PROBLEM ===\nP text\n=== OFFICIAL SOLUTION (reference) ===\nR text\n=== MARKING GUIDANCE (r) ===\nM"
    assert G.marker_blocks(p) == ("P text", "R text")
    assert G.marker_blocks("no blocks") == (None, None)


def test_orphan_numbers():
    assert G.orphan_numbers("value 251990 and 25", "equals $25^{1990}$") == ["251990"]
    assert G.orphan_numbers("take 1990", "the year 1990") == []


def test_find_clean_source():
    prose = "Show that the remainder of the polynomial value when divided by three is always zero for every admissible choice of the parameter."
    garbled = "Let c satisfy c3 − c = 24. " + prose
    clean = "Let $c$ satisfy $c^3 - c = 24$. " + prose
    other = r"Let $x$ be real. Find the minimum of $x^2 + 1$ over all reals."
    hits = G.find_clean_source(garbled, [garbled, other, clean])   # the garbled copy has no LaTeX: never a clean source
    assert [i for i, _ in hits] == [2] and hits[0][1] == 1.0
