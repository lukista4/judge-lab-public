# Statement garbling: detector and generator <-> marker statement contract

Code: `judgelab.garble` (tests: `tests/test_garble.py`, 15 signal cases including 8 negatives, plus 10 contract cases).
Labels: **MEASURED** / **DERIVED** / **PROJECTED**. This was a single-agent audit; the spot check was one agent's reading.

## Problem

Competition statements extracted from PDFs lose structure. In the audited data:
- `33^1992` became `331992`;
- a fraction `1/4` vanished;
- `∛(6n)` became `3 6n`;
- `{1, ..., n}` became `t1, ..., nu`;
- `+` became a backtick.

A model then solves a different problem, and a grader marks the answer against yet another version.

## Detector (no clean reference needed)

Direct signals (GARBLE):
- typographic ligatures;
- line-break hyphens;
- flattened layout, i.e. runs of >= 4 spaces (ignored inside LaTeX or tables);
- a flattened root index (`√3 n`);
- font-encoding substitutes (`` n ` 1 ``, `“` for `=`, `t1, ..., nu`).

Indirect signal (RISK): unicode math operators with no LaTeX and no `^`, so exponents and fractions cannot be
represented.

The detector says nothing about meaning. An exponent lost without any other trace is invisible to it.

## Contract

The statement the generator saw and the statement **every** marker prompt carried must be equal after NFKC and
whitespace normalisation. Rows are stratified conservatively:
- **corrupted:** the two statements differ, or there is a structural garble, or there are orphan numbers plus a signal;
- **ambiguous:** a soft signal only, or orphan numbers alone, or no reference;
- **clean**;
- **unjoined.**

An *orphan number* is a multi-digit integer in the statement that appears in no digit run of the reference solution.
It is evidence of a digit merge.

`find_clean_source` looks for a LaTeX version of a damaged statement by prose word 5-gram containment, which is robust to
the garbled math tokens.

## Findings (MEASURED unless marked)

| set | n | GARBLE | RISK |
|---|---|---|---|
| web-scraped olympiad proof pool (built from PDF collections) | 200 | 92 (46%) | 52 (26%) |
| a curated readout set | 60 | 1 (ligature only, meaning intact) | 0 |
| a large mixed training pool | 250,328 | 85 (0.03%) | 767 (0.3%) |
| an open reasoning-trace set (OpenR1) | 36,833 | 21 | 29 |

- **Spot check** (DERIVED, one agent's reading): in 8 of 13 flagged rows of the 200-problem pool, the damage changed the
  meaning.
- **Filter gap:** the upstream builder had already labelled 100 of the 200 problems "garbled", yet they entered the pool.
  The detector also flags 54 of the 98 problems that the builder labelled "clean".
- **Confound** (DERIVED; rows are attempts, so they are not independent): among model failures classed "misread or
  changed statement", 91% were on GARBLE or RISK problems. The pool base rate is 72%. A large part of that failure
  class is probably an input defect, not a model defect.
- **Clean sources:** only 2 of 195 damaged problems had a LaTeX version in the local sets.
