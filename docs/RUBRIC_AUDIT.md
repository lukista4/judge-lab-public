# Rubric audit: AI-written point schemes vs official marking rules

Code: `judgelab.rubric` (scale map, max-over-paths vs additive scoring; tests in `tests/test_rubric.py`).
Status: DERIVED (one agent's hand reading of counted differences). The official texts and item ids are not published.

## What was compared

- 38 development items from a national olympiad. Each had an AI-written 0-7 point scheme of the form "K1..Kn,
  additive", which graders marked against.
- 14 of the 38 also had official marking rules.
- Counting the official rules by keyword overstated their number (23), because the keyword also means "point" in
  geometry. A regex on the official full-mark sentence gave 14.

## Findings

1. **Scale mismatch.** The official full mark was **6** on all 14 items. Every AI scheme used 7.
   - Comparisons need an explicit monotone map: `rescale(s, 6, 7)` gives 0,1,2,4,5,6,7.
   - The map is not injective, so a 1-point difference on the 7-scale can be rounding.
2. **Max-over-paths lost.** On 8 of the 14 items the official rules give the maximum over alternative solution routes.
   The AI scheme wrote one route as an additive chain, so a solver on another valid route is under-credited. In the
   worst case a complete solution scores 0 (`tests/test_rubric.py`).
3. **Partial credit lost.** On 6 items the official rules give a point for the correct answer, or for a stated
   conjecture. The AI schemes' zero-credit field said the opposite.
4. **Caps and deductions missing.** For example, "-1 for a missing final check" and "incomplete without case X: at
   most 3" had no counterpart in the AI scheme.
5. Six items allocated the same total in different blocks. One item matched closely.

## Consequence (DERIVED)

- On these items, grader marks measure agreement with an AI-written scheme, not with the organisers' marking.
- Paired arm-vs-arm differences under the same scheme stay usable. Absolute levels need the scale map and carry the
  alternative-route risk.
- 24 of the 38 items had no official rules at all.
- Re-marking under the official rules was proposed and not run (PROJECTED).
