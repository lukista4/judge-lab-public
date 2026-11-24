# Methodology

The project's question is whether a LoRA-trained judge scores human-marked olympiad proofs better than its base model. A small
true effect is easy to fake by accident (selection on noise, a changed reader, a peeked number). The rules below exist to prevent that.

## Pre-registration

- Decision rules (metrics, bars, panel sizes, seeds, verdict mapping) are written down and frozen before the numbers they decide exist.
- Amendments are allowed but recorded with what had been seen before they were written. Example: the Stage 0 flip rule
  (READER_EQUIVALENCE.md) was fitted after the Stage 0 rows were seen. It is labelled **non-blind**, the dissent against it is kept
  in the record, and the blind Stage 1 test remains the deciding one.
- A failed attempt stays failed under the rules that were frozen when it ran; a rerun is a new attempt with a new record.

## Blind gates

- Deciding numbers are not opened before the freeze. Whoever runs a blind gate posts only the verdict and the gate booleans until the
  freeze line exists.
- Labels (human correctness marks) are held separately; the Stage 1 scorer reads them and writes aggregates only, never per-proof rows.
- The equivalence verifier imports no reader or generator code, so the gate cannot be tuned through a shared helper.

## Cross-family review

Key arguments are reviewed by a model from a different family than the one that wrote them. One such review rejected the argument
that the two remaining Stage 0 discrepancies were single bf16 rounding flips ("two ulps, nothing else"). That led to Stage 0b, a direct
capture of the logits around the bf16 cast, which then failed the explanation (READER_EQUIVALENCE.md).

## Provenance before statistics

Inputs are refused (`SystemExit "REFUSED ..."`) before any statistic is computed when provenance does not match: wrong runtime string on
a row, a fake-engine run, a different penalty scope, more than one adapter hash or base-model pin set across readers, call ids that do
not match the manifest, held-out paths in a replay input. A refusal is never a pass and is not counted as an attempt.

## No mixing

Reads from different readers are never pooled inside one comparison. The same holds for GPU classes inside a paired design. Stage 0b
showed that the two readers differ by more than one bf16 rounding step on some logits, so a mixed comparison would carry a reader
effect that no test bounds.

## INCONCLUSIVE over a false PASS

An equivalence bar that grows with noise would certify a noisy panel. Every decisive metric therefore has a pre-registered ceiling:
if the test would pass but its bar exceeds the ceiling, the verdict is INCONCLUSIVE (underpowered), not PASS.

## Finite evidence is not proof

A PASS means "no difference larger than the bar was detected on this panel". Stated limitations that travel with any result:

- A bug confined to a few problems (for example +0.20 E and +20 % think length on 3 of 24 problems) passes the distributional test
  with high probability in simulation. There is no per-problem tail check.
- On the ADAPTER arm, E-bias and think length are reported, not decisive: a uniform E bias on that arm alone passes Stage 1. A bias
  common to both arms is caught on BASE; a LoRA-path-only bias is caught only by Stage 0's adapter-effect gate.
- At realistic noise, an E bias below the bar (about 0.18 on BASE) can pass. The ADAPTER-arm panel is small and catches only gross errors.
- Synthetic class separation in the test panels is a modelling choice; the noise levels are measured.
