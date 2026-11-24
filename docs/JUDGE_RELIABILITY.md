# Judge reliability controls: known-answer swapped-order validity, and canary control

Code: `judgelab.judge_validity`, `judgelab.canary`. Demo: `python examples/demo_judge_checks.py`.
Labels: **MEASURED** / **DERIVED** / **PROJECTED**, as in the README.

## 1. Known-answer, position-swapped validity of a pairwise judge

**Question.** A pairwise judge says which of two versions, A or B, of a text is better (`overall` 0..4). Does it detect a
planted defect, and is it free of position and length bias?

**Design.**
- Each source text is paired with a copy carrying one script-generated defect:
  - a: a proof step dropped;
  - b: an inequality sign in a stated result flipped;
  - c: a statement and its proof swapped;
  - d: about 10% neutral padding;
  - e: an identical copy.
- Each pair is judged in both A/B orders: 30 pairs give 60 orders.
- The build is deterministic.
- The answer key (`manifest`) is kept apart from the runner input (`orders`: order id, prompt, prompt hash).
- `check_leakage` proves two things for every order:
  - no id or role word reaches a prompt;
  - every prompt is exactly the judge prefix plus the two texts.

**Decision rule.** It was written and stamped before any verdict was scored:

| rule | requirement |
|---|---|
| R1 | the intact text wins >= 20 of the 24 defect pairs |
| R2 | the degraded text wins <= 2 |
| R3 | padding is caught in >= 4 of 6 |
| R4 | no identical pair is won by the same copy in both orders |
| R5 | <= 3 of the 24 defect pairs contradict themselves across the two orders |

**Results of the original study** (MEASURED; 60 of 60 orders valid for each judge):

| judge | R1 | R2 | R3 | R4 | R5 | verdict |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 23 | 0 | 6 | 0 | 0 | USABLE |
| a GPT-family model | 20 | 0 | 6 | 0 | 0 | USABLE (R1 exactly at the threshold) |

Notes:
- The second judge's replies needed four collection rules before they could be scored: ignore trailing text, accept a
  UTF-8 BOM, accept one misplaced field, and add a missing completion block. The rules were approved and applied before scoring.
- The original pairs were built from course-text chapters that cannot be published, and their answer key is not
  published either.
- This repository rebuilds the method on synthetic texts written for it (`synthetic_chapter`). The synthetic study does
  **not** reproduce the numbers above. It exists to test the builder, the leakage check and the scorer.

**What the synthetic judges show** (MEASURED on the synthetic study, deterministic):
- An oracle judge is USABLE.
- A judge that always prefers the first text fails R5 (24 contradictions).
- A judge that prefers the longer text fails R2 and R3: it never catches padding.

**What USABLE licenses** (DERIVED). The judge detects these planted defect types on these texts. It does **not** show
that the judge's preferences on real, unplanted differences are right.

## 2. Canary / planted-flaw control for LLM reviewers

**Mechanism.**
- Every batch a reviewer lane works on carries controls: proofs with a planted fatal flaw, and correct proofs.
- Records are kept per lane and stage, counted per **distinct** control.
- One missed planted flaw voids every review of that lane in that batch.
- A review counts only if:
  - its lane has seen >= 6 distinct planted flaws;
  - it caught >= 90% of them;
  - it saw at least one planted flaw in the same batch.
- A lane that catches < 75% of >= 4 planted flaws is DISQUALIFIED.
- A lane that rejects > 50% of >= 3 valid controls is LOW-YIELD. Its rejections still block a row.
- "Flag-aware" catch: a planted flaw counts as caught if the fatal-flaw flag is set **or** the score is at or below the
  rejection gate.
- A mechanical control is added per batch: a real proof truncated to about 55%, so the conclusion is missing.

**Lesson built into the bank.** One control had passed its first flaw audit, yet three markers from one family
accepted the proof, and their readings said the planted flaw was not real. The control became disputed. Since then a new planted control enters the bank
only after a second, open-book audit by a different model family. That audit must end with four agreeing lines (flaw
real, not repaired elsewhere, proof not complete, verdict REAL-FATAL); any contradiction makes the verdict UNSURE
(`parse_second_audit`).

**Results** (MEASURED, from the proof-data pipeline where this ran):
- The bank had 52 graded proofs: 30 with a planted or known fatal flaw, 22 valid.
- Two reviewer lanes were disqualified:
  - one caught 2 of 6 planted flaws;
  - the other caught 3 of 6 when blind and 2 of 6 when shown the reference.
- Their reviews counted for nothing.

The real control proofs were written on competition problems and are not published. `synthetic_bank` holds
hand-written elementary examples.

**Reading** (DERIVED). Without controls in every batch, a rubber-stamp reviewer looks like a strict one that happens to
agree. The controls cost review capacity: in the pipeline where they ran, controls were about 31% of item verdicts
(MEASURED in a separate cost audit).
