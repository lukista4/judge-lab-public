# Reader equivalence

HFJ2 = the reference Hugging Face transformers bf16 reader (one request at a time). HFJ3 = the in-process vLLM reader (batched).
HFJ3 may replace HFJ2 in an evaluation only if it produces the same judgements. The code is `judgelab/equiv.py`,
`judgelab/stage0_gate.py` and `judgelab/stage0b/`.

## Why it has to be tested

- **bf16 arithmetic.** Different kernels (cuBLAS vs fused GEMMs, attention backends, Gated-DeltaNet state kernels) accumulate in
  different orders. In bf16 a logit near a rounding midpoint can move by a full step.
- **Batching.** Batch shape changes kernel choice and reduction order; a reader must not depend on how many other requests are live.
- **Prefix caching.** A cached prefix skips recomputation; the warm and the cold read must agree.
- **Sampling semantics.** Stock vLLM applies `presence_penalty` to generated tokens only; the reference also penalises prompt tokens.
  `judgelab.logits.PromptAwarePresencePenalty` reproduces the reference rule (scope `prompt`), and provenance requires that scope.

## Stage 0: logit-level replay

Stored judge calls are re-read teacher-forced (no sampling) from their stored verdict prefix with the same function on both readers
(`replay.replay_full`): one top-100 read at the severity position, one digit read per live severity. 24 calls per arm, stratified by
prefix length (12 in [5k, 15k), 6 in [20k, 35k), 6 in [35k, 55k] tokens), both arms (BASE, ADAPTER).

Configs: `ref` (HFJ2), `B1` (HFJ3, one request at a time), `B32` (HFJ3, 32 concurrent), `warm` (HFJ3 B1 again, every prefix cached).

| gate | bar (HFJ3 B1 vs HFJ2 ref unless stated) |
|---|---|
| S0.coverage | >= 24 calls per arm, >= 6 with prompt in [35k, 55k]; all four configs present |
| S0.E | \|dE\| mean <= 0.015, p95 <= 0.04, max <= 0.08 |
| S0.TVD | severity and digit distributions: mean TVD <= 0.005, max <= 0.025 |
| S0.highmass | tokens with p >= 0.01: \|dlogp\| mean <= 0.02, max <= 0.12; a missing high-mass token fails |
| S0.mass | raw severity / digit mass: mean <= 0.005, max <= 0.02 |
| S0.adapter | per-call adapter effect E(ADAPTER) - E(BASE): corr >= 0.995, magnitude ratio in [0.92, 1.08] (needs mean effect > 0.05) |
| S0.invariance | HFJ3 B32 and warm vs B1: max \|dE\| <= 0.015 (plus TVD / mass bounds) |

`stage0_gate` adds pre-checks: identical call lists for both readers, both jobs on the same GPU class with a clean exit, exact row
counts, and prefix-cache evidence from the replay (cold B1 hit share <= 0.90, warm >= 0.95, decoy prompts in the B32 rounds).

**Flip rule (non-blind amendment).** A call is a *flip call* if its ref-vs-B1 top lists differ only by log-ratio shifts within 0.25
(one bf16 step per logit at the observed magnitudes) and the same class of shift also occurs inside HFJ3's own B1/B32/warm spread.
Flip calls leave the E/TVD/high-mass/mass bars (which then run on the other calls), at most 12 per arm, and the E difference of every
flip call is reported. This rule was fitted after seeing Stage 0 rows; see METHODOLOGY.md.

## Stage 1: distributional test (blind), corr_rss

Full-length judge reads (thinking included) of a manifest of proofs. Two independent HFJ2 replicates (s0, s1) of the same proofs set
the noise scale; HFJ3 reads K samples per proof.

```
D     = | mean_k M(HFJ3_k) - (M(HFJ2_s0) + M(HFJ2_s1)) / 2 |
sigma = SD over a pid-cluster bootstrap (B = 10,000, fixed seed) of M(HFJ2_s1) - M(HFJ2_s0)
bar   = sqrt( (2.33 * c_K * sigma)^2 + m^2 ),   c_K = sqrt((1/K + 1/2) / 2)
```

The new reader must sit as close to the reference as the reference sits to itself. HFJ3 never enters sigma. c_K accounts for the
variance of a K-sample mean against a two-replicate mean. Metrics and margins m: pooled AUC (0.02), within-problem AUC (0.025), E bias on
all / correct / wrong proofs (0.10), log mean and log median think length (log 1.05), spread ratio SD(E_HFJ3)/SD(E_HFJ2) in
[0.88, 1.12]; on ADAPTER, the adapter-effect correlation. Sanity: no missing E, missing and infra shares < 1 %.

Verdict: FAIL if a decisive metric has D > bar; INCONCLUSIVE if it would pass but its bar exceeds the pre-registered ceiling (BASE:
pooled 0.06, within 0.10, E bias 0.30 / splits 0.40, think 0.10; ADAPTER: pooled 0.12, within 0.17, effect 0.35); AMBER if only a
secondary check (shape, text distinct-4, health shares) trips; else PASS. Provenance is checked before any statistic.

## Stage 0b: what happens at the bf16 cast

Two Stage 0 calls kept a deterministic 0.25 log-ratio shift between readers. Proposed explanation: both engines compute nearly the
same fp32 logit, the values straddle a bf16 rounding midpoint, and the cast turns a sub-ulp difference into one step per logit.

The GEMM's fp32 accumulator is not observable, so `stage0b.capture` copies the bf16 hidden row h at the scored position, the bf16
weight rows W[t] of the two tokens and the native bf16 logits, and reconstructs the pre-cast value: an exact float64 dot product
(`math.fsum` of exact bf16 x bf16 products) rounded to fp32, with a float32 recompute and error bounds (ceil(log2 n)·u·Σ|h_i W_i| and
(n-1)·u·Σ|h_i W_i|, u = 2^-24) plus the distance to the nearest rounding midpoint. `stage0b.check` then requires, per call:

- **C1** per logit, the two engines' pre-cast values differ by less than one bf16 step;
- **C2** they straddle a rounding midpoint, in opposite directions for the two tokens (the observed direction);
- **C3** one common round-to-nearest-even cast reproduces the observed shift exactly and equals each engine's native bf16 output.

## What failed, and why that is informative

On one call the reconstructed pre-cast values of the two engines differ by **0.12818**, more than one bf16 step (**0.125**) at that
magnitude, with reconstruction error <= 5.4e-5. C1 fails: the shift is not a single rounding flip; the engines' fp32 logits already
differ by more than an ulp of the output format.

Consequences:

- The readers are close but not interchangeable at the logit level. Mixing HFJ2 and HFJ3 reads inside one comparison is forbidden.
- Stage 1's BASE PASS (720 calls / 360 proofs / 78 problems) supports using HFJ3 for comparisons made entirely on HFJ3; it does not
  make HFJ3 numbers comparable with HFJ2 numbers.
- Without Stage 0b the flip rule would have absorbed these calls as "rounding noise". Measuring the mechanism directly replaced an
  argument with evidence, and the evidence disagreed.
