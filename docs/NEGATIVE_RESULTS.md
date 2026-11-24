# Negative results

Things that did not work, or worked less well than expected.

**How to read this page.**
- Labels: **MEASURED** = from a run output or log. **MEASURED (2nd-hand)** = quoted in a notes file that cites a raw
  file not re-opened for this write-up. **DERIVED** = a reading or arithmetic on measured numbers.
- Each item was checked by one AI agent only, unless stated otherwise.
- "Marks" are model grades (0-7) on private problem sets, not human grades.

**Regimes.**
- Cluster: Qwen3.5-9B served with vLLM 0.33 (bf16) on L40S / A100 GPUs, on a university Slurm cluster.
- Home: the same model as GGUF (Q4_K_M) under llama.cpp with the Vulkan backend, on one 8 GB AMD RX 7600.

## Inference and serving

1. **fp8 KV cache is lossy for this model and engine** (MEASURED, cluster).
   - It gave 1.18x throughput but failed the token-equivalence gate: 25 of 50 rows diverged, against a repeat-run floor
     of 18.
   - Max |Δlogprob| was 0.496, against a bar of 0.2.
   - Rejected.
2. **A dynamic speculative-decoding schedule cannot rescue high-concurrency throughput** (MEASURED, cluster).
   - With speculative decoding configured, the engine pays its cost even at k = 0: the time ratio was 1.285.
   - MTP-3 gives 1.6-2.2x at 1-48 concurrent sequences but 0.81x at 256.
3. **Speculative decoding breaks cold/warm reproducibility** (MEASURED, cluster).
   - With MTP or n-gram drafting, greedy outputs depend on the prefix-cache state: cold and warm runs differ on 6 of 9
     prompts, against 0 of 9 without speculation.
   - Sampling-quality gates still pass. This is a reproducibility property, not a quality loss, but it rules out
     byte-exact replays.
4. **A format-specific kernel speed-up silently could not fire** (MEASURED (2nd-hand), home).
   - A custom small-batch Q4_0 matmul kernel was tested on the deployed Q4_K_M file: 104.9 vs 101.4 tok/s, a tie.
   - The tensor census explains it: the file has 0 Q4_0 tensors, and the kernel only handles Q4_0.
   - A wide-batch WMMA path won at 32 x 768 (542 vs 434 tok/s) and tied at 16 x 4,096.
   - All custom kernel lines were retired, because no Q4_0-format file passed the KL-divergence quality screen.
5. **An n-gram drafter on an already saturated consumer GPU: -27%** (MEASURED (2nd-hand), home).
   - Long thinking calls at 8 slots x 8,192 tokens ran at 74.1 vs 101.4 tok/s with the drafter.

## Training

6. **Gradient bank** (MEASURED, cluster): the gradient-parity check failed. Rejected.
7. **Fused "exact-math" LoRA kernels** (MEASURED, cluster): the gradient error was 5-10x the reference PEFT error, so
   the unit tests failed. Rejected.
8. **Closed-book frontier teachers on hard targets** (MEASURED).
   - Two external preview models were asked 229 target problems. One stopped on rate limits after 60.
   - They produced 3 usable training rows: 1 of 27 and 2 of 54 correct.
   - In 14 cases both teachers were wrong and agreed with each other.

## Evaluation and data

9. **Sequential stopping and sub-sampling rules for eval sampling did not pay** (MEASURED, cluster).
   - One rule saved nothing at 4 samples per problem.
   - Another failed its pre-registered gate (17 of 40).
   - A post-hoc band that looked good was explicitly not counted.
10. **A proposed residual-audit predictor was unstable** (MEASURED): 0 of 20 seeds agreed.
11. **Web-scraped olympiad statements were badly damaged** (MEASURED).
    - 46% of a 200-problem pool were GARBLE and 26% were RISK.
    - The upstream "garbled" labels were ignored downstream.
    - See [STATEMENT_GARBLING.md](STATEMENT_GARBLING.md).
12. **Answer-cap overflow cues do not work** (MEASURED, model-graded).
    - Thinking models often overflow a fixed answer budget: 73% of attempts hit an 8,192-token cap.
    - Soft or hard end-of-thinking cues fixed nothing.
    - An answer-only continuation was not better than a larger answer budget.
    - A 16k answer budget gave +0.80 marks [lower 95% bound +0.33] on one sample of 50 capped attempts, and -0.50 on
      another sample of 12. Mixed.
13. **LLM graders are harsh on strong proofs** (MEASURED).
    - Within the 6-7 human band, grader agreement with experts was at chance (QWK 0.02-0.05), and the grader was 1.1-1.4
      points too low.
    - See [GRADER_CALIBRATION.md](GRADER_CALIBRATION.md).

## Formal proofs and process

14. **An AI code assistant as a Lean 4 prover** (MEASURED): 3 of 25 proof stubs were accepted, and all 3 were
    calibration controls with known easy proofs. That is 0 of the 20 real stubs.
15. **Cross-family audits of LLM-written mathematics mostly found problems** (MEASURED, one count, not de-duplicated).
    - In an AI-driven mathematics side project, each LLM-written argument was audited by a model from a different
      family.
    - Of 1,298 audit rows with a verdict, 101 (7.8%) were clean CONFIRMED, 1,021 confirmed only after repairs,
      133 found a gap and 43 refuted the argument.
    - Caveats: the rows mix audit tiers and re-audits of the same subject, and about 20 rows are malformed.
    - No claim is made here about the mathematics itself.
16. **A heavily pre-registered multi-agent audit experiment cost more process than it produced** (MEASURED).
    - The apparatus certification failed, so the later checks were never run.
    - Consolidating the status record needed 44 reconciliations of conflicting status statements.
    - The pilot closed with a null verdict.
    - Lesson (DERIVED): certify the apparatus on a small scale before building governance around it.
