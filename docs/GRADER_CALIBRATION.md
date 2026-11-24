# Grader calibration against human marks, and agreement between graders

Code: `judgelab.calibration`. Aggregates: `results/calib40_proofbench_aggregate.json`, `results/grader_agreement_aggregate.json`.
Labels: **MEASURED** = from a run output; **DERIVED** = arithmetic or reading on measured numbers; **PROJECTED** = planned, not run.
Every analysis here was done by one AI agent and checked only by tests; no independent expert re-checked it.

## 1. Human-anchored calibration (ProofBench, 40 items)

**Data.** ProofBench (Ma et al. 2025; Hugging Face `wenjiema02/ProofBench`, MIT licence; paper arXiv:2510.13888). It has
435 model-written proofs of 145 competition problems (USAMO, IMO, Putnam, USA TST, APMO, EGMO, 2022-2025), three
generators per problem (OpenAI o3, Gemini 2.5 Pro, DeepSeek-R1-0528). Each proof carries an expert 0-7 rating given
against a problem-specific rubric. The dataset is **not** vendored here; fetch it from its source.

**Selection** (`select_stratified`, deterministic):
- contest year <= 2023, as a contamination guard;
- generator is not o3, because the grader under test is from the same family;
- proof length 600-30,000 characters, which drops stubs;
- a reference solution is present.

That leaves 147 eligible proofs. The band quotas are low (0-2) 13, mid (3-5) 13, high (6-7) 14. The mid band holds every
eligible mid-grade proof: grade 3 has 10 of the 13, and grades 4 and 5 have n = 2 and 1. Where possible each problem is
used once; the 40 items come from 35 distinct problems. Ids are re-assigned in hash order, so neither the id nor the
position carries the grade.

**Packet.** The grader gives one holistic 0-7 mark per proof against the official solution. The dataset rubric is
shown as marking guidance. The human mark is never in the packet.

**Results** (MEASURED; grader A = an OpenAI GPT-family model; one packet, four runs; n = 40 per run):

| run | QWK [95% bootstrap] | bias (grader - human) | MAE | exact | within 1 |
|---|---|---|---|---|---|
| 1 | 0.69 [0.48, 0.85] | -0.50 | 1.25 | 0.45 | 0.63 |
| 2 | 0.66 [0.42, 0.84] | -0.55 | 1.30 | 0.45 | 0.63 |
| 3 | 0.70 [0.49, 0.85] | -0.53 | 1.28 | 0.43 | 0.60 |
| 4 | 0.69 [0.47, 0.84] | -0.65 | 1.30 | 0.40 | 0.60 |

By human band, over the four runs (MEASURED):

| band | n | bias | QWK within band |
|---|---|---|---|
| high 6-7 | 14 | -1.14 to -1.43 | 0.02 to 0.05 |
| mid 3-5 | 13 | -0.92 to -1.00 | -0.01 to 0.14 |
| low 0-2 | 13 | +0.46 to +0.69 | 0.32 to 0.38 |

**Reading** (DERIVED):
- The overall QWK of about 0.69 comes from separating the bands, not from agreement inside a band. Inside the 6-7 band
  the agreement is at chance level.
- The grader is harsh on strong proofs and generous on weak ones.
- By generator, QWK is 0.73-0.75 on DeepSeek-R1 proofs (n = 27) and 0.33-0.40 on Gemini proofs (n = 13). These subsets are
  small.

**Copy baseline** (MEASURED, no grader calls): predict each proof's mark as the rounded mean of the expert marks of the
other two proofs of the same problem, without reading the proof. Over all 435 proofs this gives MAE 1.75 and exact 0.31.
Zero-shot grader A beats it on the 40 items (MAE 1.25-1.30), so it does read the proofs.

**Limits.**
- n = 40, and the mid band is mostly grade 3.
- The proofs are 2025-era frontier-model outputs, so they are distributed differently from the proofs this grader
  usually marks.
- The holistic packet is not the itemised point-scheme packet used in production.
- A few-shot variant (same-problem graded proofs shown as exemplars) was designed but never run (PROJECTED only).

## 2. Agreement of cheaper graders with the reference grader (394 proofs)

Three candidate graders and grader A each marked the same 394 proofs. The proofs were written by a 9B model on
olympiad-style problems (MEASURED; `paired_agreement`; high = score >= 6):

| candidate | mean diff vs A | exact | within 1 | QWK | precision / recall of high vs A |
|---|---|---|---|---|---|
| Mistral (hosted) | +1.02 (lenient) | 0.45 | 0.66 | 0.75 | 0.83 / 0.94 |
| GLM (hosted) | -0.08 | 0.65 | 0.93 | 0.92 | 0.96 / 0.86 |
| Claude Sonnet | -0.57 | 0.55 | 0.87 | 0.93 | 0.98 / 0.85 |

**Reading** (DERIVED): this table measures agreement, not accuracy. Section 1 shows that grader A itself under-grades
the 6-7 band relative to human experts, so a cheap grader can agree closely with a reference that is biased against
humans. Agreement with a model reference is model-adjudicated, not human gold.
