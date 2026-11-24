"""judgelab: an in-process vLLM judge reader and a reader-equivalence test kit for LLM proof judges.

Modules
  protocol     the judge-client contract the reader depends on (an illustrative ExampleProtocol is included)
  engine       one shared vLLM engine + a thread-safe Broker (continuous batching across judge calls); FakeEngine for CPU tests
  proxy        an HTTP-judge-server-compatible proxy (tokenize / stream_think / complete) over the Broker
  verdicts     the verdict sub-calls (top-100 logprob read, greedy extraction) with the reference server's semantics
  logits       prompt-aware presence penalty as a vLLM V1 logits processor
  replay       one teacher-forced replay shared by both readers (Stage 0 rows)
  equiv        Stage 0 and Stage 1 equivalence gates (numpy only)
  stage0_gate  the post-run Stage 0 verdict with input pre-checks
  stage0b      bf16 pre-cast logit capture and the rounding-flip mechanism check
  calibration     grader marks vs human marks; band-stratified selection; grader-vs-grader agreement
  judge_validity  known-answer, position-swapped validity test for pairwise judges
  canary          planted-flaw control records for LLM reviewers
  garble          statement-garbling detector and generator <-> marker statement contract
  rubric          scale maps and max-over-paths scoring for rubric audits
  evalstats       paired bootstrap, cross-fit calibration, ECE null band, predictive dev band
"""
__version__ = "0.2.0"
