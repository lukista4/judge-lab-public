# Performance

Labels: **MEASURED** = read from run telemetry or a benchmark log; **DERIVED** = arithmetic on measured numbers; **PROJECTED** = not
measured yet. Throughput is generated tokens per second per GPU. Model: Qwen3.5-9B bf16, 65,536-token context, thinking reads.

## Baselines

| setup | gen tok/s | label |
|---|---|---|
| reference HF transformers reader, one request at a time, L40S | 35.5 (~30-35 typical) | MEASURED |
| in-process vLLM reader, bench at batch 64 / 128 / 192, A100-40GB | 1,416 / 2,081 / 1,933 | MEASURED (bench workload, not production shape) |
| in-process vLLM reader, production shards, L40S | 195-328 | MEASURED |

## Why production runs at 195-328 tok/s, not ~2,000

Five production shards on L40S (max_num_seqs 64, gpu_memory_utilization 0.92, 24 calls per shard), MEASURED from telemetry:

| shard | wall s | gen tok/s | mean live (gen tokens per step) | peak live | prefix-cache hit |
|---|---|---|---|---|---|
| 1 | 1809 | 205 | 6.9 | 24 | 0.74 |
| 2 | 943 | 279 | 9.8 | 24 | 0.74 |
| 3 | 1472 | 195 | 6.3 | 24 | 0.74 |
| 4 | 1002 | 328 | 13.2 | 24 | 0.73 |
| 5 | 1037 | 231 | 7.3 | 24 | 0.73 |

- Peak live = 24 = the shard's call count. Neither the 64 slots nor the KV pool (~742k tokens, MEASURED) bound concurrency.
- Think lengths vary widely, so a shard drains: most of the wall time runs with far fewer than 24 live requests (**tail-bound**).
- The KV pool holds ~25-35 live long thinks at 20-25k tokens each (DERIVED).

## Levers

| lever | effect | label |
|---|---|---|
| continuous batching across calls (Broker) | the 195-328 tok/s above vs ~35 for the reference reader | MEASURED |
| prefix caching | 0.73-0.74 hit share in production (verdict reads reuse the think prefix) | MEASURED |
| more calls per engine (>= 72 per engine, about 3x the KV-bound live count) | keeps live count near the KV limit; removes most of the tail | PROJECTED |
| max_num_seqs: 64 on 46 GB cards, 128 on A100 | 128 was the best bench setting on A100-40GB | MEASURED (bench) |
| max_num_seqs 512 + gpu_memory_utilization 0.95 | 0.985x | MEASURED, rejected |
| MTP-3 speculative decoding | 1.848x at batch 24, 1.604x at batch 1; 0.81x at 256 live | MEASURED without runtime LoRA |
| MTP-3 with runtime LoRA + the penalty processor + raw logprobs | unknown until certified against the non-MTP reader | PROJECTED |
| several LoRA adapters in one pooled engine | fills KV with more live requests across arms | PROJECTED (needs isolation tests of prefix-cache sharing across adapters) |

Notes:
- MTP gains are largest at low-to-moderate concurrency, which is where tail-bound production shards spend their time.
  It loses at high concurrency (>= 128-256 live).
- Any change to the reader (MTP, multi-LoRA, kernel flags) is a new reader for the purposes of READER_EQUIVALENCE.md and must not
  enter a running comparison.
