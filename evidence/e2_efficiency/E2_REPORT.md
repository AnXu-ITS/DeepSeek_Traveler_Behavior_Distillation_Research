# Historical Teacher and Student efficiency comparison

This experiment measures successful Teacher requests and local Student inference, and projects the cost of repeated population-level prediction. Projected Teacher population times are not measured city-scale builds. Model-only, per-state and end-to-end construction timings exclude different work; they should not be compared without their denominators. A complete lifecycle cost would also include original labeling, failures, retries and training.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/evidence/e2_efficiency/E2_REPORT.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../docs/RESEARCH_DESIGN.md>) · [Training](<../../docs/TRAINING.md>) · [Results](<../../docs/RESULTS.md>) · [Data and access](<../../docs/DATA_SOURCES.md>) · [Model use](<../../docs/MODEL_USE.md>)

## Historical numerical table 1

|decision model| mean | P50 | P95 | min | max |unit|
|---|---|---|---|---|---|---|
| DeepSeek(100 requests / 87 successful measured requests, cache_bypass, K=1) | 86.539 | 87.414 | 162.793 | 11.387 | 184.215 | s |
| DeepSeek mean 95% bootstrap CI(B=2000, seed=42) | [76.745, 96.550] | | | | | s |
| S9 CPU sequential(threads=1, same as 100 state prefix) | 0.272 | 0.250 | 0.351 | 0.231 | 1.019 | ms |
| S9 CPU sequential(threads=1, 100,000 states in full) | 0.332 | 0.265 | 0.797 | 0.218 | 59.652 | ms |
| S9 CPU sequential(default threads=24, 100,000 states in full) | 0.334 | 0.264 | 0.815 | 0.215 | 52.899 | ms |

## Historical numerical table 2

| N | mean | P50 | P95 |
|---|---|---|---|
| 1 | 116.093 s | 116.093 s | 116.093 s |
| 10 | 70.964 s | 79.500 s | 162.793 s |
| 50 | 90.645 s | 89.170 s | 166.717 s |
| 100(87 successful) | 86.539 s | 87.414 s | 162.793 s |

## Historical numerical table 3

|decision model|measurement definition|throughput| 10k equivalent wall-clock time |
|---|---|---|---|
| DeepSeek | sequentially measured(N=100) | 0.7 states/min | 240.4 h(projection) |
| DeepSeek | 4-worker projection(historical reference 5–6 times/minutes) | ~5–6 states/min | ~28–33 h(projection) |
| S9 | CPU sequential threads=1(measured) | 3015 states/s | 3.3 s |
| S9 | batch 32(encode+forward, measured n=20,000) | 21909 states/s | 0.5 s |
| S9 | batch 256(encode+forward, measured n=20,000) | 25399 states/s | 0.4 s |
| S9 | batch 1024(encode+forward, measured n=20,000) | 21049 states/s | 0.5 s |

## Historical numerical table 4

| N |shared input construction|decision time|total wall-clock time| input ms/state | decision ms/state |
|---|---|---|---|---|---|
| 1 | 0.1 s | 0.0 s | 0.1 s | 105.9 | 3.29 |
| 100 | 26.2 s | 0.1 s | 26.3 s | 262.3 | 1.04 |
| 1,000 | 258.7 s | 1.0 s | 259.7 s | 258.7 | 1.00 |
| 10,000 | 2215.8 s | 10.5 s | 2226.3 s | 221.6 | 1.05 |
| 100,000 | 20808.8 s | 98.4 s | 20907.2 s | 208.1 | 0.98 |

## Historical numerical table 5

|ratio|value|
|---|---|
| latency ratio(DeepSeek mean / S9 threads=1 mean, same as 100 states) | 318,534× |
| throughput ratio(S9 batch1024 / DeepSeek sequential) | 1,821,566× |
| 10k cost ratio(DeepSeek / S9) | ∞(S9=$0); DeepSeek absolute value is T3 of cost@10k column |
