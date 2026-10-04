# Historical pipeline latency audit

The implemented deployment chain performs Student inference in Python before MATSim starts and hands off plans as files. It does not make per-decision Python/Java service calls during the reported fixed-plan simulation. The audit separates process setup, supply/feature construction, Student inference, plan construction, simulation and event parsing. Reusing unchanged supply calculations can save much more time than optimizing an already compact model. Cold, warm and model-only timings use different denominators; historical successful-request Teacher timings do not include all retry and training costs.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/PIPELINE_LATENCY_AUDIT.md).

[Current research](<RESEARCH_DESIGN.md>) · [Training](<TRAINING.md>) · [Results](<RESULTS.md>) · [Data and access](<DATA_SOURCES.md>) · [Model use](<MODEL_USE.md>)

## Student inference latency by batch size

| batch | ms/state | decisions/s |
|---|---|---|
| 1(per-state, original implementation) | 0.295 | 3,393 |
| 16 | 0.054 | 18,686 |
| 64 | 0.062 | 16,014 |
| **256(best)** | **0.051** | **19,489** |
| 1024 | 0.059 | 16,904 |

## Cold and cached feature-construction latency

|path| ms/state |relative to cold path|
|---|---|---|
| accessibility recomputation(original implementation, recomputed in each scenario/process) | 80.6 | 1× |
| cold first-pass feature construction(accessibility + alt routing, unique OD) | 100.9 | 1× |
| **disk cache hit + memoization(second scenario/second run, same as OD)** | **0.046**(including disk loading 0.01s/1000 records) | **2,478× faster than accessibility recomputation; 2,200× faster than cold first pass** |

## Measured and projected runtime by population size

|population size| pure Student inference(batches 256) |current end-to-end construction| optimized end-to-end construction(cache+batch processing) |
|---|---|---|---|
| 1k | 0.05 s(measured) | 353 s(E3 measured) | ≈ 0.1 s features+inference(measured component)+ leg/XML(not measured) |
| 10k | 0.51 s(measured) | 3,063 s(E3 measured) | ≈ 1 s features+inference(measured component)+ leg/XML(not measured; upper bound=E3 residual 19.3 min) |
| 50k | 2.6 s(measured projection: 0.051 ms×50k) | 13,907 s(E3 measured) | ≈ 5 s features+inference + leg/XML |
| 100k | 5.1 s(measured projection) | 20,907 s(E2 measured) | ≈ 10 s features+inference + leg/XML |
| 1M | **51 s(measured: encode 58.2 s + forward 1.03 s)** | ≈ 58 h(**estimated**: E2 measured 100k=5.8 h linear projection ×10) | ≈ 46 s features+inference(**estimated**: 0.046 ms×1M, prerequisite OD precomputed; cold first pass reported separately ≈28 h)+ leg/XML |
