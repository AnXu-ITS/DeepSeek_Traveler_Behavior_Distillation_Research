# Reference pipeline validation record

The reference implementation is checked against the production Student and routing implementation on fixed inputs. The checks distinguish probability differences, identical mode decisions, cached/uncached construction and MATSim execution. Singapore and Helsinki validation conditions use their own supply snapshots and cache states. This is deployment parity evidence rather than a new human-behavior validation or proof of suitability for arbitrary cities.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/REFERENCE_PIPELINE_VALIDATION.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<RESEARCH_DESIGN.md>) · [Training](<TRAINING.md>) · [Results](<RESULTS.md>) · [Data and access](<DATA_SOURCES.md>) · [Model use](<MODEL_USE.md>)

## Historical numerical table 1

|check|result|
|---|---|
| student_mode consistent | **200/200** |
| outbound_mode consistent | **200/200** |
| return_mode consistent | **200/200** |
| departure_min / departure_shift_min(2 decimal places)consistent | **200/200** |
| home_node / dest_node consistent | **200/200** |
| outbound / return leg info(route_links, pt fallback reasons and related fields)consistent | **200/200** |
| batch vs per-state `decide()`: mode | **200/200** |
| batch vs per-state: max \|Δshift\| | **3.81e-6 min**(floating point GEMM noise) |
| batch vs per-state: max \|Δprob\| | **1.79e-7** |
| MemoizedEncoder vs production `extractor.encode` | **200/200 field-by-field equality** |
| population.xml SHA256 | **equal**(`aa57754a2e51e185…`) |
| config.xml SHA256 |**equal**|
| adapter_manifest.json(JSON semantics) |**equal**|
| original path build / Reference build | 72.0 s / 77.5 s(same order of magnitude; Reference additional persistence 2,316 records SP routing) |

## Historical numerical table 2

| | run 1(cold) | run 2(warm) |
|---|---|---|
| total build | 153.5 s | **15.6 s**(9.8×) |
| feature preparation | 105.6 s | **0.01 s** |
| student inference(batch 256) | 0.02 s | 0.02 s |
| plan + XML | 47.9 s | 15.6 s |
| cache requests | 8,608(15 hit) | **8,608(100% hit)** |
| cache entries(acc / tt / sp) | 500 / 1,485 / 6,608 |as left|
| cache size | 3.64 MB |as left|
| estimated routing saved | — | **139.4 s** |
| mode distribution | bike 176 / car 146 / pt 114 / walk 64 |**identical**|

## Historical numerical table 3

|check|result|
|---|---|
| persona_id / trip_id / student_mode / outbound_mode / return_mode / home_node / dest_node | **10,000/10,000 all identical** |
| departure_shift_min / departure_min | **9,999/10,000**(1 cases 2-dp rounding-boundary decision flip: -5.8↔-5.79, 0.01 min; versus Singapore D6 comparable floating-point noise) |
| mode distribution | **versus frozen E5 C0 identical**: pt 2,304 / bike 3,458 / walk 1,383 / car 2,855 |
| **population.xml** | **and E5 original pipeline artifact bytes identical(SHA256 `4b797a3ef98c1adfcfed…` equal)** |
| original pipeline build(freeze record) | **9,507.0 s**(factory mean 606 ms/state, Helsinki road network 290k nodes/638k links) |
| Reference cold build(including cache construction) | **10,491.8 s**(+10.4%: persistence 145,092 routing calls; acc 9,999 / tt 28,281 / sp 106,812, cache 55.98 MB) |
| **Reference warm build(cache reuse)** | **103.2 s** → **relative to original pipeline 92.1×; relative to cold 101.7×** |
| warm cache | hit rate **100%**(146,812/146,812), estimated routing saved **10,735 s** |

## Historical numerical table 4

|city| original pipeline build | Reference cold(cache construction) | Reference warm | **vs original pipeline** | **cold→warm** |share of cold-construction cost removed|decision parity| population.xml |
|---|---|---|---|---|---|---|---|---|
| Singapore C0(10k) | 3,063 s(E3 measured) | 3,223.6 s | 469.5 s | **6.5×** | **6.9×** | 85.4% | 10,000/10,000(2 boundary decision flip) | versus original pipeline byte-identical(N=200 fixture validation) |
| Helsinki C0(10k) | 9,507.0 s(E5 freeze record) | 10,491.8 s | **103.2 s** | **92.1×** | **101.7×** | 99.0% | 10,000/10,000(1 boundary decision flip) | **and E5 original artifact bytes identical(SHA256 equal)** |
