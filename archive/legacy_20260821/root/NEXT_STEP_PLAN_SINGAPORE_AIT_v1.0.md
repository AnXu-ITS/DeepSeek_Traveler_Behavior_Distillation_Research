# Research blueprint and staged Singapore plan

The research plan connects generated traveler/context states, repeated numerical Teacher targets, compact behavior models, response diagnostics and transport simulation. It separates model development from frozen-model supply experiments and proposes staged execution, scale and scenario checks. These are historical plans rather than evidence that every proposed milestone was completed. Current results, model choices and access limitations are summarized in the main project documentation.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/archive/legacy_20260821/root/NEXT_STEP_PLAN_SINGAPORE_AIT_v1.0.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Retained English technical listings

```text
network.xml
```


```text
data/singapore/gtfs/raw/singapore-gtfs.zip
data/singapore/gtfs/README_snapshot.md
data/singapore/gtfs/source_metadata.json
data/singapore/gtfs/checksum.sha256
```


```text
Student decision
    ↓
MATSim
    ↓
observed congestion / travel time
    ↓
context update
    ↓
Student re-decision
    ↓
new population demand
    ↓
MATSim
```


```text
K=3
K=5
K=7
```


```text
Teacher pairwise noise:
K=3 > K=5 > K=7
```


```text
fare ↑      → PT ↓
parking ↑   → car ↓
rain ↑      → walk/bike ↓
PT delay ↑  → PT ↓
```


```text
data/
  singapore/
    osm/
      raw/
      clipped/
      network/
    gtfs/
      raw/
      processed/
      README_snapshot.md
      source_metadata.json
      checksum.sha256
    population/
    scenarios/

outputs/
  singapore_smoke/
  singapore_population/
  singapore_feedback/
  baselines/
  teacher_noise/
  efficiency/
  external_validation/

reports/
  FINAL_DEVELOPMENT_BASELINE.md
  SINGAPORE_DATA_AUDIT.md
  SINGAPORE_NETWORK_VALIDATION.md
  SINGAPORE_SCENARIO_REPORT.md
  BASELINE_COMPARISON.md
  TEACHER_NOISE_CAUSAL_REPORT.md
  EFFICIENCY_REPORT.md
  EXTERNAL_PLAUSIBILITY_REPORT.md
  FINAL_PAPER_EVIDENCE_MATRIX.md
```
