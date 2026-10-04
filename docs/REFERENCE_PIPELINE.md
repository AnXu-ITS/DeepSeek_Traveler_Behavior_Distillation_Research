# Reference MATSim integration pipeline

The reference pipeline reads population CSV and scenario YAML, validates fields, computes supply attributes, loads frozen S9 once, predicts in batches, routes selected modes and writes MATSim population/plans. Optional execution requires compatible Java/MATSim and prepared transport supply. Feature order, availability, vocabulary and normalization follow the checkpoint. Caches bind the supply/routing configuration; deployment does not query the Teacher or retrain the Student. Consult the current model-use and execution guides for the recommended entry points and failure semantics.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/REFERENCE_PIPELINE.md).

[Current research](<RESEARCH_DESIGN.md>) · [Training](<TRAINING.md>) · [Results](<RESULTS.md>) · [Data and access](<DATA_SOURCES.md>) · [Model use](<MODEL_USE.md>)

## Cold-cache and warm-cache build timings

|population size| first build(cold cache) | second build(warm cache) | MATSim execution |
|---|---|---|---|
| 25 trips(sample) | ~7 s | ~1 s | ~115 s(including events parsing ~43 s) |
| 500 trips | 153 s | 16 s(9.8×) | not measured(and 10k comparable execution mode) |
| 10,000 trips(Singapore C0) | 3,224 s(≈54 min; original pipeline 3,063 s) | 470 s(≈7.8 min, 6.9×, hit rate 100%) | Phase C measured 137 s(E3) |
| 10,000 trips(Helsinki C0, second city) | 10,492 s(≈2.9 h; original pipeline 9,507 s) | **103 s(92×, hit rate 100%)** | E5 executed(exit 0) |

## Reference-pipeline command and software citation

```powershell
python scripts/run_reference_pipeline.py --config configs/reference_example.yaml --run-matsim
```


```bibtex
@software{traveler_student_matsim_reference_pipeline,
  title = {Reference MATSim Integration Pipeline for the Distilled Traveler Agent},
  note = {Research artifact accompanying the traveler-behavior distillation study.},
  year = {2026},
}
```
