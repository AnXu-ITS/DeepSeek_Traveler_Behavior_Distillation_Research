# Transit execution validity checks

This historical check traces assigned PT demand into route itineraries and simulated transit events. Boarding events, unique boarding people and completed journeys must be counted separately. Transit availability and success of the bounded route search are also distinct. Current execution semantics and response gaps are given in the main project guide.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/singapore_phase_b5/pt_validity_v3/pt_validity.md).

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Transit routing validity by trip direction

|direction| intended PT | routed PT | direct PT | transfer PT | fallback | success rate |
|---|---|---|---|---|---|---|
| outbound | 455 | 449 | 346 | 103 | 6 | 0.9868 |
| return | 449 | 437 | 341 | 96 | 12 | 0.9733 |
|**total**| 904 | 886 | 687 | 199 | 18 | **0.9801** |
