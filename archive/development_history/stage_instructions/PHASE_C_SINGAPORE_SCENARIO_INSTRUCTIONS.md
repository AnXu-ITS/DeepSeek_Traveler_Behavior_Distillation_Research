# Singapore Phase C scenario experiment

Phase C compares a baseline, heavy rain, higher PT fare, perceived PT delay, road disruption and rain plus delay on one synthetic Singapore population. The later corrected S9 run uses 10,000 travelers, effective flow/storage factors of 0.3 and fixed supply with one MATSim iteration. Mode assignment, unique people boarding PT, total boarding events, completed legs and completed journeys have different denominators. Earlier S8 scenarios are deprecated; source dates and model identifiers determine the applicable result.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/stage_instructions/PHASE_C_SINGAPORE_SCENARIO_INSTRUCTIONS.md).

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Scenario-run command and experiment sequence

```bash
python scripts/singapore/run_phase_c.py --scenarios C0_baseline,C1_heavy_rain \
    --num-agents 10000 --checkpoint releases/s8_supply_aware_v1/checkpoint/model.pt \
    --output outputs/singapore_phase_c
```


```text
C0 baseline → C1 heavy rain → C2 PT fare increase → C3 transit delay
            → C4 road disruption → C5 joint scenario
```
