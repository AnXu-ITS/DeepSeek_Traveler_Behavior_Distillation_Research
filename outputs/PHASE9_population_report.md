# Prototype synthetic-population experiment

This early experiment samples heterogeneous synthetic travelers, maps their attributes to the Student state schema and constructs simulation demand. It is a development-stage population/adapter check. Its synthetic network and mode representation differ from the later real-geometry supply experiments and current Helsinki response-preservation study.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/PHASE9_population_report.md).

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Student and executed mode shares by scenario

|scenario| bike | car | pt | walk |
|---|---|---|---|---|
| baseline (student) | 0.193 | 0.246 | 0.500 | 0.061 |
| baseline (executed) | 0.193 | 0.246 | 0.500 | 0.061 |
| rain (student) | 0.001 | 0.396 | 0.557 | 0.046 |
| rain (executed) | 0.001 | 0.396 | 0.557 | 0.046 |
| fare_surge (student) | 0.168 | 0.316 | 0.457 | 0.059 |
| fare_surge (executed) | 0.168 | 0.316 | 0.457 | 0.059 |
| combined (student) | 0.000 | 0.396 | 0.554 | 0.050 |
| combined (executed) | 0.000 | 0.396 | 0.554 | 0.050 |

## Scenario mode-share changes from baseline

|scenario| bike | car | pt | walk |
|---|---|---|---|---|
| rain | -19.2pp | +15.0pp | +5.7pp | -1.5pp |
| fare_surge | -2.5pp | +7.0pp | -4.3pp | -0.2pp |
| combined | -19.2pp | +15.0pp | +5.4pp | -1.2pp |

## Average leg and trip durations by scenario

|scenario| avg leg | avg trip |
|---|---|---|
| baseline | 6098.9 | 9099.6 |
| rain | 4999.6 | 8959.2 |
| fare_surge | 5520.8 | 9010.0 |
| combined | 4999.6 | 8959.2 |

## Population-scenario experiment command

```powershell
.venv\Scripts\python.exe scripts\run_population_experiment.py --checkpoint outputs/student_v0_3_c/checkpoints/best.pt --num-personas 1000 --trips-per-persona 2 --scenarios baseline,rain,fare_surge,combined --output data/population_experiment
```
