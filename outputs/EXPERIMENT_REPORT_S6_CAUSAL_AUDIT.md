# S6 mechanism diagnostic study

S6 compares natural perturbations with states that change context labels, numerical travel attributes or both. These comparisons test what information Teacher and Student responses follow, and whether distillation preserves the observed mechanism. The contrasts are diagnostic model experiments; they are not field estimates of causal effects. Mechanism states later contribute a separate source to the controlled response benchmark.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Historical numerical table 1

| axis | model | E_natural | E_broken | E_mediator | R_shortcut | R_mediator |
|---|---|---:|---:|---:|---:|---:|
| transit_delay | Teacher | 0.3475 | 0.2071 | 0.2894 | 0.7299 | 0.9701 |
| transit_delay | C0_preS5 | 0.3935 | 0.1165 | 0.3230 | 0.3701 | 0.9003 |
| transit_delay | C1_S5 | 0.3732 | 0.1284 | 0.2848 | 0.3458 | 0.8425 |
| congestion | Teacher | 0.2533 | 0.1172 | 0.1978 | 0.5160 | 0.8892 |
| congestion | C0_preS5 | 0.2029 | 0.1567 | 0.1517 | 0.8119 | 0.7488 |
| congestion | C1_S5 | 0.1778 | 0.1710 | 0.1080 | 1.0206 | 0.6722 |
| parking_cost | Teacher | 0.4504 | 0.3139 | 0.2203 | 0.6871 | 0.5586 |
| parking_cost | C0_preS5 | 0.4591 | 0.4717 | 0.0226 | 1.0309 | 0.0525 |
| parking_cost | C1_S5 | 0.4535 | 0.4642 | 0.0236 | 1.0197 | 0.0588 |

## Historical numerical table 2

| axis | model | ΔP natural | ΔP broken | ΔP mediator |
|---|---|---:|---:|---:|
| transit_delay | Teacher | -0.1654 | -0.0975 | -0.1378 |
| transit_delay | C0_preS5 | -0.1940 | -0.0214 | -0.1615 |
| transit_delay | C1_S5 | -0.1835 | -0.0399 | -0.1424 |
| congestion | Teacher | -0.1120 | -0.0385 | -0.0949 |
| congestion | C0_preS5 | -0.0972 | -0.0699 | -0.0758 |
| congestion | C1_S5 | -0.0833 | -0.0740 | -0.0521 |
| parking_cost | Teacher | -0.2216 | -0.1526 | -0.1080 |
| parking_cost | C0_preS5 | -0.2279 | -0.2343 | 0.0111 |
| parking_cost | C1_S5 | -0.2256 | -0.2309 | 0.0111 |
