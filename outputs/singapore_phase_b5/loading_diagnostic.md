# Network loading diagnostic

This diagnostic examines network loading, departures, congestion and completion under the historical supply and demand representation. Active-mode speeds and whether a mode is network-routed or teleported affect capacity and travel-time interpretation. A low stuck count alone does not establish faithful physical or behavioral simulation.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/singapore_phase_b5/loading_diagnostic.md).

[Current research](<../../docs/RESEARCH_DESIGN.md>) · [Training](<../../docs/TRAINING.md>) · [Results](<../../docs/RESULTS.md>) · [Data and access](<../../docs/DATA_SOURCES.md>) · [Model use](<../../docs/MODEL_USE.md>)

## Highest-flow road links and volume-to-capacity ratios

| rank | link | peak flow (veh/h) | capacity | V/C |
|---|---|---|---:|---:|
| 1 | `10695495432_10695495431` | 253 | 700 | 0.361 |
| 2 | `10695495431_10695495433` | 253 | 700 | 0.361 |
| 3 | `10695495428_10695495432` | 251 | 700 | 0.359 |
| 4 | `10666521289_5187353835` | 246 | 700 | 0.351 |
| 5 | `5187353835_10564385656` | 246 | 700 | 0.351 |
| 6 | `10695495433_10666521289` | 245 | 700 | 0.350 |
| 7 | `10695495438_10695495437` | 254 | 800 | 0.318 |
| 8 | `10695495437_10695495436` | 254 | 800 | 0.318 |
| 9 | `10695495436_10695495434` | 254 | 800 | 0.318 |
| 10 | `10695495435_10564385654` | 254 | 800 | 0.318 |
| 11 | `10564385657_10695495440` | 253 | 800 | 0.316 |
| 12 | `10695495440_10695495439` | 253 | 800 | 0.316 |
| 13 | `10695495439_10695495438` | 253 | 800 | 0.316 |
| 14 | `10695495434_10695495435` | 253 | 800 | 0.316 |
| 15 | `10569598629_10689717164` | 248 | 800 | 0.310 |
| 16 | `10689717164_10564385655` | 248 | 800 | 0.310 |
| 17 | `10564385655_10564385660` | 248 | 800 | 0.310 |
| 18 | `10564385660_10564385658` | 248 | 800 | 0.310 |
| 19 | `10564385658_10564385659` | 248 | 800 | 0.310 |
| 20 | `1840121849_8310679163` | 247 | 800 | 0.309 |
