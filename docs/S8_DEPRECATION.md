# S8 deprecation record

S8 is deprecated because the accessibility pipeline used road-link free-flow speeds to compute walking and cycling times. The historical audit found a median effective active-mode speed of 29.1 km/h and no state in which PT was faster than walking in the 338-state erroneous dataset. S9 rebuilt the accessibility data with walking at 1.34 m/s and cycling at 4.17 m/s, retaining car free-flow routing. Its corrected dataset has 336 states and 1,502 valid Teacher queries. S8 weights and original tags remain historical evidence; they are not a current model recommendation. The historical Singapore adaptation used teleported active modes at 1.39/3.9 m/s; the later Helsinki network-speed experiment has a different execution specification.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/S8_DEPRECATION.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<RESEARCH_DESIGN.md>) · [Training](<TRAINING.md>) · [Results](<RESULTS.md>) · [Data and access](<DATA_SOURCES.md>) · [Model use](<MODEL_USE.md>)

## Historical numerical table 1

|items| S8 | S9 |
|---|---|---|
| walk speed | link freespeed | 1.34 m/s |
| bike speed | link freespeed | 4.17 m/s |
| pt access/egress walking | link freespeed | 1.34 m/s |
| MATSim walk/bike | network modes | teleported(walk 1.39 / bike 3.9 m/s) |
|dataset| 338 states(biased) | 336 states(rebuild+relabeling 1,502 requests) |
