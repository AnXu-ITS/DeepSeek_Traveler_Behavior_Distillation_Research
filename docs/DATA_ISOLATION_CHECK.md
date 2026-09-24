# Data isolation and frozen-artifact safeguards

The reference pipeline loads frozen checkpoints, vocabularies and normalization without fitting to deployment populations. Generated smoke/development outputs must have separate paths and identifiable seeds. Route caches must bind network, schedule, routing rules, departure time and code identity; an unversioned historical cache is not reusable evidence. S8 legacy data and S9 corrected data must be distinguished explicitly despite similar filenames. The historical review lists nine contamination/overwrite risks; current model use and material scope are described in the linked guides.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/DATA_ISOLATION_CHECK.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<RESEARCH_DESIGN.md>) · [Training](<TRAINING.md>) · [Results](<RESULTS.md>) · [Data and access](<DATA_SOURCES.md>) · [Model use](<MODEL_USE.md>)
