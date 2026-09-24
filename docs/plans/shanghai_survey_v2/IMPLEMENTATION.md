# Shanghai questionnaire design history

This is an earlier questionnaire-design or implementation record. It documents the development of respondent attributes, scenario cards, display logic and data mappings. It is not evidence that every proposed question or numerical setting was administered. The final field instrument has 23 required single-choice questions, including ten scenario tasks. Use the complete English field questionnaire and current sample-flow/result documentation for the paper.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/plans/shanghai_survey_v2/IMPLEMENTATION.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../RESEARCH_DESIGN.md>) · [Training](<../../TRAINING.md>) · [Results](<../../RESULTS.md>) · [Data and access](<../../DATA_SOURCES.md>) · [Model use](<../../MODEL_USE.md>)

For the administered instrument, use the [complete English Shanghai questionnaire](<../../surveys/SHANGHAI_QUESTIONNAIRE.md>) and [survey results](<../../surveys/RESULTS.md>) instead of these earlier design drafts.

## Historical numerical table 1

| cardID |condition| total driving duration/cost | PTtotal duration/cost | PTentering stop/waiting/in vehicle/transfer/leaving stop |number of transfers|
|---|---|---|---|---|---:|
| B0 |clear-weather baseline| 20minutes/15CNY | 35minutes/4CNY | 7/5/16/0/7 | 0 |
| W1 | heavy rain, time and cost held fixed | 20minutes/15CNY | 35minutes/4CNY | 7/5/16/0/7 | 0 |
| D1 | PTadditional delay15minutes | 20minutes/15CNY | 50minutes/4CNY | 7/20/16/0/7 | 0 |
| WD1 | heavy rain＋PTdelay15minutes | 20minutes/15CNY | 50minutes/4CNY | 7/20/16/0/7 | 0 |
| F1 | PTfare4→6 | 20minutes/15CNY | 35minutes/6CNY | 7/5/16/0/7 | 0 |
| P1 | parking10→30 | 20minutes/35CNY | 35minutes/4CNY | 7/5/16/0/7 | 0 |
| R1 | road disruption, additional driving cost20minutes | 40minutes/15CNY | 35minutes/4CNY | 7/5/16/0/7 | 0 |
| A_WALK | PTeach access walk increases by5minutes | 20minutes/15CNY | 45minutes/4CNY | 12/5/16/0/12 | 0 |
| A_WAIT | initial waiting for normal service increases by10minutes | 20minutes/15CNY | 45minutes/4CNY | 7/15/16/0/7 | 0 |
| A_TRANSFER | 1transfers, total transfer time10minutes | 20minutes/15CNY | 45minutes/4CNY | 7/5/16/10/7 | 1 |

## Retained English technical listings

```powershell
.venv/Scripts/python.exe -B docs/plans/shanghai_survey_v2/survey_adapter.py --respondents exports/respondents.csv --output-dir outputs/shanghai_survey_v2_run1
```


```powershell
.venv/Scripts/python.exe -B docs/plans/shanghai_survey_v2/build_package.py
.venv/Scripts/python.exe -B docs/plans/shanghai_survey_v2/validate_package.py
```
