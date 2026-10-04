# Reproduction guide and material inventory

This guide describes the baseline evidence package from 24 September 2026 and its reproduction scope. The [3 October manuscript support package](manuscript-support/2026-10-03/README.md) adds the later thirty-persona reference comparison, forty-run physical-supply experiment and offline summary checks; [records R1–R5](manuscript-support/2026-09-30/README.md) provide extended supporting material. The repository supports local inference, inspection of retained results and re-evaluation of the released controlled models. It is **not a claim that every experiment can be rerun from a fresh clone without additional data or runtime setup**.

## Material inventory

| Category | Included location | Scope / remaining requirement |
|---|---|---|
| Code | `src/`, `reference_pipeline/`, `scripts/`, `scripts/revision_20260921/` | Current and later experimental implementations; historical source snapshots accompany controlled fits |
| Configurations | `configs/matched_response.yaml`; per-run `run.json`; release configs | Full controlled recipe and model metadata; some simulation scripts retain author-environment assumptions |
| Environment | `pyproject.toml`, `requirements-research.txt`, per-run metadata | Dependencies and recorded versions; not a single fully locked environment for every historical stage |
| Random seeds | Per-run metadata, configurations and tables | Controlled training and assignment seeds are distinct |
| Data | `outputs/matched_response_v1/bundle/`; selected pre-existing supply/target records | Prepared synthetic benchmark included; raw human records and large supply/event files retained separately |
| Splits | Prepared bundle manifest and endpoint/pair files | 28/6/6 persona split; 1,839/365/437 endpoint counts |
| Models | `releases/`; controlled `train/`; modular baseline records | SA-Student, predecessors and selected experimental models |
| Checkpoints | `model.pt`, controlled `best.pt`, modular `.pt` files | Selected weights included unchanged; not every transient optimizer state |
| Logs | Controlled histories/status/run files | Compact histories included; per-epoch sampled-unit logs and full event logs are not duplicated |
| Statistics | `evidence/paper_20260924/`; controlled evaluations; baseline outputs | Aggregate tables, metrics, predictions on synthetic states and analysis scripts |
| Figure source data | Four manuscript CSVs; `docs/assets/` | Current figure values and supplied visual assets; Figure 1 PDF and rendered preview |
| Third-party assets | Existing supply metadata and data-source guide | External software/feed terms apply; no new license grant or change of repository visibility |

See the [file manifest](../evidence/paper_20260924/artifact_manifest.json) for source and destination SHA-256 values. Some newly copied text metadata replace author-machine absolute paths with repository-relative paths; those transformations are recorded. Numerical source data and checkpoint bytes remain unchanged. The [verification report](../evidence/paper_20260924/verification.json) distinguishes executed checks from retained historical claims.

## 1. Check local inference

```bash
python -m pip install -e .
python -m pip install -r requirements-research.txt
python examples/predict_released_student.py
```

This requires no Teacher API, Java or network supply. The example uses a synthetic held-out state and the frozen SA-Student checkpoint. It reports 24,562 parameters and validates normalized probabilities and the ±60-minute departure bound.

## 2. Re-evaluate an archived controlled checkpoint

```bash
python scripts/reproduce_archived_checkpoint.py --run outputs/matched_response_v1/train/soft_kl_seed42 --split test --output outputs/reproduced_soft_kl_seed42
```

The helper verifies the source archive against its training record, extracts it to a temporary directory and invokes the original evaluator. This avoids altering the working source or weakening the evaluator's source-fingerprint check. The prepared bundle and selected checkpoint retain their original content checks.

Historical fingerprints use platform-specific path separators as well as exact source bytes. The command has been checked on Windows, the original research platform. Cross-platform re-evaluation may require an explicitly versioned portability change; do not bypass a fingerprint failure and report it as an exact historical reproduction. Normal Git line-ending conversion can also make the current source fingerprint differ, which is why the archived-source helper is provided.

## 3. Train a new controlled run

Use the commands in [TRAINING.md](TRAINING.md). The prepared bundle avoids re-querying the Teacher. New output directories prevent overwriting the retained fits. Training smoke and full training are different modes; a smoke run is not evidence for the published results.

## 4. Reproduce aggregate survey and simulation analyses

The repository provides result CSVs, manuscript tables and analysis implementations. Complete respondent bootstrap or contemporary Teacher re-evaluation additionally requires participant-linked inputs, which are not included in this update. The raw questionnaires and responses are unchanged in the authors' retained workspace. Reading an aggregate CSV is not an independent rerun of the respondent-level analysis.

The 140-run Helsinki experiment additionally requires prepared Helsinki supply, the fixed population and the Java/MATSim runtime. Its complete event directories occupy approximately 57.5 GB in the retained workspace; this update adds run-level/paired summaries, protocol information and analysis code rather than that entire archive. The summary evidence contains 140 completed runs and 70 baseline/delay pairs. The article's original availability text predates this repository expansion.

No new Teacher request, full training campaign or MATSim simulation is part of this repository documentation update. Verification includes actual local inference and an archived-checkpoint re-evaluation; the report states their exact scope.

## Claims and material coverage

| Claim | Direct repository evidence | Reproduction coverage |
|---|---|---|
| Controlled response comparison | Prepared bundle, 12 selected neural fits, test predictions, comparison metrics and MNL-S records | Re-evaluation supported; retained targets avoid backend drift |
| Modular timing comparison | Selected timing models, per-seed metrics, training protocol and source-breakdown CSV | Compact model/metric evidence included |
| Stated-choice response agreement | Complete English instruments, aggregate scores and 78-row response table | Aggregate inspection supported; participant-level bootstrap needs retained data |
| Contemporary Teacher decomposition | Campaign summary, aggregate decomposition, prompt/analysis metadata | Aggregate inspection supported; participant-linked request payloads withheld |
| Predicted-to-simulated response gap | 140-run aggregate ledger, 70-pair manuscript CSV, driver and event-analysis code | Summary inspection supported; complete fresh simulation needs large supply/runtime materials |
| Figure 1 and manuscript context | Original figure PDF, rendered preview and manuscript PDFs | Included, with no new scientific results generated |

This is a materials assessment, not venue-compliance certification or a claim that a separate formal result-acceptance workflow has been completed. Missing raw records in this clone are explicitly distinguished from records retained by the authors.

[Data terms and access](DATA_SOURCES.md) · [Model use](MODEL_USE.md) · [Project home](../README.md)

## Verify this repository edition

Run `python scripts/check_repository_docs.py` to check Markdown language, local links, code fences and the copied-evidence hashes without changing research outputs.

## Current manuscript additions (3 October 2026)

See the [compact support package](manuscript-support/2026-10-03/README.md) and [extended records R1–R5](manuscript-support/2026-09-30/README.md). The package verifies retained summary arithmetic; it does not rerun model training, respondent analyses, Teacher acquisition or MATSim.
