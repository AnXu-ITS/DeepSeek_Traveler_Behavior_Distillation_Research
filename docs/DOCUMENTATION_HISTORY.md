# Documentation and research history

The current repository navigation follows **Beyond Static Imitation: Preserving LLM-Derived Traveler Responses for Transport Simulation**, manuscript snapshot dated 24 September 2026. The preceding remote revision was `de906ac8a3efae692797bd08e340375107b8e52a`.

## Current reader-facing documents

The [project overview](../README.md), [research design](RESEARCH_DESIGN.md), [data sources](DATA_SOURCES.md), [training guide](TRAINING.md), [result synthesis](RESULTS.md), [model-use guide](MODEL_USE.md), [execution workflow](EXECUTION_WORKFLOW.md) and [survey collection](surveys/README.md) were written against the current manuscript and retained evidence.

The questionnaires are English documentation versions of the actual retained instruments. They preserve task conditions and choices, while consolidating repeated bilingual lines or repeated option lists. They do not replace or alter the administered forms or raw answers. Earlier Shanghai survey designs remain historical design drafts.

## Earlier reports

Existing Chinese Markdown documents have been reorganized into English historical editions in the review clone. These are editorial consolidations, not line-by-line translations: they preserve the purpose and interpretation of each stage, selected original numerical tables and English code listings, with a direct link to the complete historical version. Historical reports retain their original model names, experimental stages and numbers; translation does not make an old result current. The 189 retained historical numerical tables use a reviewed English terminology glossary; numerical literals are retained. For the current scientific interpretation, use the authored guides above and the manuscript tables.

The original-language documents remain recoverable at the preceding Git commit. Frozen model weights, normalization, configurations and numerical release artifacts are not retrained or re-estimated by this update. Where release documentation is translated, its bytes necessarily differ from the historical freeze; original release tags and historical checksum files remain the authority for the original package. The new file manifest describes this documentation/evidence edition separately and does not overwrite the historical checksum manifests.

| Historical stage | How to interpret it now |
|---|---|
| Early v0.x / S1–S6 | Development of targets, losses and prototype evaluations |
| S7-W3 | Frozen generic behavioral predecessor |
| S8 | Deprecated because active-mode speed inputs were wrong |
| S9 | Frozen supply-adapted model, named SA-Student in the current paper |
| Singapore Phase A/B/B.5/C and E1–E5 | Earlier supply, scale, timing and transfer experiments with their own populations and denominators |
| Matched response v1 | Controlled objective comparison with common endpoint information and persona splits |
| September 21 revision experiments | Modular timing, full stated-response analysis, contemporary Teacher comparison and 140-run Helsinki execution study |

Earlier claims about a model being “healthy,” a gate passing, a stop rule, or a planned feedback loop are historical operational judgments. They are not additional evidence of human calibration, causal validity or a completed behavioral feedback loop in the current article.

## What changed in this repository edition

- Added an English research overview, audience-specific navigation and the manuscript's Figure 1.
- Added the requested traveler-input → Student → demand-plan → routing → MATSim → events workflow.
- Added English questionnaires, complete aggregate response tables, training/model documentation and provenance.
- Added prepared synthetic benchmark data, selected later fits, aggregate revision results and corresponding code.
- Added a minimal released-model inference example and an archived-source checkpoint evaluator.
- Retained repository visibility, original survey records, original checkpoints and historical numerical evidence.

The local review clone is a normal Git working repository that can be inspected and synchronized with the named GitHub repository. It is not a bare `git clone --mirror` directory and does not rewrite the original workbench's uncommitted changes.
