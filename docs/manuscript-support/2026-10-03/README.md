# Current manuscript: compact evidence and verification

**Beyond Static Imitation: Evaluating LLM-Derived Traveler Responses for Transport Simulation**

An Xu, Chengbo Zhang, Zekai Jin, Yimin Zhao and Yunfei Yin

This directory provides the compact numerical inputs used by the manuscript checked on 3 October 2026. It complements [records R1–R5](../2026-09-30/README.md). Files were copied from the manuscript package; no new Teacher requests, training runs, respondent analyses or simulations were conducted for this release.

| Manuscript evidence | Supplied files | Reproduction scope |
|---|---|---|
| Thirty-persona reference comparison | `data_tables/revision20260925/formal_analysis.json`, `formal_paired_contrasts.csv`, `formal_descriptive_summary.csv` | Recompute the primary mean and persona-bootstrap interval from retained synthetic-persona differences |
| Acquisition and prespecification | `acquisition_protocol.json`, `formal_launch_plan.json`, `service_metadata_diagnostic.json` in the same directory | Inspect recorded model, provider, settings and sample rule; no service credentials or human payloads |
| Controlled learning diagnostics | `weight_selection_summary.csv`, `controlled_paired_comparisons.json`, `delay_only_test_summary.json`, `gradient_summary.json`, `repeat_disjoint_summary.json` | Inspect the retained model/seed summaries; these are not replacement checkpoints |
| Helsinki physical-supply experiment | `physical_run_metrics.json`, `physical_verification.json`, `physical_verified_contrasts.json`, `stagewise_summary.json` | Verify forty model–arm–seed summaries and stage arithmetic; does not rerun the network |
| Five-seed decomposition | `data_tables/revision20260926/five_seed_stage_decomposition.csv` and `generated/revision20260926/table_five_seed_stages.tex` | Inspect exact and publication-precision values, including separate assignment-seed SD |
| Latency evidence | `generated/motivation20261001/table_inference_motivation.tex` and [R4](../2026-09-30/R4_compute.md) | Recorded remote and local measurements; different successful-request denominators; no fresh benchmark |

## Offline verification

Install NumPy in an existing Python environment and run from the repository root:

```sh
python docs/manuscript-support/2026-10-03/verify_summaries.py
```

The script recomputes the primary thirty-persona contrast and its 10,000-resample interval using the retained seed, verifies the forty-run population/completion counts, and recomputes the six five-seed response decompositions. [verification.json](verification.json) records the output: primary difference −0.00319918695, 95% interval [−0.00486298883, −0.00162214324]. All forty runs have equal assigned, routed and boarded PT counts.

The existing [experiment source and archive](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/tree/ba7c5d304b46b46f8f74c451cda2825a3916bf6d) remain separately versioned. Fresh fitting and MATSim execution require the inputs and runtime described there. Reading or reaggregating this compact package is not a complete experimental rerun.

## Reference identity and historical text

The retained acquisition protocol identifies **`deepseek-v4.1-flash` through OpenCode Go**, and `formal_analysis.json` labels this result as agreement with a later Flash reference. It does not identify the new-persona campaign as an official-direct V4 Pro collection. The archived Pro supervision and same-task survey comparison are separate collections. Manuscript or historical support prose describing all three collections as the same Pro service is inconsistent with these retained records and requires correction or replacement evidence. No acquisition record was rewritten during this release.

R1–R5 contain preserved LaTeX excerpts with historical numbering and availability wording. The current manuscript places the four additional diagnostics in Supplementary Section S6. Public availability of this repository does not release participant workbooks, participant-linked model requests, full event archives or third-party assets under a new license.

## Author declarations

The [current declarations](DECLARATIONS.md) contain the author-supplied CRediT and ethics statements, HIT-specific funding, competing interests and AI disclosure limited to Figure 1 for figure preparation. Historical acquisition protocols remain unchanged.
