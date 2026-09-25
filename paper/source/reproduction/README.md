# Accompanying research materials

All paths are relative to the source-package root. This index describes included materials; it is not a claim that participant-level data or the full experimental repository is public.

| File | Contents and scope |
|---|---|
| `reproduction/input_fields.csv` | Original machine-readable field definitions: 9 categorical, 17 global numerical and 12 per-alternative numerical fields. Availability and mode identity remain separate. |
| `reproduction/accessibility_teacher_prompt.txt` | Supplied accessibility-aware system/user prompt, version `teacher_s8_accessibility_v0.1`. Not a log of historical model backends or individual respondent payloads. |
| `reproduction/s9_adaptation_specification.json` | Original descriptive adaptation manifest. The manuscript display name is SA-Student; the archival filename and keys remain unchanged. Not a drop-in executable training configuration. |
| `data_tables/figure3_response_timing.csv` | Supplied response/timing summaries for two distinct model-selection experiments. Embedded source-table names refer to the earlier manuscript numbering. |
| `data_tables/figure4_response_bars.csv` | The 78 original full-sample model/contrast summaries used for Figure 4, at supplied precision. |
| `data_tables/teacher_decomposition_published_precision.csv` | Signed differences, absolute-bias summaries and interval conventions at archived reporting precision; not respondent-level Teacher vectors. |
| `data_tables/helsinki_run_pairs_published_precision.csv` | The 70 archived paired-run summaries. Repeated modular timing rows do not represent new independent choice fits. Not person-level events or raw probabilities. |
| `generated/narrative/` | Other existing numerical summaries in LaTeX, plus a new study-design table. |
| `figures/narrative/` | Revised publication figures and available editable sources. |

## Provenance retained from the uploaded package

The uploaded README recorded the private study repository `AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research` and commit `de906ac8a3efae692797bd08e340375107b8e52a`. Its recorded references were:

| Referenced upstream item | Recorded identifier / role |
|---|---|
| `src/traveler_distillation/student/features.py` | Base fields; blob `4c5012786588eb74c699669e76e60954e5c8e7f5` |
| `src/traveler_distillation/accessibility/accessibility_features.py` | Additional PT fields; blob `f6fce6ae14eff77644c59697b38b5cd80accdc48` |
| `configs/generation_v0_1.yaml`, `configs/joint_sampling.yaml` | Construction terminology; manuscript counts come from supplied summaries |
| `scripts/eval_s5_joint.py` | Interaction contrast; blob `b05f88e984a0a6f0cd9935da445f6afae8522efb` |
| `scripts/eval_s8_accessibility.py` | Feasibility measure; blob `3b734627100468227dc413d22a427e4ce5ebe65a` |
| `src/traveler_distillation/teacher/prompts_s8.py` | Accessibility prompt; blob `7c018f7fc0dc1b3c59ea364368adbdaccf86d1a4` |

The supplied README also attributed the Helsinki CSV to archived LaTeX run-pair tables at manuscript-history commit `fa50b42`. These are **inherited provenance statements**, not accesses or independent verification performed in the current editing session. The current revision preserves the four numerical CSVs, prompt, field dictionary and adaptation JSON byte-for-byte.

## Scope and outstanding records

Individual survey records, original deployed instruments, complete historical Teacher calls, fitted weights, the predecessor's complete training history and full simulation events are not supplied here. Complete parsing/retry rules and the routing treatment of all zero-feasible-mass or substitution cases also remain outside this manuscript package. Supplementary Section S7 reports these boundaries. No Teacher requests, training runs or transport simulations were performed for this revision.
