# Extended records for the streamlined manuscript supplement

**Beyond Static Imitation: Preserving LLM-Derived Traveler Responses for Transport Simulation**

This directory preserves material moved out of the journal supplement during the 30 September 2026 editing pass. It was extracted from the supplied `Overleaf_language_revision(1).zip`. No model was trained, API queried, survey re-estimated or transport simulation rerun during this migration.

## Extended records

| Record | Contents | What remains in the journal package |
|---|---|---|
| [R1 — Learning diagnostics](R1_learning.md) | Selected timing epochs, seed/source breakdowns and ancillary feasibility diagnostics | Architecture, objectives, selection rules, budgets, the primary comparison, intervention errors and repeated-target sensitivity |
| [R2 — Prompt pilots](R2_prompts.md) | Four-person prompt/metadata pilot and additional choice scores on Teacher subsets | The 24-person sampling rule, matching protocol, Teacher-response variation and response-discrepancy tables |
| [R3 — Singapore illustration](R3_singapore.md) | Synthetic scenarios underlying Figure 1, population-seed sensitivity, completion and journey summaries | The Figure 1 illustration and its synthetic/pre-routing qualification; the Helsinki experiment remains central |
| [R4 — Computation and costs](R4_compute.md) | Inference benchmarks, scale and reuse timings, token accounting and retrospective valuations | The measurements and timing boundaries supporting the approximately 0.35% inference share |
| [R5 — Provenance](R5_provenance.md) | Service-metadata subgroups, collection/version notes and the earlier material inventory | Distinct Teacher collections, key acquisition settings, backend-identification limits and access restrictions |

Complete human-response tables, task encodings, sample exclusions, consent/ethics statements, statistical definitions and the formal new-persona protocol remain in the compact journal supplement. The active-mode speed sensitivity, including its original figure, moves to article Appendix B rather than being removed.

## Preservation and interpretation

The record pages contain verbatim LaTeX excerpts and exact table source. Their `\label` and `\ref` names belong to the source snapshot; the shortened journal supplement is renumbered. [manifest.tsv](manifest.tsv) records source paths, line ranges, content hashes and the tables moved out of the Overleaf tree. Some core context is repeated next to extended tables so the records remain interpretable.

Original notes can mention other repository commits, local directories, participant records or event archives. Preserving those statements does not newly supply those artifacts or verify their current availability. No raw participant workbook, participant-linked API payload, credential or full MATSim event archive is added here. Historical pricing remains historical, and unknown charges have not been replaced by zero.

This change adds documentation on a separate branch. It does not replace released checkpoints, alter existing experiment outputs, grant licenses or change repository visibility. The repository remains access-controlled. Reviewer access must be arranged with the corresponding author; a private repository link alone is not an anonymously accessible journal supplement.
