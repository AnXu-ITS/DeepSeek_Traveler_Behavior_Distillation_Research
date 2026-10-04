# Historical author-workflow source

Five source files from the 24 September evidence inventory are preserved here byte-for-byte from [commit 3c41e7e](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/tree/3c41e7e1826d53c74493910ad9b9eaf31f5f6657). Their original SHA-256 values remain in the [evidence manifest](../../../evidence/paper_20260924/artifact_manifest.json); `original_path` records the former location.

These files depend on the author's former local workspace, including unavailable validation tools and external instruction documents. They are provenance records, not supported execution entry points. Do not run them or use their success messages as fresh validation.

| Archived source | Maintained entry point |
|---|---|
| [Teacher-completion launcher](scripts/revision_20260921/complete_teacher.ps1) | [Environment-based launcher](../../../scripts/revision_20260921/complete_teacher.ps1) |
| [Prompt-pilot launcher](scripts/revision_20260921/complete_prompt_pilot.ps1) | [Environment-based launcher](../../../scripts/revision_20260921/complete_prompt_pilot.ps1) |
| [Resident-sensitivity analysis](scripts/revision_20260921/human_complete_resident_sensitivity.py) | [Configurable-output analysis](../../../scripts/revision_20260921/human_complete_resident_sensitivity.py) |
| [Closure finalization](cvpr_workspace/analysis/statistics/finalize_closure.py) | Retired; [supported verification scope](../../../docs/REPRODUCIBILITY.md#maintained-entry-points-and-historical-source) |
| [Closure sealing](cvpr_workspace/analysis/statistics/seal_closure.py) | Retired; [supported verification scope](../../../docs/REPRODUCIBILITY.md#maintained-entry-points-and-historical-source) |

No archived source, checkpoint, numerical result, acquisition record or original hash was regenerated for this maintenance change. The archive does not include the external instruction documents or credentials.
