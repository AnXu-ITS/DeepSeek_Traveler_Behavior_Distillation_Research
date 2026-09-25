# Response-fidelity experiments: 25 September 2026

The author authorized the experimental sequence and asked to stop before incorporating new results into the manuscript. Independent new-human repair validation is deferred because no new respondents are available. Historical questionnaires and responses are unchanged.

The experiment report and completion ledger will be published in [`evidence/experiments_20260925`](../evidence/experiments_20260925). The manuscript source synchronized earlier in this revision does not contain these new results.

## Experiments and interpretation

- Twenty-seven controlled fits cross soft KL, signed L1 and direction+magnitude with four response weights and three common seeds. Static-KL and response-error selectors save separate checkpoints from each identical training trajectory. All fits use 120 epochs and 6,360 optimizer steps. No test metric selects a checkpoint or weight.
- Four MNL regularization candidates are evaluated under the same two choice-validation criteria. Departure selection is kept separate. Utility differences and finite input perturbations are reported without inventing value-of-time coefficients.
- The full combination audit traces endpoints, pairs, interactions, training units and checkpoint initialization. The existing fare×congestion combination is held out; its constituent factors are not.
- Equal-size disjoint Teacher repeats use 1-vs-1 for K=3 and 2-vs-2 for K=5. Shared endpoints and allocations are preserved across methods. These correlated allocations quantify target sensitivity, not a precise noise ceiling.
- Six additional fits remove all positive context-delay endpoints and whole delay mechanism quadruplets from train and validation, refit feature statistics on the retained training data, and compare soft KL with signed L1. Both methods use 120 epochs and 5,280 updates within this experiment. Its historical test results remain retrospective.
- Existing execution ledgers are reaggregated using paired people and assignment seeds. Four physical-supply arms use 1,000 fixed Helsinki people, S9 and MNL with independent neural timing, and five paired assignment seeds. Equivalent historical runs are reused only after checking people, probabilities, departure shifts and planner masks. New supply removes alternating trips within each line and ordered stop pattern. Both the MATSim schedule and planner index change.

The synthetic acquisition requested `deepseek-v4.1-flash` through OpenCode Go, and valid responses returned that model name. It is a new model/provider reference and cannot be labeled DeepSeek V4 Pro or direct official DeepSeek service. Both the 432-response pilot and the 1,080-response formal sample passed arithmetic recomputation and provenance audit. There were 1,618 attempts and 1,512 valid replies; 1,525 replies reported usage, including 13 invalid replies, while usage on 93 attempts remains unknown. No credentials are stored here, and the scripts did not change account balance-fallback settings.

The formal primary signed-minus-soft-KL response-error difference is **-0.003199**, with paired-persona 95% interval **[-0.004863, -0.001622]**. Static-selector means are 0.112887 for soft KL, 0.109688 for signed L1, 0.108824 for direction+magnitude and 0.105160 for MNL. The latter comparisons are descriptive. Removing the delay family reverses the signed-vs-soft ordering on delayed states: differences are +0.005320 for the 8-minute delay and +0.010336 for the 45-minute delay. The full tables retain every intervention and both selectors.

Some valid replies reported zero reasoning tokens: 3 pilot replies and 104 formal replies. The request parameters and returned model name did not change, and no immutable backend fingerprint was supplied. A post hoc metadata diagnostic preserves all observations and reports complete-persona strata. The new results concern the realized service mix; metadata differences do not establish a backend change or a causal reasoning-mode effect.

Reported usage totals **6,561,811 tokens**: 2,206,391 input and 4,355,420 output. The request-time rate/cache estimate is **USD 3.4952** of Go quota; the all-peak/no-cache recomputation is USD 5.8884. These cover known usage only and are neither invoices nor verified balance debits. The monthly subscription is separate. See the [Chinese author report](../evidence/experiments_20260925/EXPERIMENT_REPORT_CN.md) and packaged per-attempt records.

The 432-response pilot is separate from the prospective formal sample. The prespecified persona-level signed-minus-soft-KL contrast has pilot SD 0.006306; applying the locked halfwidth-0.01 rule and minimum sample size gives 30 new formal personas and 1,080 valid responses. The pilot is excluded from formal inference. [Analysis and transport-recovery rules](SYNTHETIC_ANALYSIS_PROTOCOL.md) retain every intervention level and failed attempt. The single fixed numeric trip limits this synthetic test; it does not establish geographic or human generalization.

## Reproduction

Use the retained research data/runtime identified in the original bundle manifest. Set `AIT_RAW_ROOT` to that research workbench directory. It must contain the frozen raw Teacher repeats, historical model artifacts, Helsinki transit supply, Java/MATSim setup and earlier execution ledgers. The new scripts do not modify these inputs.

```powershell
$env:AIT_RAW_ROOT = 'C:\path\to\research-workbench'
python scripts/revision_20260925/offline_audits.py
python scripts/revision_20260925/controlled.py resource
python scripts/revision_20260925/controlled.py train
python scripts/revision_20260925/controlled.py evaluate
python scripts/revision_20260925/mnl_selection.py
python scripts/revision_20260925/offline_diagnostics.py
python scripts/revision_20260925/delay_family.py
python scripts/revision_20260925/physical_supply.py run
python scripts/revision_20260925/audit_baseline_los.py
python scripts/revision_20260925/physical_analysis.py
python scripts/revision_20260925/final_delivery_audit.py
python scripts/revision_20260925/report_results.py
python scripts/revision_20260925/package_results.py
```

Restore the frozen `protocol.json` and original matched-response bundle before fitting. Completed controlled runs are skipped; incomplete directories are retained and require explicit recovery. The physical-supply runner refuses to overwrite unfinished MATSim outputs. Raw events remain local because of their size; their hashes and person ledgers are retained for auditing.

Synthetic calls are a separate explicit operation. `synthetic_teacher.py` reads its credential only from standard input, stops on client/access/quota errors, reserves a conservative token-cost envelope, and records failed attempts and returned model IDs. Its recovery amendment only retries missing valid slots under bounded attempt and known-usage limits. A partial cohort cannot be analyzed as complete. Unreported transport usage remains unknown; the account subscription payment, quota-equivalent estimates and actual invoices are distinct.

After restoring the experiment archive at the repository root, no API key is required to recompute the synthetic results:

```powershell
python scripts/revision_20260925/synthetic_support.py
python scripts/revision_20260925/synthetic_analysis.py pilot
python cvpr_workspace/analysis/statistics/synthetic_audit.py pilot
python scripts/revision_20260925/synthetic_analysis.py formal
python cvpr_workspace/analysis/statistics/synthetic_audit.py formal
python cvpr_workspace/analysis/statistics/service_metadata_diagnostic.py
```

The MNL inference path uses the repository implementation, verified against all 437 historical endpoints to absolute tolerance `1e-12`. Fresh historical fitting, raw-repeat reconstruction and MATSim still need the external materials listed above. Offline recomputation of the new synthetic comparisons uses the frozen API replies and checkpoints included in the archive; it does not prove that a future backend would return the same responses.

Go model availability, endpoint and rates were checked against [OpenCode Go documentation](https://opencode.ai/docs/go/) on 25 September 2026. Prices and promotional quotas can change; recheck them before resuming later.
