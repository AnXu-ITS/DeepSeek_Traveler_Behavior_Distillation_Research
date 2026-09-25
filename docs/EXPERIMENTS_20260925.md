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

The synthetic acquisition requests `deepseek-v4.1-flash` through OpenCode Go. It is a new model/provider reference and cannot be labeled DeepSeek V4 Pro or direct official DeepSeek service. Preflight returned HTTP 400 requiring Global regions in the workspace Privacy settings. Acquisition must remain stopped until the author resolves that setting. No credentials are stored here, and no balance fallback is enabled by the scripts.

The 432-response pilot is separate from a prospective formal sample. Formal size is based on the prespecified persona-level signed-minus-soft-KL contrast and a planning halfwidth of 0.01; the allowed range is 30–120 new personas. The single fixed numeric trip limits the scope of this synthetic test. It does not establish geographic or human generalization.

## Reproduction

Use the retained research data/runtime identified in the original bundle manifest. Set `AIT_RAW_ROOT` to that research workbench directory. It must contain the frozen raw Teacher repeats, historical model artifacts, Helsinki transit supply, Java/MATSim setup and earlier execution ledgers. The new scripts do not modify these inputs.

```powershell
$env:AIT_RAW_ROOT = 'C:\path\to\research-workbench'
python scripts/revision_20260925/offline_audits.py
python scripts/revision_20260925/controlled.py resource
python scripts/revision_20260925/controlled.py train
python scripts/revision_20260925/controlled.py evaluate
python scripts/revision_20260925/mnl_selection.py
python scripts/revision_20260925/delay_family.py
python scripts/revision_20260925/physical_supply.py run
python scripts/revision_20260925/audit_baseline_los.py
python scripts/revision_20260925/physical_analysis.py
python scripts/revision_20260925/report_results.py
python scripts/revision_20260925/package_results.py
```

Restore the frozen `protocol.json` and original matched-response bundle before fitting. Completed controlled runs are skipped; incomplete directories are retained and require explicit recovery. The physical-supply runner refuses to overwrite unfinished MATSim outputs. Raw events remain local because of their size; their hashes and person ledgers are retained for auditing.

Synthetic calls are a separate explicit operation. `synthetic_teacher.py` reads its credential only from standard input, stops on client/access/quota errors, reserves a conservative token-cost envelope, and records failed attempts and returned model IDs. Do not run it before resolving the provider setting. A partial pilot cannot be analyzed as complete. The account subscription payment, quota-equivalent token estimates and actual invoices are distinct.

Go model availability, endpoint and rates were checked against [OpenCode Go documentation](https://opencode.ai/docs/go/) on 25 September 2026. Prices and promotional quotas can change; recheck them before resuming later.
