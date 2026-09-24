# Model card and usage guide

## Intended use

The Students are compact surrogates for scenario-based travel behavior research. They map structured traveler/trip/context/supply attributes to probabilities over car, PT, bicycle and walking, plus a departure adjustment in minutes. They can generate behavioral demand for subsequent routing and MATSim simulation.

Use them to study the behavior of the retained surrogate under explicit inputs. The present evidence does not establish calibrated citywide mode shares, causal policy impacts, or individual-level travel forecasts. Select and validate a model against the intervention and outcome relevant to your application: low Teacher error does not guarantee human-response agreement.

## Which model should I load?

| Need | Artifact | Notes |
|---|---|---|
| Frozen supply-adapted model used in the paper | `releases/s9_supply_aware_v2/checkpoint/model.pt` | SA-Student is the manuscript name; S9 is the archival identifier |
| Controlled joint objective | `outputs/matched_response_v1/train/<objective>_seed<seed>/best.pt` | Choice-selected fit; objectives: `soft_kl`, `ce_kl`, `signed_l1`, `direction_magnitude` |
| MNL-S choice coefficients | `outputs/reviewer_closure_20260920/baseline/model.npz` | Same-feature soft-target linear choice model |
| Modular timing comparison | `outputs/revision_20260921/baselines/` | Separate departure-MAE selection; see run metadata |
| Earlier generic initialization | `releases/s7_w3_generic_core_v1/` | Historical predecessor |

S8 is deprecated. A filename containing `s8` in an architecture or adapter class may still implement the compatible 12-feature schema used by S9; the checkpoint path, schema assertions and hash determine the artifact actually loaded.

## Minimal inference

Run from the repository root:

```bash
python -m pip install -e .
python -m pip install -r requirements-research.txt
python examples/predict_released_student.py
```

The example loads one synthetic held-out state from the prepared benchmark, uses the production `StudentAdapter`, validates probability normalization and timing bounds, and prints the checkpoint SHA-256. This is an inference check, not a human-validity test. Python packaging requires Python ≥3.11; the historical research used a later Python/PyTorch environment recorded in run metadata. Dependency minimums are not a fully pinned historical environment.

For programmatic use:

```python
from pathlib import Path
from reference_pipeline.student_adapter import StudentAdapter
from traveler_distillation.schemas.state import UniversalTravelerState

adapter = StudentAdapter(Path("releases/s9_supply_aware_v2/checkpoint/model.pt"), device="cpu")
state = UniversalTravelerState.model_validate(your_state_dictionary)
prediction = adapter.predict([state])[0]
```

`your_state_dictionary` must follow the retained schema. Consult the [synthetic example](../examples/predict_released_student.py), [field dictionary](../evidence/paper_20260924/methods/input_fields.csv) and [feature implementation](../src/traveler_distillation/student/features.py). Supply-aware fields are implemented in [accessibility_features.py](../src/traveler_distillation/accessibility/accessibility_features.py).

Output fields are `mode`, `mode_probabilities` and `departure_time_shift_min`. The adapter reports probabilities for available alternatives and selects the largest as `mode`; a simulation may instead sample or apply a route-feasibility mask. Probabilities are not realized trip shares. Keep the checkpoint's own normalizer and vocabularies.

## Constructing a simulation

The reusable pipeline reads population CSV and a YAML configuration:

```bash
python run_pipeline.py --help
python run_pipeline.py --config configs/reference_example.yaml
```

Inspect the configuration before running: referenced network, timetable, vehicles, prepared routing inputs and output locations must exist. Running MATSim additionally requires the compatible Java/MATSim installation. Historical experiments used Java 25 and MATSim 2026.0, with one iteration in the reported execution comparison. Several historical experiment scripts retain author-machine setup assumptions and are provided as research implementations, not a turnkey fresh-city installer.

[Execution workflow](EXECUTION_WORKFLOW.md) explains assignment, adjusted departure times, route failure and event denominators. Use the paper-specific Helsinki driver only after reconstructing its fixed population and supply prerequisites.

## Applying the model to a new city

Validate the meaning, units and supported categories of every field; distinguish unavailable alternatives from failed route search. Inspect PT time components, coverage, transfer-search limits and active-mode speeds. Evaluate both mode-probability levels and paired responses for the intended intervention, then follow them through assignment, routing and simulated events.

A new city's numerical supply inputs do not automatically require retraining, but zero-shot execution alone does not establish behavioral validity. Adaptation should be justified by measured input/response discrepancies and evaluated on held-out observations. Neither human questionnaires nor evaluation tasks should be silently reused for training.

## Known limitations

- Teacher outputs are elicited judgments, and historical service drift is not fully identifiable.
- The held-out controlled benchmark has only six synthetic personas; its bootstrap intervals are conditional on this benchmark.
- Survey samples are nonrepresentative and require missing-input assumptions; car wording differs from the private-car modeling contract.
- The PT search is bounded to direct/one-transfer connections. Failure is an algorithm-specific feasibility result.
- The current MATSim experiment fixes behavioral predictions; it does not estimate behavioral adaptation or equilibrium under a physical disruption.
- Poorer-accessibility, fare and delay responses exhibit substantive model-specific mismatches. See [results](RESULTS.md) before model selection.
