# Training and model selection

## Supervision and split construction

DeepSeek supplies numerical mode-probability judgments and departure adjustments. These are elicited judgments, not human frequencies or token probabilities. Available repeated vectors are averaged within a state. The authors confirm that all Teacher elicitation used DeepSeek V4 Pro supplied directly by the official DeepSeek API (`https://api.deepseek.com`, request model `deepseek-v4-pro`). Later response fields agree with that name. Immutable backend identities and complete failure/retry records are not established for every earlier supervision source. The updated official API documentation confirms continued Pro service after 14 September 2026; the old news-page routing plan is not used to relabel these experiments as Flash. Re-evaluation should use retained targets rather than assume a new API response reproduces the historical service.

| Source / partition | Coverage |
|---|---|
| Controlled training | 1,839 states; 28 personas; 1,681 pairs, exposing 3,362 endpoints per epoch |
| Controlled validation | 365 states; six personas |
| Controlled test | 437 states; six personas; 398 response pairs; 94 interaction contrasts |
| Test source composition | 226 single-context, 51 accessibility, 96 joint-context and 64 mechanism states |
| Accessibility adaptation | 336 states: 234/51/51 train/validation/test; OD pools 79/14/20 |
| Accessibility Teacher repetition | 89 states × 3 + 247 states × 5 = 1,502 valid queries |

The [prepared bundle](../outputs/matched_response_v1/bundle) contains the exact synthetic endpoints, pairs, interactions, extractor and content hashes. This is the preferred starting point for controlled-model reproduction.

## Joint Student architecture

Each current joint neural Student has **24,562 parameters**. Nine categorical fields have eight-dimensional embeddings. A global encoder combines them with 17 standardized numerical fields in two layers of width 64. Alternative attributes and a mode embedding enter a two-layer encoder of width 32; the shared utility scorer has 64 hidden units. Hidden activations are ReLU; the global encoder and scorer use dropout 0.1. Masked softmax normalizes choice probabilities over available alternatives. Departure output is `60 * tanh(z)` minutes, bounded to ±60.

Normalization and categorical vocabularies must travel with the checkpoint. Replacing them with values fitted on a new city changes the model.

## Controlled objectives

| Objective | Endpoint supervision | Additional supervision |
|---|---|---|
| Soft KL | KL(Teacher || Student) + departure Huber loss | None |
| CE+KL | Soft KL plus unit-weight cross-entropy for the Teacher's leading mode | None |
| Signed response | Soft KL | Absolute error in signed probability changes |
| Direction+magnitude | Soft KL | Unit-weight sign penalty and absolute magnitude error |

The response losses use the union of alternatives available in either endpoint. The direction denominator excludes Teacher changes with magnitude ≤1e-6; the magnitude denominator does not. The departure Huber transition is one minute. Paired penalties do not guarantee sign agreement; the supplement derives cancellation regions and the changed idealized target under CE+KL.

**Matched settings:** seeds 42, 2026 and 7; random initialization shared within seed; Adam; learning rate 0.000125; weight decay 0.0001; batches of 32 pairs; 120 epochs; 6,360 updates. The probability-response experiment selects the earliest epoch attaining the lowest validation macro-source KL. It does not select on the test set or human responses.

## MNL-S and modular timing

MNL-S has four linear utilities over 111 explanatory variables, giving 444 coefficients. L-BFGS-B minimizes soft-target cross-entropy with L2 regularization; 0.0001 is selected from {0.0001, 0.001, 0.01, 0.1} by validation macro-source KL, with at most 2,000 iterations.

Timing is a separate experiment selected by validation departure MAE. Ridge uses penalty 0.001; the bounded-linear predictor has 111 parameters; the independent neural timing predictor has 18,289. MNL-S plus neural timing therefore has 18,733 parameters. Joint objectives are independently retrained under the same timing-selection criterion. Do not combine their selected epochs with those from the probability-response experiment.

## SA-Student adaptation

SA-Student is the frozen release `s9_supply_aware_v2`, initialized from S7-W3. Six added PT-supply input columns begin at zero; existing weights and normalization are preserved, and new-field normalization is fitted on accessibility training states. The source mixture is single-context : joint-context : mechanism : accessibility = 2:1:1:1, with additional persona and accessibility-curve updates.

Adaptation uses seed 42, Adam with the same learning rate/weight decay, batches of 32 states or pairs and eight mechanism quadruplets, at most 120 epochs, and patience 15. Epoch **17** is selected by accessibility-validation KL subject to retention constraints: at most one percentage point loss in single-context accuracy and at most 10% increases in single-context KL and KL on previously seen joint contexts. Its response mask uses the perturbed state, unlike the union mask in the controlled comparison.

[Descriptive adaptation specification](../evidence/paper_20260924/methods/s9_adaptation_specification.json) · [Accessibility prompt](../evidence/paper_20260924/methods/accessibility_teacher_prompt.txt). The specification JSON describes the recipe; it is not an executable training configuration.

S8 is deprecated because walking/cycling speed inputs were incorrect. Use S9 for the frozen supply-adapted model. S7 and S8 remain identifiable historical artifacts.

## Reproduction commands

From the repository root, with research dependencies installed:

```bash
python scripts/matched_response.py --help
python scripts/matched_response.py smoke --bundle outputs/matched_response_v1/bundle --output outputs/local_smoke
python scripts/matched_response.py train --bundle outputs/matched_response_v1/bundle --output outputs/local_training
python scripts/matched_response.py evaluate-suite --bundle outputs/matched_response_v1/bundle --suite outputs/local_training --split test --output outputs/local_test
```

Output folders must be new. Smoke runs use reduced training/validation data and cannot evaluate the real test set. Full training is a new experiment; a matching protocol does not imply bitwise-identical weights across different numerical environments. The archived run files record versions and selected-checkpoint hashes. The included `code_snapshot.zip` files preserve the controlled training source. [Reproduction scope](REPRODUCIBILITY.md).
