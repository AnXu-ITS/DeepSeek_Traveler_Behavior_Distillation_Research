# Proposed response-fidelity revision experiments

**Historical planning snapshot.** The author subsequently authorized experiments, changed the new Teacher request to Go DeepSeek V4.1 Flash, deferred new-human validation, and required stopping before manuscript writing. See the [current experiment record](EXPERIMENTS_20260925.md). The original proposal and its Pro pricing assumptions below are preserved for provenance.

**Status when this plan was written: proposed, not executed or approved.** The author requested a plan and API budget before any new experiment. The present revision only edits the manuscript and repository and audits existing metadata/usage. Existing numerical findings, raw questionnaires and answers remain unchanged. The base experimental commit is `4e67286ce72680a9e3a916a3fb7e97b1548b53f7`.

## Decisions supported by the current evidence

The paper already supports a bounded diagnostic contribution: static imitation, finite responses, human agreement and execution consistency are distinct. It does not yet demonstrate an independently validated human-alignment repair or general response-learning benefits across unseen families. Broader claims need corresponding tests; adding cities or repeating every historical run is not the default response.

The existing `fare_x_congestion` combination is already excluded from train/validation in `configs/joint_sampling.yaml`. The prepared manifest excludes 112 train and 24 validation candidates, and test endpoints include 24 `joint_unseen` states. Thus compositional evidence is not absent. Existing aggregate results and six-persona coverage do not replace a full leakage audit or a separately reported, general intervention-disjoint test.

## Stage A: recover value from existing targets first — zero API calls

1. Audit the existing compositional split across endpoints, pairs, interactions, auxiliary targets and checkpoint lineage. Report the existing held-out-combination results separately, including failures; do not call the whole test set intervention-disjoint.
2. Compare equal-size disjoint subsets of repeated Teacher elicitations on the same response pairs. For three repeats use 1-vs-1 allocations; for five repeats use 2-vs-2 with a rotated omitted repeat. Reuse identical partitions across Students. These are correlated sensitivity allocations, not independent trials or a precise noise ceiling.
3. Cross probability objective with selection criterion: soft KL, signed response and direction+magnitude, each selected by static KL and response error. Include MNL-S under matched endpoint/response validation rules. Use the same three seeds as paired optimization replicates; increasing seeds does not replace independent personas. Do not substitute departure-MAE selection for response selection. Historical per-epoch checkpoints may be incomplete: when necessary, retrain from frozen targets into new directories rather than relabel an existing fit.
4. For the two response objectives, use predeclared response weights 0.25, 0.5, 1 and 2, shared update budgets and selection budgets; weight zero is the shared soft-KL control. Record loss scales and gradient contributions, and inspect identified MNL utility contrasts and finite perturbations. This grid is proposed, not claimed to have run. Do not compute value of time without identified, compatible time/cost coefficients.
5. Re-aggregate existing stagewise execution ledgers with common paired seeds to distinguish probability correction, assignment, route fallback and boarding. Existing summaries alone cannot create stage-specific confidence intervals when the needed paired observations are absent.

These steps may involve new local fitting or analysis, so they remain pending even though they require **0 new API requests**. Save all runs, including null outcomes, under a new protocol ID. A small single-fit resource check should precede a full local training matrix after approval. Existing 12-fit times are useful context, not a performance promise for the new protocol.

## Stage B: unseen intensity and intervention tests

Use official DeepSeek V4 Pro only, as requested. Proposed pilot: 12 new synthetic personas × 12 distinct states × 3 repeats = **432 valid responses**. The state manifest should include one shared baseline; paired delay/fare/access levels; joint changes linking known single-factor endpoints; and two additional prespecified states. Deduplicate shared endpoints before querying. Specify exact intensity intervals and physical feasibility before execution. The pilot estimates persona-level paired-difference variability; it is not the confirmatory test.

Freeze separate persona, intensity and family partitions. For family holdout, remove every target, pair, auxiliary label and adaptation endpoint from the held-out family. Keep input semantics and preprocessing fixed. Use existing targets for fitting when they support the partition, and keep pilot personas out of the final test. If target coverage cannot support a leak-free partition, report that limitation before expanding acquisition.

Choose the formal number of personas from a meaningful precision target and pilot variance, using `n ≈ (1.96 s_d / h)^2` only as a planning approximation. For budgeting, **60 entirely new personas × 12 states × 3 repeats = 2,160 valid responses** is an example, not a power justification. Pilot and formal calls are additive here. Final inference uses persona-level paired comparisons, with seed variation separately reported, and interpolation is distinguished from extrapolation.

This is within-service generalization. A second independent Teacher would test a different claim and conflicts with an all-V4-Pro-only scope unless explicitly authorized; it is optional and excluded from this budget.

## Stage C: independently evaluate a human-response repair

This is the strongest route from diagnosis to a useful intervention. Use new consenting respondents and numerically aligned tasks, with explicit currency, alternative meanings and availability. Freeze baseline/perturbation counterbalancing and task-order randomization. Do not alter the old questionnaires or answers.

Compare original Teacher-only MNL-S, a constrained utility/calibration repair, and a conventional choice baseline using the same human-label budget. Separate respondent-level calibration/validation/test groups and label every human-trained model. A budget illustration is 120 new people with a 60/30/30 split and ten tasks each; 30 final test respondents is **not** automatically sufficient. Determine final sample size from the prespecified response and precision requirements before collection. Existing survey findings already informed this repair hypothesis, so reuse of those full samples cannot be called a new independent validation.

Querying all 120 × 10 tasks with three V4 Pro repeats would require **3,600 valid responses**. Querying only a fixed 30-person final subset would use 900, but would not provide Teacher comparisons for all calibration/validation participants. The 3,600-call scenario below is the transparent full-sample budget. Survey recruitment/incentive costs are separate and cannot be estimated from API logs. Existing consent wording does not establish approval for a future study; settle the applicable institutional process before recruiting.

Predefine one primary accessibility-response error; report baseline calibration, Teacher fidelity and the effects on other contrasts. A failed repair is still reportable. Retrofitted nested analysis of old data can be an exploratory sensitivity check, with 0 API calls when frozen predictions suffice, but does not erase prior full-sample hypothesis exposure.

## Stage D: physical supply experiment — zero API calls with fixed Students

Needed if the paper is to claim response under real supply changes; the current perceived-delay execution audit can stand on its narrower scope. Start with one reproducible headway disruption in Helsinki. Four arms: baseline input/baseline supply; perceived-delay-only input/baseline supply; frozen baseline behavior/disrupted supply; and behavior recomputed from disrupted level of service/disrupted supply. Do not add the same delay twice after rebuilding level of service.

Proposed minimum: SA-Student and MNL-S with neural timing, one declared assignment rule, five paired assignment seeds, four arms = **40 runs**, using the same 1,000-person population and paired random numbers. This is a fixed-checkpoint execution test; it does not estimate training uncertainty. Use a different, fully balanced matrix if the claim includes training or assignment-rule effects. Report raw/adjusted probabilities, assignment, routed mode, boarding, completion and journey time. No new Teacher calls are needed. Archive modified supply hashes and checks that the intended service actually changed. Claim feedback/equilibrium only if a separate feedback design is run.

## API planning scenarios

The final retained 1,440-response campaign averaged 1,614 input and 2,260 output tokens, including reasoning, with about 61% cached input. These averages are used only as planning assumptions for similar prompts. The table includes **10% additional usage-bearing calls** and the observed token/cache mix. Prices are the official V4 Pro rate cards checked on 25 September 2026. This is not a confidence interval or a guarantee.

| Proposed acquisition | Valid responses | Expected tokens including allowance | USD off-peak–peak | CNY off-peak–peak | Peak no-cache planning envelope, USD |
|---|---:|---:|---:|---:|---:|
| Synthetic pilot | 432 | 1.84 M | 2.34–4.67 | 15.93–31.86 | 32.14 |
| Formal synthetic example, separate from pilot | 2,160 | 9.21 M | 11.68–23.36 | 79.64–159.28 | 160.43 |
| New human-task example, all 120 people | 3,600 | 15.34 M | 19.47–38.94 | 132.74–265.47 | 267.38 |
| **All three illustrative acquisitions** | **6,192** | **26.39 M** | **33.49–66.97** | **228.31–456.61** | **459.95** |

The planning envelope assumes at most 2,000 uncached input tokens and 16,384 output tokens per allowed attempt, with 476/2,376/3,960 attempts respectively. It is conditional, not an enforced spending cap. A 32,768-token retry would exceed this assumption and must be separately budgeted. Before any approved launch, a runner must count input budgets, enforce attempt and monetary limits, stop on exhausted balance or persistent rate errors, preserve usage for invalid answers, and not keep retrying HTTP 402. Freeze model, prompt hash, reasoning settings, UTC times, response model/fingerprint and deduplication keys. No credentials belong in the repository.

**Recommended next decision:** authorize Stage A first, then decide whether to authorize only the 432-response Stage B pilot. Formal sample size, human collection and physical-supply runs should be confirmed from their frozen protocols separately. No step in this plan has been launched by the editorial revision.
