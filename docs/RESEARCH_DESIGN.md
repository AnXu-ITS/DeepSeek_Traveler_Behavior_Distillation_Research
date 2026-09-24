# Research direction and study design

The central question is whether behavioral compression preserves a response to changed conditions, rather than only a prediction at one state. For the same traveler and trip, the study compares the difference between a baseline and an intervention:

$$\Delta p_j = p_j(x')-p_j(x),\quad j\in\{T,S\}.$$

The four choice coordinates are car, PT, bicycle and walking. Departure adjustment is a separate continuous outcome. These finite scenario contrasts are not estimates of population elasticities or causal effects by themselves.

![Research concept](assets/figure1.png)

## Three linked comparisons

| Question | Experimental unit and coverage | Comparison | Main outcome |
|---|---|---|---|
| Can a Student preserve Teacher responses? | 437 test states, 398 paired changes and 94 interactions, from six held-out synthetic personas | Four joint neural objectives; supply-matched MNL-S | Probability-response error, macro-source KL, interaction error |
| Do responses agree with people? | 332 Singapore residents and 321 mapped Shanghai respondents, each with ten tasks | Fixed Students against stated choices; contemporary Teacher on 24 people per city | Response bias, accuracy, PT share bias, Brier score and clipped NLL |
| What survives execution? | One fixed 1,000-person Helsinki population; 140 simulations | Ten fitted models; three assignment rules; baseline and perceived delay | Predicted-to-simulated PT response gap |

The study also compares modular departure predictors, numerical input assumptions, synthetic OD assignments, active-mode speeds and scenario-construction time. Each addresses a different source of discrepancy; none substitutes for the other evidence levels.

## Model families

**Joint neural Students** share feature encoders and have separate mode-choice and departure heads. Soft KL, CE+KL, signed response and direction+magnitude objectives use matched inputs, initialization, pair order, availability and update budgets within each seed.

**MNL-S** fits four linear utilities to the same soft endpoint targets. It receives the same information and exposure weights, but its regularization and optimization are appropriate to the linear specification. Ridge, bounded-linear and independent-neural timing are evaluated separately.

**SA-Student (S9)** adapts an earlier generic Student to additional PT supply attributes. Its inherited training, auxiliary losses and treatment of availability differ from the controlled models. It is an additional deployed model, not an architecture-only or loss-only ablation.

## State representation and data separation

A state combines nine categorical traveler/trip/context fields, 17 global numerical fields and 12 numerical attributes per alternative. Mode identity and availability are represented separately. The encoded information includes travel time and cost, weather, household and mobility characteristics, and PT access, waiting, in-vehicle, egress and transfer conditions. See the [field dictionary](../evidence/paper_20260924/methods/input_fields.csv).

The controlled train/validation/test split has 1,839/365/437 states and 28/6/6 disjoint synthetic personas. Within the accessibility source, OD pools are also separated. Human questionnaires are never used for fitting, checkpoint selection or tuning to respondent answers. Contemporary Teacher evaluation is answer-blind and uses the same encoded state and availability as each Student.

## Interpretation

Endpoint fidelity can bound response error, but finite capacity and optimization make the allocation of fitting effort matter. Paired supervision reweights information already present in endpoint targets; it adds no new behavioral observations. Better Teacher fidelity does not guarantee better human agreement. Route feasibility, simulated PT boarding and journey completion are separate outcomes.

The current experiment holds behavioral predictions fixed during MATSim execution. Earlier prototype feedback-loop reports describe a different development stage and do not establish a closed-loop equilibrium result for this paper.

[Training](TRAINING.md) · [Results](RESULTS.md) · [Back to project](../README.md)
