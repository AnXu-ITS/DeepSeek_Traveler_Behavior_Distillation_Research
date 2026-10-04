# Data sources, provenance and access

## Evidence types

| Material | Origin | What it supports | Important boundary |
|---|---|---|---|
| Synthetic traveler states | Generated personas, trips and scenario conditions | Controlled Teacher–Student comparisons | Generated profiles are not survey respondents or a representative population |
| Teacher targets | DeepSeek numerical probability and departure judgments | Distillation and response fidelity | Targets are neither revealed choices nor token probabilities |
| Singapore supply | OSM road extract and a retained GTFS snapshot | PT accessibility and simulated network demand | Open road geometry does not establish behavioral field validity |
| Helsinki supply | OSM roads and HSL timetable data | Transfer and the 140-run execution comparison | One fixed 1,000-person synthetic population; physical timetable unchanged under perceived delay |
| Shanghai supply | OSM roads, station/line information and a synthetic metro timetable | Input and active-mode-speed sensitivity | Timetable/headways are constructed assumptions, not observed operations |
| Singapore questionnaire | Google Forms, 20 August–5 September 2026 | 332 eligible residents; 3,320 stated choices | Convenience and snowball recruitment |
| Shanghai questionnaire | Wenjuanxing, 12–18 September 2026 | 333 eligible completions; 321 mapped people; 3,208 explicit modeled choices | Twelve eligible people enter only human sensitivity; two “none suitable” answers are not four-mode labels |

## Synthetic supervision

The controlled benchmark joins four sources: single-context, joint-context, mechanism and accessibility states. [The prepared manifest](../outputs/matched_response_v1/bundle/manifest.json) gives hashes, source lineage, splits and counts. The training/validation/test groups contain 28/6/6 personas, with 1,839/365/437 states. Source files are named in [matched_response.yaml](../configs/matched_response.yaml). The prepared bundle makes the controlled experiment usable without new Teacher requests.

The accessibility source contains 336 states and 1,502 valid Teacher queries, with separate persona and OD pools. The [SA-Student release](../releases/s9_supply_aware_v2/README.md) provides feature definitions, provenance and checkpoint metadata. The 51-state accessibility test set is one part of the 437-state controlled test set; it is not interchangeable with the 226-state single-context benchmark.

## Supply provenance

**Singapore.** The study area covers Tampines and Pasir Ris. The retained raw snapshot metadata lists 603 routes, 5,376 stops and 230,915 trips before study-area preparation. It identifies a user-provided snapshot and the feed publisher string `singapore-gtfs-2025`; it is not evidence that the exact feed was downloaded from an official LTA endpoint. Its archive SHA-256 is `1fcc6f5f2766d34aab08d40adabbb094f9124feba6e39aba671e5f532d4eb3cd`. The prepared study supply and its filtering differ from these raw-feed counts. Consult the [Singapore data documentation](../data/singapore/DATA_README.md) and release provenance before reconstructing it.

The retained Singapore feed documentation further identifies **estimated bus travel times and frequency-based/synthetic MRT schedules**. Thus the study combines real road geometry with a community-constructed supply snapshot; it must not be described as an official measured timetable.

**Helsinki.** Retained preparation metadata records feed coverage from 26 August to 24 October 2026. The original OSM source PBF has SHA-256 `8c59e968929d3751add20a98ac8c36b6928d081aaca19221ddf8939accef197d`. The existing [supply metadata directory](../evidence/e5_helsinki/supply) records the historical extraction/preparation. These are snapshot facts, not claims about the current live timetable.

**Shanghai.** The Xuhui study area uses real road geometry and station/line information, with synthetic metro service. Construction assumes peak/off-peak/late-evening headways and approximate station ordering; it is a sensitivity environment. No observed respondent OD trip is inferred from it. Reassigning OD pairs and exchanging complete supply profiles test whether gains arise from respondent-specific matching or a general numerical input shift. See [Shanghai data documentation](../data/shanghai/DATA_README.md).

## Survey provenance and English instruments

[Singapore questionnaire](surveys/SINGAPORE_QUESTIONNAIRE.md) and [Shanghai questionnaire](surveys/SHANGHAI_QUESTIONNAIRE.md) reproduce the question content in English for repository readers. They are documentation translations, not revised surveys or additional data collection. The original workbooks and instruments remain unchanged in the authors' source workspace.

The car option displayed in the surveys includes driving/taxi wording. The paper represents it analytically as private car with ownership and licence requirements. It scores all explicit choices, including eight Singapore car and two Shanghai bicycle choices outside modeled availability. The translated instruments retain the displayed wording so readers can assess this mismatch.

Numerical Shanghai durations were researcher-specified encodings: the questionnaire explicitly states only a PT fare of CNY 6, a parking fee of CNY 30 and one transfer. Do not describe the additional encoded durations as respondent observations. The twelve Singapore and eight Shanghai sensitivity profiles change selected missing-input assumptions; they are not a complete uncertainty distribution.

## Access and redistribution

This repository is publicly accessible as verified on 3 October 2026. Original code/materials do not acquire a new open-source license through documentation edits. No repository-wide license grant has been added. Third-party assets retain their own terms, including OpenStreetMap attribution and ODbL obligations. The Singapore snapshot's redistribution permission and exact upstream identity should be resolved before a public raw-feed release; this update does not add that archive. Check the applicable timetable-provider and software terms before redistribution.

Included: synthetic controlled targets, selected checkpoints, experiment/statistical code, aggregate survey/Teacher/simulation results, English questionnaires, figures and manuscript snapshots. Retained separately: raw participant workbooks, participant-linked states/predictions/Teacher requests, full event archives and large external supply resources. Supporting-material requests can be made through the corresponding-author contact in the manuscript, subject to participant privacy and third-party terms.

The surveys were administered anonymously, with study information and electronic consent before participation. Participation was voluntary, and no direct personal identifiers were collected. Only de-identified attributes required for the model analyses were transmitted to the external model service. The study was conducted without a formal institutional ethics review or exemption determination. See the [author-supplied ethics and consent statement](manuscript-support/2026-10-03/DECLARATIONS.md#ethics-and-consent-statement).

[Evidence inventory](REPRODUCIBILITY.md) · [Survey results](surveys/RESULTS.md)
