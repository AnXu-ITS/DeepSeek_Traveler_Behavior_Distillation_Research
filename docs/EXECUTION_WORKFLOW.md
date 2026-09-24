# From traveler inputs to MATSim events

```text
Traveler attributes + scenario conditions
                  ↓
       Behavioral state encoding
                  ↓
               Student
                  ↓
Mode probabilities + departure adjustment
                  ↓
       Demand-plan construction
                  ↓
Mode assignment + route-feasibility check
                  ↓
          Multimodal routing
                  ↓
       MATSim population/plans
                  ↓
          MATSim execution
                  ↓
       Realized trips and events
```

## What each stage does

| Stage | Input → output | Implementation / interpretation |
|---|---|---|
| Behavioral state encoding | Attributes, trip, context and supply → checkpoint features | The fitted extractor preserves feature order, vocabulary, normalization and availability |
| Student | Encoded state → four-mode probabilities and departure adjustment | Local inference; no Teacher call at deployment |
| Demand-plan construction | Probabilities and timing → candidate demand | Retains the initial traveler population and its OD assignments |
| Mode assignment and feasibility | Candidate demand → assigned mode | Deterministic, stochastic or feasibility-constrained stochastic assignment |
| Multimodal routing | Assigned mode and adjusted time → paths and itineraries | Direct/one-transfer PT search; explicit walking fallback where specified |
| MATSim population/plans | Routed demand → executable person plans | Activities, departures, mode legs and routes |
| MATSim execution | Plans + network + vehicles + schedule → simulated events | Current paper: one iteration, fixed behavioral predictions |
| Realized trips and events | Events → person-level and event-level summaries | PT boarding, completed journeys, travel time and stuck events have different meanings |

The reusable entry point is [run_pipeline.py](../run_pipeline.py). The current Helsinki experiment is implemented in [helsinki_execution.py](../scripts/revision_20260921/helsinki_execution.py), with event analysis in [execution_analysis.py](../scripts/revision_20260921/execution_analysis.py). The general pipeline and the paper-specific experiment have different configuration requirements; they are not interchangeable commands.

## Timing, availability and feasibility

Supply attributes are computed at the requested departure time. The fixed Student predicts an adjustment, which is applied before feasibility checking and final routing. No second behavioral prediction is made. Return trips are routed at their scheduled return time.

Behavioral availability defines which alternatives the model may choose. Route feasibility additionally requires that the search return a connection in that mode at the adjusted time. The PT search returns direct or one-transfer paths; failure does not prove that no connection exists anywhere in the physical network.

Deterministic assignment uses the largest predicted probability. Ordinary stochastic assignment samples from the distribution. Feasibility-constrained assignment masks alternatives lacking a returned route, renormalizes the remaining probability once, and samples. Zero retained probability mass stops the paper experiment; it is not silently replaced by an invented choice. No such case occurred in the 140 completed runs.

Without feasibility-constrained assignment, failed car, bicycle or PT routing falls directly back to walking. It does not choose the next most probable mode. The general adapter can emit an origin-link-only walk after walking-search failure; no such route was found in the inspected 140,000 run-person records. Deployment on other networks must therefore validate connectivity explicitly.

## Helsinki experiment

The design uses 1,000 fixed travelers, OSM roads and HSL timetables. It compares baseline against a **15-minute perceived PT delay**, with the physical timetable unchanged. Ten fitted model instances are evaluated: one SA-Student and three each of soft KL, direction+magnitude, and MNL-S plus neural timing. Each has two deterministic runs and twelve sampled runs (two conditions × two sampling rules × three assignment seeds), for **140 runs / 70 pairs**. Flow and storage capacity factors are 0.3.

Every PT probability or share uses the initial 1,000 travelers as denominator. Simulated PT use means boarding before reaching the outbound destination, not successful journey completion. Total boarding events can include transfers and return trips and must not be reported as unique PT users.

The response gap is simulated minus predicted change. Assignment seeds describe sampling variability; training seeds describe fitted-model variability. The design measures response preservation with fixed behavior, not adaptation to a real service disruption or a day-to-day equilibrium.

[Results](RESULTS.md#simulation-execution) · [Model use](MODEL_USE.md) · [Back to project](../README.md)
