# Singapore Phase A execution checks

Phase A checks preparation of explicit road/transit supply, schema-compatible Student demand and MATSim execution on a small population. The checks concern file/network validity, route construction and simulated mode events. They are prerequisites for later scale/scenario experiments, not an empirical calibration of real traveler behavior.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/singapore_phase_a/PHASE_A_GATE.md).

[Current research](<../../docs/RESEARCH_DESIGN.md>) · [Training](<../../docs/TRAINING.md>) · [Results](<../../docs/RESULTS.md>) · [Data and access](<../../docs/DATA_SOURCES.md>) · [Model use](<../../docs/MODEL_USE.md>)

## Singapore execution-readiness checks

|check item|result|
|---|---|
| 100 items S7-W3 agents baseline run exit code | **0** |
| car / pt / walk / bike four modes executed legs | **all observed**(departure events grouped by legMode count) |
| scheduled PT actual execution | **5,580 items transit all service trips depart**(TransitDriverStarts=5,580) |
| PT passenger boarding and alighting | **8 boarding / 8 alighting**(PersonEnters/LeavesPtVehicle paired) |
| stuckAndAbort | **0** |
|journey completeness| departure 5,796 = arrival 5,796(no failed trips) |

## Road-network and transit-supply artifact inventory

|artifacts|population size|
|---|---|
| `osm/tampines_pasir_ris.osm`(Overpass, 2026-08-25) | 102,784 nodes / 30,140 ways |
| `osm/network.xml`(UTM 48N) | 100,867 nodes / 191,617 links / 3,520 km; car largest connected component 95% |
| `transit/transitSchedule.xml` | 851 stops / 170 transit lines / **5,580 service trips**(including 256 items MRT service trips) |
| `transit/transitVehicles.xml` | 5,580 vehicles(busType/railType) |
| `transit/network_with_transit.xml` | +851 stop-access links(ai_in/ai_out)+28,681 transit dedicated reverse links(busr_*)+25,441 pedestrians/bicycle reverse links(pdr_*) |
