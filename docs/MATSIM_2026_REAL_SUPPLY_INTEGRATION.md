# MATSim 2026 supply integration notes

These implementation notes cover the transition from synthetic/prototype demand to explicit networks, transit schedules, vehicles and routed person plans. Network route serialization, PT access/transfer/egress legs and event definitions must match MATSim conventions. The historical integration gates establish that particular scenarios execute; they do not establish calibrated real-world congestion or general route-search completeness. The current paper tracks predicted probabilities, assigned modes, routed modes, PT boarding and journey completion separately.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/MATSIM_2026_REAL_SUPPLY_INTEGRATION.md).

[Current research](<RESEARCH_DESIGN.md>) · [Training](<TRAINING.md>) · [Results](<RESULTS.md>) · [Data and access](<DATA_SOURCES.md>) · [Model use](<MODEL_USE.md>)

## Transit-vehicle XML example

```xml
<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.matsim.org/files/dtd http://www.matsim.org/files/dtd/vehicleDefinitions_v1.0.xsd">
  <vehicleType id="busType">
    <capacity><seats persons="40"/><standingRoom persons="20"/></capacity>
    <length meter="12.0"/>
    <accessTime secondsPerPerson="1.0"/>
    <doorOperation mode="serial"/>
    <passengerCarEquivalents pce="2.0"/>
  </vehicleType>
  <vehicle id="veh_xxx" type="busType"/>
</vehicleDefinitions>
```
