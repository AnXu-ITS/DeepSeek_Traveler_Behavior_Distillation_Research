"""Singapore real-supply pipeline (Phase A).

Modules here build the MATSim supply side from real data:

- ``projection``      WGS84 -> UTM 48N (pure Python).
- ``osm_network``     OSM XML extract -> MATSim network.xml (+ stats/QA).
- ``gtfs_prep``       Singapore GTFS -> region-filtered stops/trips.
- ``build_transit``   stop snapping + route routing -> transitSchedule.xml +
                      transitVehicles.xml (+ artificial access links).
"""
