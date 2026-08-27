# S9 Transit Accessibility Feature Definitions (frozen)

Identical field set and routing rules to S8 (`releases/s8_supply_aware_v1/accessibility/feature_definitions.md`).

## S9 corrections (the S8 data error)

| Quantity | S8 (wrong) | S9 (fixed) |
|---|---|---|
| walk travel time | length / link freespeed (effective ~29 km/h) | length / 1.34 m/s (~4.8 km/h) |
| bike travel time | length / link freespeed (effective ~29 km/h) | length / 4.17 m/s (15 km/h) |
| car travel time | length / freespeed (unchanged) | length / freespeed |
| pt access/egress walk | length / freespeed (too fast) | length / 1.34 m/s |
| MATSim walk/bike legs | network modes at link freespeed, consuming road capacity | teleported, walk 1.39 m/s / bike 3.9 m/s |

Measured after the fix (114-OD pool): walk median 3.28 km/h, bike 10.14 km/h, car 31.09 km/h.

## Routing rules — unchanged from S8

- access radius 700 m; egress 700 m (extended 1.5 km); max 1 transfer;
- boarding buffer max(300 s, access+60 s); connection window 180-2700 s;
- departure window 1 h; coverage ratio = D_vehicle/D_OD clip [0,1];
- infeasible sentinel travel time 120 min; FVR is learned behavior.
