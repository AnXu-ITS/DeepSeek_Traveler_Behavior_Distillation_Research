# Singapore GTFS Snapshot

- **File**: `raw/singapore-gtfs.zip`（335,668,255 bytes）
- **SHA-256**: `1fcc6f5f2766d34aab08d40adabbb094f9124feba6e39aba671e5f532d4eb3cd`
- **Origin**: user-provided snapshot, publisher field = "Singapore GTFS",
  `https://github.com/thecrapone/singapore-gtfs-2025`（community-constructed feed）
- **Feed dates**: 2025-01-01 – 2030-12-31（feed_version 1.0）
- **Archived**: 2026-08-24 (moved from workspace root)

## Content (verified 2026-08-24)

| Table | Rows |
|---|---|
| agencies | 6 (LTA, SBS Transit, SMRT, Tower Transit, Go-Ahead) |
| routes | 603 (593 bus route_type=3, 9 MRT route_type=1) |
| stops | 5,376 |
| trips | 230,915 |
| stop_times | 8,169,065 |
| calendar | 4 service patterns |

- Tampines + Pasir Ris study bbox (lat 1.330–1.400 × lon 103.900–104.015):
  **746 stops**.
- Timezone: Asia/Singapore.

## Publication wording (MANDATORY)

This feed is a **community-constructed GTFS snapshot** (singapore-gtfs-2025),
NOT an official LTA DataMall feed:

- Bus services/routes/stops are LTA-DM-based; **bus travel times include estimates**.
- **MRT schedules are frequency-based / synthetic**, not measured timetables.
- The paper must not describe this ZIP as an "official measured GTFS timetable".
