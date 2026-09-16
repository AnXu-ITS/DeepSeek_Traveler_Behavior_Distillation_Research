#!/usr/bin/env python
"""Shanghai transit build: GTFS prep -> scheduled-transit MATSim supply.

One command (after the GTFS zip is present, see ``download_shanghai_gtfs.py``):

    python scripts/shanghai/build_shanghai_transit.py data/shanghai/gtfs/raw/shanghai-gtfs.zip

Steps:
  1. ``prep_shanghai_gtfs.prep_gtfs`` — region-filter stops/trips (Xuhui bbox,
     metro-only route mapping, UTM 51N stop coords) -> prep_stops/prep_trips.jsonl;
  2. ``build_transit`` — snap stops to the road network, route each trip's stop
     sequence (rail-on-road, same Phase A degradation as Singapore/Helsinki),
     write transitSchedule.xml / transitVehicles.xml / network_with_transit.xml
     + snapping/routing reports + trips_by_stop.json + activity_nodes.json.

The road network ``data/shanghai/osm/network.xml`` must already exist
(``build_shanghai_network.py``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WB / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from traveler_distillation.singapore.build_transit import build_transit  # noqa: E402
from prep_shanghai_gtfs import prep_gtfs  # noqa: E402

TIME_WINDOW = (5 * 3600, 23 * 3600)  # 05:00-23:00, same as Singapore/Helsinki


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--network", default=str(_WB / "data/shanghai/osm/network.xml"))
    ap.add_argument("--out", default=str(_WB / "data/shanghai/transit"))
    args = ap.parse_args()

    prep_gtfs(args.zip_path, args.out)
    build_transit(
        args.network,
        Path(args.out) / "prep_stops.jsonl",
        Path(args.out) / "prep_trips.jsonl",
        args.out,
        time_windows=[TIME_WINDOW],
    )
    print("\nShanghai transit supply written to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
