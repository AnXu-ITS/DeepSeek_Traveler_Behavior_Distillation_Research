#!/usr/bin/env python
"""Shanghai (Xuhui) OSM -> MATSim network (E6 Phase A step; no shared-code changes).

Mirrors ``scripts/helsinki/build_helsinki_network.py``: the stock
``build_network`` hardcodes the Singapore projection (``utm48n``) at module
level, so this script swaps it IN THIS PROCESS ONLY to ``utm51n`` (Shanghai,
zone 51N / central meridian 123E). The patch never affects the Singapore path
(separate process).

Usage:
    python scripts/shanghai/build_shanghai_network.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WB / "src"))

from traveler_distillation.singapore import osm_network  # noqa: E402
from traveler_distillation.singapore.osm_network import build_network  # noqa: E402
from traveler_distillation.singapore.projection import utm51n  # noqa: E402

# process-local projection swap (Shanghai = UTM 51N); Singapore path unaffected
osm_network.utm48n = utm51n


def main() -> int:
    osm_path = _WB / "data/shanghai/osm/xuhui.osm"
    net_path = _WB / "data/shanghai/osm/network.xml"
    stats_path = _WB / "data/shanghai/osm/network_stats.json"
    stats = build_network(osm_path, net_path, stats_path)
    print(
        f"\nXuhui network: ways={stats['ways_kept']} nodes={stats['nodes']} "
        f"links={stats['links']} km={stats['total_network_km']} "
        f"car_conn={stats['connectivity']['car']['fraction_in_largest']} "
        f"bike_conn={stats['connectivity']['bike']['fraction_in_largest']} "
        f"walk_conn={stats['connectivity']['walk']['fraction_in_largest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
