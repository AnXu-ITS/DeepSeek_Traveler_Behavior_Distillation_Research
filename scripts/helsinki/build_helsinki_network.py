#!/usr/bin/env python
"""Helsinki OSM -> MATSim network (E5 Phase A step; no shared-code changes).

`src/traveler_distillation/singapore/osm_network.py::build_network` hardcodes
the Singapore projection (`utm48n`) at module level. Per the E5 integration
contract (design §6: shared code unchanged, Singapore defaults untouched), the
projection is swapped IN THIS PROCESS ONLY (`osm_network.utm48n = utm35n`),
then the stock `build_network` runs unchanged. The patch never affects the
Singapore pipeline (separate process).

QA-1 parity summary is printed against the frozen Singapore network stats.

Usage:
    python scripts/helsinki/build_helsinki_network.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WB / "src"))

from traveler_distillation.singapore import osm_network  # noqa: E402
from traveler_distillation.singapore.osm_network import build_network  # noqa: E402
from traveler_distillation.singapore.projection import utm35n  # noqa: E402

# process-local projection swap (Helsinki = UTM 35N); Singapore path unaffected
osm_network.utm48n = utm35n


def main() -> int:
    osm_path = _WB / "data/helsinki/osm/helsinki_sub.osm"
    net_path = _WB / "data/helsinki/osm/network.xml"
    stats_path = _WB / "data/helsinki/osm/network_stats.json"
    stats = build_network(osm_path, net_path, stats_path)

    sg = json.loads((_WB / "data/singapore/osm/network_stats.json").read_text(encoding="utf-8"))
    print("\n=== QA-1 parity (Singapore frozen -> Helsinki measured) ===")
    print(f"ways_kept:   {sg['ways_kept']} -> {stats['ways_kept']} ({stats['ways_kept'] / sg['ways_kept']:.2f}x)")
    print(f"nodes:       {sg['nodes']} -> {stats['nodes']} ({stats['nodes'] / sg['nodes']:.2f}x)")
    print(f"links:       {sg['links']} -> {stats['links']} ({stats['links'] / sg['links']:.2f}x)")
    print(f"total_km:    {sg['total_network_km']} -> {stats['total_network_km']} ({stats['total_network_km'] / sg['total_network_km']:.2f}x)")
    for m in ("car", "bike", "walk"):
        print(f"connectivity {m}: {sg['connectivity'][m]['fraction_in_largest']} -> {stats['connectivity'][m]['fraction_in_largest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
