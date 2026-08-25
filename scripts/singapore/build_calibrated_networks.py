#!/usr/bin/env python
"""Phase B.5C step 1: intersection-aware effective-capacity networks.

Uses the OSM traffic_signals nodes to apply green-ratio effective approach
capacities: every CAR link entering a signalized intersection gets
C_eff = C_osm * g/C, with g/C in {0.35, 0.45, 0.55} (three sensitivity
variants). All link ids and topology are untouched — only capacity values of
approach links change, so the transitSchedule and all routes stay valid.

Honest paper wording (per user instruction): "Signal timings were not
explicitly available; therefore, effective approach capacities were
represented using green-ratio sensitivity factors."

Usage:
    python scripts/singapore/build_calibrated_networks.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

_GREEN_RATIOS = (0.35, 0.45, 0.55)


def main() -> int:
    src = Path("data/singapore/transit/network_with_transit.xml")
    txt = src.read_text(encoding="utf-8")
    signal_ids = set(
        Path("data/singapore/osm/traffic_signal_nodes.txt").read_text(encoding="utf-8").split()
    )

    # approach links: car link whose TO node is a signalized intersection
    approach_ids = set()
    for m in re.finditer(
        r'<link id="([^"]+)" from="[^"]+" to="([^"]+)"[^>]*modes="([^"]+)"',
        txt,
    ):
        if "car" in m.group(3).split(",") and not m.group(1).startswith(("ai_", "busr_", "pdr_")):
            if m.group(2) in signal_ids:
                approach_ids.add(m.group(1))
    print(f"approach links to signalized intersections: {len(approach_ids)}", flush=True)

    for gc in _GREEN_RATIOS:
        def _sub(m):
            lid = m.group(1)
            cap = float(m.group(6))
            if lid in approach_ids:
                cap *= gc
            return (f'<link id="{lid}" from="{m.group(2)}" to="{m.group(3)}" '
                    f'length="{m.group(4)}" freespeed="{m.group(5)}" capacity="{cap:.1f}" '
                    f'permlanes="1.0" oneway="1" modes="{m.group(7)}"')
        out = re.sub(
            r'<link id="([^"]+)" from="([^"]+)" to="([^"]+)" length="([\d.]+)" '
            r'freespeed="([\d.]+)" capacity="([\d.]+)" permlanes="1.0" oneway="1" '
            r'modes="([^"]+)"',
            _sub, txt,
        )
        dst = Path(f"data/singapore/transit/network_gc{int(gc*100):02d}.xml")
        dst.write_text(out, encoding="utf-8")
        print(f"g/C={gc} -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
