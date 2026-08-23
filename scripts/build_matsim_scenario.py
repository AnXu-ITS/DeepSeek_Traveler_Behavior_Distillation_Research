#!/usr/bin/env python
"""Build a MATSim scenario from a distilled student checkpoint.

Generates synthetic personas/trips, asks the student for its decisions under a
given context, and writes network.xml / population.xml / config.xml ready for
``java ... org.matsim.run.Controler config.xml``.

Usage:
    python scripts/build_matsim_scenario.py \
        --checkpoint outputs/student_v0_2_a/checkpoints/best.pt \
        --num-personas 20 --num-trips 2 \
        --output data/matsim_smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.config import load_yaml
from traveler_distillation.generators import (
    PersonaGenerator,
    TripGenerator,
)
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.matsim import MATSimAdapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-personas", type=int, default=20)
    ap.add_argument("--num-trips", type=int, default=2)
    ap.add_argument("--weather-intensity", type=float, default=0.0,
                    help="0.0=clear, >0.0=rain (context scenario)")
    ap.add_argument("--fare-multiplier", type=float, default=1.0)
    ap.add_argument("--output", default="data/matsim_smoke")
    args = ap.parse_args()

    gen_cfg = load_yaml(args.config)
    personas = PersonaGenerator(seed=args.seed, config=gen_cfg).generate(args.num_personas)
    trips = TripGenerator(seed=args.seed, config=gen_cfg).generate(args.num_trips)

    baseline = gen_cfg.get("baseline_context", {})
    context = DynamicContext(
        context_id="C_MATSIM_SMOKE",
        weather=Weather(
            condition="clear" if args.weather_intensity <= 0 else "rain",
            intensity=args.weather_intensity,
        ),
        road_congestion=baseline.get("road_congestion", 0.3),
        transit_delay_min=baseline.get("transit_delay_min", 0),
        transit_disruption=baseline.get("transit_disruption", False),
        road_disruption=baseline.get("road_disruption", False),
        fare_multiplier=args.fare_multiplier,
        parking_cost_multiplier=baseline.get("parking_cost_multiplier", 1.0),
        congestion_charge=baseline.get("congestion_charge", 0.0),
    )

    adapter = MATSimAdapter(args.checkpoint)
    manifest = adapter.build_scenario(
        personas, trips, context, args.output, grid_n=20, spacing_m=1000.0
    )

    from collections import Counter
    mode_counts = Counter(m["student_mode"] for m in manifest)
    print(f"built scenario: {len(manifest)} trips, mode distribution={dict(mode_counts)}")
    print(f"files: network.xml, population.xml, config.xml under {args.output}")
    print("run with:  java -cp <matsim.jar;libs\\*> org.matsim.run.Controler config.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
