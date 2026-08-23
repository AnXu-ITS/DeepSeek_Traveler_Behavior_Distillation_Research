#!/usr/bin/env python
"""Phase 10: network feedback loop.

Loop: student decides under a context -> MATSim executes -> observed network
congestion is mapped back into the context -> student re-decides -> ... until
the congestion estimate converges (|c_obs - c_ctx| < eps) or max iterations.

Usage:
    python scripts/run_phase10_loop.py \
        --checkpoint outputs/student_v0_3_c/checkpoints/best.pt \
        --num-personas 300 --trips-per-persona 1 \
        --scenarios baseline,rain,fare_surge \
        --max-iterations 6 --eps 0.02 \
        --output data/phase10_loop
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.config import load_yaml
from traveler_distillation.generators import PersonaGenerator, TripGenerator
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.matsim import MATSimAdapter
from traveler_distillation.matsim.runner import (
    run_matsim,
    parse_modestats,
    parse_linkstats,
)

SCENARIOS = {
    "baseline": {"weather_intensity": 0.0, "fare_multiplier": 1.0},
    "rain": {"weather_intensity": 1.0, "fare_multiplier": 1.0},
    "fare_surge": {"weather_intensity": 0.0, "fare_multiplier": 2.0},
}


def make_context(name: str, gen_cfg: dict, congestion: float) -> DynamicContext:
    spec = SCENARIOS[name]
    baseline = gen_cfg.get("baseline_context", {})
    wi = spec["weather_intensity"]
    return DynamicContext(
        context_id=f"C_LOOP_{name}",
        weather=Weather(condition="clear" if wi <= 0 else "rain", intensity=wi),
        road_congestion=round(congestion, 4),
        transit_delay_min=baseline.get("transit_delay_min", 0),
        transit_disruption=baseline.get("transit_disruption", False),
        road_disruption=baseline.get("road_disruption", False),
        fare_multiplier=spec["fare_multiplier"],
        parking_cost_multiplier=baseline.get("parking_cost_multiplier", 1.0),
        congestion_charge=baseline.get("congestion_charge", 0.0),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--pop-seed", type=int, default=2026)
    ap.add_argument("--num-personas", type=int, default=300)
    ap.add_argument("--trips-per-persona", type=int, default=1)
    ap.add_argument("--grid-n", type=int, default=10,
                    help="grid size; 10x10 (denser per-link traffic) gives a "
                         "stronger congestion signal for the feedback loop")
    ap.add_argument("--link-capacity", type=float, default=100.0,
                    help="per-link hourly capacity (veh/h); 100 on the 10x10 grid "
                         "makes queues emerge at ~300 personas (documented choice)")
    ap.add_argument("--scenarios", default="baseline,rain,fare_surge")
    ap.add_argument("--max-iterations", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--initial-congestion", type=float, default=0.3)
    ap.add_argument("--smoothing", type=float, default=0.5,
                    help="c_new = (1-s)*c_obs + s*c_ctx (damping for the feedback loop)")
    ap.add_argument("--output", default="data/phase10_loop")
    ap.add_argument("--skip-matsim", action="store_true")
    args = ap.parse_args()

    gen_cfg = load_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    n = args.num_personas
    tpp = args.trips_per_persona
    personas = PersonaGenerator(seed=args.pop_seed, config=gen_cfg).generate(n)
    all_trips = TripGenerator(seed=args.pop_seed, config=gen_cfg).generate(n * tpp)
    trips_per_persona = [all_trips[i * tpp:(i + 1) * tpp] for i in range(n)]

    adapter = MATSimAdapter(args.checkpoint)
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    results = {}
    for name in scenario_names:
        if name not in SCENARIOS:
            continue
        print(f"\n===== feedback loop: {name} =====", flush=True)
        c_ctx = args.initial_congestion
        trace = []
        converged = False
        for it in range(1, args.max_iterations + 1):
            context = make_context(name, gen_cfg, c_ctx)
            sdir = out / name / f"iter{it}"
            manifest = adapter.build_scenario(
                personas, all_trips, context, sdir, grid_n=args.grid_n,
                trips_per_persona=trips_per_persona, link_capacity=args.link_capacity,
            )
            student_modes = Counter(m["student_mode"] for m in manifest)
            total = sum(student_modes.values())
            share = {m: round(c / total, 4) for m, c in student_modes.items()}

            rec = {
                "iteration": it,
                "context_congestion": round(c_ctx, 4),
                "student_mode_share": share,
                "matsim_mode_share": None,
                "observed_congestion": None,
                "mean_delay_ratio": None,
                "matsim_exit_code": None,
            }
            if not args.skip_matsim:
                code, tail = run_matsim(sdir)
                rec["matsim_exit_code"] = code
                rec["log_tail"] = tail[:400]
                if code != 0:
                    print(f"  iter {it}: MATSim FAILED (exit {code})", flush=True)
                    trace.append(rec)
                    break
                rec["matsim_mode_share"] = parse_modestats(sdir / "output" / "modestats.csv")
                # MATSim 2026 writes link stats to ITERS/it.0/0.linkstats.txt.zst
                linkstats = parse_linkstats(sdir / "output" / "ITERS" / "it.0" / "0.linkstats.txt.zst")
                rec["observed_congestion"] = linkstats.get("congestion")
                rec["mean_delay_ratio"] = linkstats.get("mean_delay_ratio")
                rec["n_cells"] = linkstats.get("n_cells", linkstats.get("n_links"))

            c_obs = rec["observed_congestion"]
            print(f"  iter {it}: c_ctx={c_ctx:.3f} -> mode={share} -> c_obs={c_obs}",
                  flush=True)
            trace.append(rec)

            if c_obs is None:
                print("  (no congestion measurement; stopping)")
                break
            if abs(c_obs - c_ctx) < args.eps:
                converged = True
                print(f"  CONVERGED at iter {it} (|c_obs - c_ctx| < {args.eps})", flush=True)
                break
            c_ctx = round((1 - args.smoothing) * c_obs + args.smoothing * c_ctx, 4)

        results[name] = {"converged": converged, "trace": trace}

    (out / "phase10_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out / 'phase10_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
