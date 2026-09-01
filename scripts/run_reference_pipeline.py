#!/usr/bin/env python
"""Reference MATSim Integration Pipeline — one-command entry point.

Usage:
    python scripts/run_reference_pipeline.py --config configs/reference_example.yaml
    python scripts/run_reference_pipeline.py --config configs/reference_example.yaml --validate-only
    python scripts/run_reference_pipeline.py --config configs/reference_example.yaml --run-matsim

Exit codes: 0 ok · 1 input/config/build error · 3 MATSim run failed.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # repo root (reference_pipeline pkg)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # production package

from reference_pipeline import ReferencePipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/reference_example.yaml",
                    help="reference pipeline YAML config")
    ap.add_argument("--validate-only", action="store_true",
                    help="check config + population input + supply paths, then exit")
    ap.add_argument("--run-matsim", action="store_true",
                    help="run MATSim after building (overrides matsim.run_after_build)")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the route/accessibility cache for this run")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    try:
        pipeline = ReferencePipeline(args.config, repo_root)
        if args.no_cache:
            pipeline.cfg.cache.enabled = False
        if args.validate_only:
            report = pipeline.validate()
            print("\n=== VALIDATE-ONLY PASS ===")
            for k, v in report.items():
                print(f"{k}: {v}")
            return 0
        summary = pipeline.run(run_matsim_flag=True if args.run_matsim else None)
        print("\n=== BUILD DONE ===")
        print(f"output_dir: {pipeline.out_dir}")
        print(f"travelers: {summary['inputs']['num_travelers']}, "
              f"trips: {summary['inputs']['num_trips']}")
        print(f"student modes: {summary['mode_distribution']['student_mode_counts']}")
        print(f"total build: {summary['timings']['total_build_s']}s")
        if summary["cache"]:
            print(f"cache hit rate: {summary['cache']['hit_rate']*100:.1f}% "
                  f"({summary['cache']['hits']}/{summary['cache']['hits'] + summary['cache']['misses']})")
        if summary["matsim"] is not None:
            print(f"MATSim: exit={summary['matsim']['exit_code']} "
                  f"wall={summary['matsim']['wall_clock_s']}s")
            if summary["matsim"]["exit_code"] != 0:
                print("MATSim FAILED — see run_summary.json log_tail", file=sys.stderr)
                return 3
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
