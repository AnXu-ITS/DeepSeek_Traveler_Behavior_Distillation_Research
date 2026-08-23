#!/usr/bin/env python
"""Generate the teacher dataset (v0.1).

Usage:
    python scripts/generate_teacher_dataset.py --dry-run
    python scripts/generate_teacher_dataset.py --mock --num-personas 5 --num-trips 2
    python scripts/generate_teacher_dataset.py --num-personas 10 --num-trips 3 --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.config import load_dotenv, load_yaml
from traveler_distillation.teacher import (
    DeepSeekTeacherClient,
    MockTeacherClient,
    TeacherResponseParser,
    TeacherResponseValidator,
)
from traveler_distillation.dataset import TeacherDatasetBuilder


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate teacher dataset v0.1")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--num-personas", type=int, default=None)
    ap.add_argument("--num-trips", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", default="data/teacher_v0_1")
    ap.add_argument("--dry-run", action="store_true", help="do not call the teacher API")
    ap.add_argument("--max-samples", type=int, default=None, help="cap total teacher calls")
    ap.add_argument("--mock", action="store_true", help="use deterministic mock teacher")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    gen_cfg = load_yaml(args.config)
    t_cfg = load_yaml(args.teacher_config)
    teacher_section = t_cfg.get("teacher", {})

    seed = args.seed if args.seed is not None else gen_cfg.get("generation", {}).get("seed", 42)
    num_personas = args.num_personas if args.num_personas is not None else 10
    num_trips = args.num_trips if args.num_trips is not None else 3

    parser_obj = TeacherResponseParser()
    validator = TeacherResponseValidator(
        probability_tolerance=teacher_section.get("probability_tolerance", 1e-3),
        departure_shift_min=teacher_section.get("departure_shift_min", -60),
        departure_shift_max=teacher_section.get("departure_shift_max", 60),
    )

    if args.mock or args.dry_run:
        teacher = MockTeacherClient()
    else:
        teacher = DeepSeekTeacherClient(
            model=teacher_section.get("model"),
            temperature=teacher_section.get("temperature", 0.2),
            max_retries=teacher_section.get("max_retries", 2),
            timeout_seconds=teacher_section.get("timeout_seconds", 120),
            max_tokens=teacher_section.get("max_tokens", 8192),
        )

    builder = TeacherDatasetBuilder(
        generation_config=gen_cfg,
        teacher_config=t_cfg,
        teacher_client=teacher,
        parser=parser_obj,
        validator=validator,
        output_dir=args.output,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    result = builder.build(
        num_personas=num_personas,
        num_trips=num_trips,
        seed=seed,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        s = result["statistics"]
        print("\n" + "=" * 50)
        print("GENERATION SUMMARY")
        print("=" * 50)
        print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
