#!/usr/bin/env python
"""Smoke test: 1 persona, 1 trip, baseline + 2 perturbations, real teacher."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.config import load_dotenv, load_yaml
from traveler_distillation.teacher import (
    DeepSeekTeacherClient,
    TeacherResponseParser,
    TeacherResponseValidator,
)
from traveler_distillation.generators import (
    PersonaGenerator,
    TripGenerator,
    BaselineStateGenerator,
    CounterfactualContextGenerator,
)


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    gen_cfg = load_yaml("configs/generation_v0_1.yaml")
    t_cfg = load_yaml("configs/teacher_v0_1.yaml")
    teacher_section = t_cfg.get("teacher", {})

    seed = 42
    personas = PersonaGenerator(seed=seed, config=gen_cfg).generate(1)
    trips = TripGenerator(seed=seed, config=gen_cfg).generate(1)
    persona, trip = personas[0], trips[0]

    baseline = BaselineStateGenerator(gen_cfg).generate(persona, trip)
    cf_gen = CounterfactualContextGenerator(gen_cfg)

    # two perturbations: weather 0.5 and fare 2.0
    states = [
        ("baseline", baseline),
    ]
    states += [("weather_intensity=0.5", cs.state) for cs in cf_gen.generate(baseline, "weather_intensity", [0.5])]
    states += [("fare_multiplier=2.0", cs.state) for cs in cf_gen.generate(baseline, "fare_multiplier", [2.0])]

    client = DeepSeekTeacherClient(
        model=teacher_section.get("model"),
        temperature=teacher_section.get("temperature", 0.2),
        max_retries=teacher_section.get("max_retries", 2),
        timeout_seconds=teacher_section.get("timeout_seconds", 120),
        max_tokens=teacher_section.get("max_tokens", 8192),
    )
    parser = TeacherResponseParser()
    validator = TeacherResponseValidator(
        probability_tolerance=teacher_section.get("probability_tolerance", 1e-3),
    )

    for label, state in states:
        print("\n" + "=" * 60)
        print(f"CASE: {label}")
        print("available modes:", state.available_modes)
        try:
            raw = client.respond(state)
            action = parser.parse(raw)
            result = validator.validate(state, action)
            print("teacher action:", json.dumps(action.model_dump(), ensure_ascii=False, indent=2))
            print("validation:", result.valid, result.reason)
        except Exception as exc:
            print("FAILED:", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
