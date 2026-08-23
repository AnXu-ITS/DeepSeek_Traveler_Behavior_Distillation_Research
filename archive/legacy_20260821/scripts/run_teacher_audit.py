#!/usr/bin/env python
"""Run the teacher audit (real API): pilot + repeatability + persona contrast + ablation.

Usage:
    python scripts/run_teacher_audit.py --mock                 # offline smoke
    python scripts/run_teacher_audit.py                        # real API (~95 calls)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
from traveler_distillation.generators import (
    PersonaGenerator,
    TripGenerator,
    BaselineStateGenerator,
    AlternativeGenerator,
)
from traveler_distillation.schemas.dataset import TeacherDatasetSample, TeacherMetadata
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.audit import state_hash, RepeatabilityRecord, PersonaContrastRecord

logger = logging.getLogger("audit")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _call_teacher(teacher, parser, validator, state: UniversalTravelerState):
    """Call teacher -> parse -> validate. Returns action or None on any failure."""
    try:
        raw = teacher.respond(state)
        action = parser.parse(raw)
        result = validator.validate(state, action)
        if not result.valid:
            logger.warning("invalid teacher response: %s", result.reason)
            return None
        return action
    except Exception as exc:  # TeacherClientError / parse / validation
        logger.warning("teacher call failed: %s", exc)
        return None


def _read_pilot(path: Path) -> list[TeacherDatasetSample]:
    samples = []
    if not path.exists():
        return samples
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(TeacherDatasetSample.model_validate(json.loads(line)))
    return samples


def _append_jsonl(path: Path, records: list) -> None:
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _select_repeatability_states(pilot: list[TeacherDatasetSample]) -> list[TeacherDatasetSample]:
    by_axis = defaultdict(list)
    for s in pilot:
        if s.teacher is not None and s.perturbation.axis != "baseline":
            by_axis[s.perturbation.axis].append(s)
    picks: list[TeacherDatasetSample] = []

    def take(axis, k):
        for s in by_axis.get(axis, [])[:k]:
            picks.append(s)

    take("weather_intensity", 2)
    take("road_congestion", 2)
    take("transit_delay", 1)
    take("fare_multiplier", 1)
    take("parking_cost_multiplier", 1)
    take("road_disruption", 1)
    return picks


def _run_repeatability(teacher, parser, validator, states, out_path, metadata, t_section):
    records = []
    for i, sample in enumerate(states, 1):
        h = state_hash(sample.state)
        for rep in range(3):
            action = _call_teacher(teacher, parser, validator, sample.state)
            records.append(
                RepeatabilityRecord(
                    audit_state_id=f"A{i:03d}",
                    source_sample_id=sample.sample_id,
                    repeat_index=rep,
                    state_hash=h,
                    state=sample.state,
                    teacher=action,
                    teacher_metadata=metadata,
                    timestamp=_now(),
                    force_repeat=True,
                ).model_dump(mode="json")
            )
            logger.info("repeatability A%03d rep%d: %s", i, rep, "ok" if action else "failed")
    _append_jsonl(out_path, records)


def _run_persona_contrast(gen_cfg, teacher, parser, validator, out_path, metadata, seed):
    persona = PersonaGenerator(seed=seed, config=gen_cfg).generate(1)[0]
    # ensure the car_ownership contrast is meaningful (license held)
    if not persona.driving_license:
        persona = persona.model_copy(update={"driving_license": True})
    trip = TripGenerator(seed=seed, config=gen_cfg).generate(1)[0]
    ctx = BaselineStateGenerator(gen_cfg).baseline_context(persona.persona_id, trip.trip_id)
    alt_gen = AlternativeGenerator(gen_cfg)

    groups = {
        "income_group": ["low", "medium", "high"],
        "car_ownership": [False, True],
        "schedule_flexibility": ["low", "medium", "high"],
        "habitual_mode": ["car", "pt", "bike"],
        "mobility_limitation": ["none", "mild", "significant"],
    }
    records = []
    aid = 0
    for attr, values in groups.items():
        gid = f"PG_{attr}_000001"
        for val in values:
            aid += 1
            p = persona.model_copy(deep=True)
            setattr(p, attr, val)
            state = UniversalTravelerState(
                persona=p,
                trip=trip,
                context=ctx,
                alternatives=alt_gen.generate(p, trip, ctx),
            )
            action = _call_teacher(teacher, parser, validator, state)
            records.append(
                PersonaContrastRecord(
                    audit_state_id=f"A{aid:03d}",
                    kind="persona_contrast",
                    persona_group_id=gid,
                    attribute=attr,
                    state_hash=state_hash(state),
                    state=state,
                    teacher=action,
                    teacher_metadata=metadata,
                    timestamp=_now(),
                ).model_dump(mode="json")
            )
            logger.info("persona contrast %s=%s: %s", attr, val, "ok" if action else "failed")
    _append_jsonl(out_path, records)


def _run_ablation(pilot, teacher, parser, validator, out_path, metadata, gen_cfg):
    picks = [s for s in pilot if s.teacher is not None][:3]
    if not picks:
        logger.warning("no pilot samples for ablation")
        return
    alt_gen = AlternativeGenerator(gen_cfg)
    modifications = [("habitual_mode", "mixed"), ("income_group", "medium")]
    records = []
    for i, sample in enumerate(picks, 1):
        for attr, val in modifications:
            p = sample.state.persona.model_copy(deep=True)
            setattr(p, attr, val)
            state = UniversalTravelerState(
                persona=p,
                trip=sample.state.trip,
                context=sample.state.context,
                alternatives=alt_gen.generate(p, sample.state.trip, sample.state.context),
            )
            action = _call_teacher(teacher, parser, validator, state)
            records.append(
                PersonaContrastRecord(
                    audit_state_id=f"A{900 + i:03d}",
                    kind="ablation",
                    persona_group_id=None,
                    attribute=attr,
                    state_hash=state_hash(state),
                    state=state,
                    teacher=action,
                    teacher_metadata=metadata,
                    timestamp=_now(),
                    note=f"change {attr}->{val} from {sample.sample_id}",
                ).model_dump(mode="json")
            )
            logger.info("ablation %s->%s: %s", attr, val, "ok" if action else "failed")
    _append_jsonl(out_path, records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-personas", type=int, default=2)
    ap.add_argument("--num-trips", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--output-dir", default="outputs/teacher_audit_v0_1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume pilot generation: skip existing sample_ids; max-samples counts only new samples")
    ap.add_argument("--skip-repeatability", action="store_true")
    ap.add_argument("--skip-persona", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-pilot", action="store_true", help="skip pilot generation; reuse existing pilot_samples.jsonl")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    gen_cfg = load_yaml(args.config)
    t_cfg = load_yaml(args.teacher_config)
    t_section = t_cfg.get("teacher", {})
    seed = args.seed if args.seed is not None else gen_cfg.get("generation", {}).get("seed", 42)

    parser = TeacherResponseParser()
    validator = TeacherResponseValidator(
        probability_tolerance=t_section.get("probability_tolerance", 1e-3),
        departure_shift_min=t_section.get("departure_shift_min", -60),
        departure_shift_max=t_section.get("departure_shift_max", 60),
    )

    if args.mock:
        teacher = MockTeacherClient()
    else:
        teacher = DeepSeekTeacherClient(
            model=t_section.get("model"),
            temperature=t_section.get("temperature", 0.2),
            max_retries=t_section.get("max_retries", 2),
            timeout_seconds=t_section.get("timeout_seconds", 120),
            max_tokens=t_section.get("max_tokens", 8192),
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = TeacherMetadata(
        model=getattr(teacher, "model", "unknown"),
        prompt_version=t_section.get("prompt_version", "teacher_v0.1"),
        dataset_version=gen_cfg.get("dataset", {}).get("version", "unknown"),
    )

    # ---- 1. pilot (50 samples) ----
    if not args.skip_pilot:
        builder = TeacherDatasetBuilder(
            gen_cfg, t_cfg, teacher, parser, validator,
            output_dir=out,
            overwrite=args.overwrite,
            resume=args.resume,
            dataset_filename="pilot_samples.jsonl",
            failed_filename="failed_samples.jsonl",
        )
        builder.build(
            num_personas=args.num_personas,
            num_trips=args.num_trips,
            seed=seed,
            max_samples=args.max_samples,
        )
    pilot = _read_pilot(out / "pilot_samples.jsonl")
    logger.info("pilot samples read: %d", len(pilot))

    # ---- 2. repeatability ----
    if not args.skip_repeatability:
        states = _select_repeatability_states(pilot)
        logger.info("repeatability states selected: %d", len(states))
        _run_repeatability(teacher, parser, validator, states, out / "repeatability_samples.jsonl", metadata, t_section)

    # ---- 3. persona contrast ----
    if not args.skip_persona:
        _run_persona_contrast(gen_cfg, teacher, parser, validator, out / "persona_contrast_samples.jsonl", metadata, seed)

    # ---- 4. ablation ----
    if not args.skip_ablation:
        _run_ablation(pilot, teacher, parser, validator, out / "persona_contrast_samples.jsonl", metadata, gen_cfg)

    # ---- manifest ----
    manifest = {
        "audit_version": "teacher_audit_v0.1",
        "model": getattr(teacher, "model", "unknown"),
        "prompt_version": t_section.get("prompt_version", "teacher_v0.1"),
        "seed": seed,
        "num_personas": args.num_personas,
        "num_trips": args.num_trips,
        "pilot_max_samples": args.max_samples,
        "timestamp": _now(),
    }
    (out / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("audit complete. outputs in %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
