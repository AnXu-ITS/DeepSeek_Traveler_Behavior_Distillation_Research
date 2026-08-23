#!/usr/bin/env python
"""Weather v0.1.1 re-audit: 3 persona+trip groups x 5 weather levels (15 calls).

Generates rain-intensity counterfactuals under the fixed semantics
(intensity=0 -> clear, intensity>0 -> rain) and records teacher responses
plus per-group response-curve metrics (Spearman rho, direction reversals,
delta-P from baseline).

Usage:
    python scripts/run_weather_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
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
from traveler_distillation.audit import spearman_rank, direction_reversals


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-groups", type=int, default=3)
    ap.add_argument("--output-dir", default="outputs/teacher_audit_v0_1/weather_semantics_v0_1_1")
    args = ap.parse_args()

    load_dotenv()
    gen_cfg = load_yaml(args.config)
    t_section = load_yaml(args.teacher_config).get("teacher", {})

    parser = TeacherResponseParser()
    validator = TeacherResponseValidator(
        probability_tolerance=t_section.get("probability_tolerance", 1e-3),
        departure_shift_min=t_section.get("departure_shift_min", -60),
        departure_shift_max=t_section.get("departure_shift_max", 60),
    )
    teacher = DeepSeekTeacherClient(
        model=t_section.get("model"),
        temperature=t_section.get("temperature", 0.2),
        max_retries=t_section.get("max_retries", 2),
        timeout_seconds=t_section.get("timeout_seconds", 120),
        max_tokens=t_section.get("max_tokens", 8192),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    personas = PersonaGenerator(seed=args.seed, config=gen_cfg).generate(args.num_groups)
    trips = TripGenerator(seed=args.seed, config=gen_cfg).generate(args.num_groups)
    baseline_gen = BaselineStateGenerator(gen_cfg)
    cf_gen = CounterfactualContextGenerator(gen_cfg)

    samples = []
    failed = []
    attempted = 0
    valid = 0

    for gi in range(args.num_groups):
        persona, trip = personas[gi], trips[gi]
        baseline = baseline_gen.generate(persona, trip)
        # level 0.0 = baseline (clear), levels > 0 = rain counterfactuals
        states_by_level = {0.0: baseline}
        for cs in cf_gen.generate(baseline, "weather_intensity", levels[1:]):
            states_by_level[float(cs.level)] = cs.state

        for level in levels:
            state = states_by_level[level]
            attempted += 1
            record = {
                "group_id": f"W{gi + 1:03d}",
                "level": level,
                "weather_condition": state.context.weather.condition,
                "weather_intensity": state.context.weather.intensity,
                "persona_id": persona.persona_id,
                "trip_id": trip.trip_id,
                "state": state.model_dump(mode="json"),
                "teacher": None,
                "failure_type": None,
                "timestamp": _now(),
            }
            try:
                raw = teacher.respond(state)
                action = parser.parse(raw)
                result = validator.validate(state, action)
                if not result.valid:
                    raise ValueError(f"validation failed: {result.reason}")
                record["teacher"] = action.model_dump(mode="json")
                valid += 1
                print(f"[{record['group_id']}] level={level} ({record['weather_condition']}) -> {action.selected_mode} {action.mode_probabilities}")
            except Exception as exc:
                ft = getattr(exc, "failure_type", "validation_failure" if "validation" in str(exc) else "api_error")
                record["failure_type"] = ft
                failed.append(record)
                print(f"[{record['group_id']}] level={level} FAILED ({ft}): {exc}")
            samples.append(record)

    # ---- write samples ----
    with (out / "weather_samples.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    if failed:
        with (out / "failed_samples.jsonl").open("w", encoding="utf-8") as f:
            for s in failed:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- metrics ----
    metrics = _compute_metrics(samples)
    (out / "weather_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = _build_report(metrics, attempted, valid, len(failed))
    (out / "weather_response_report.md").write_text(report, encoding="utf-8")

    print(f"\nattempted={attempted} valid={valid} failed={len(failed)}")
    print(f"wrote {out}")
    return 0


def _compute_metrics(samples: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        if s["teacher"] is not None:
            groups[s["group_id"]].append(s)

    per_group = []
    for gid in sorted(groups):
        rs = sorted(groups[gid], key=lambda s: s["level"])
        base_probs = rs[0]["teacher"]["mode_probabilities"] if rs else {}
        mode_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for s in rs:
            for mode, p in s["teacher"]["mode_probabilities"].items():
                mode_points[mode].append((s["level"], p))

        mode_stats = {}
        for mode in sorted(mode_points):
            pts = sorted(mode_points[mode])
            xs = [l for l, _ in pts]
            ys = [p for _, p in pts]
            rho = spearman_rank(xs, ys) if len(pts) >= 2 else 0.0
            revs = direction_reversals(ys)
            deltas = {
                lvl: round(p - base_probs.get(mode, 0.0), 4) for lvl, p in pts if lvl > 0.0
            }
            mode_stats[mode] = {
                "curve": [{"level": l, "prob": p} for l, p in pts],
                "spearman_rho": round(rho, 4),
                "direction_reversals": revs,
                "delta_p_from_baseline": deltas,
            }

        per_group.append(
            {
                "group_id": gid,
                "persona_id": rs[0]["persona_id"],
                "trip_id": rs[0]["trip_id"],
                "levels": [
                    {
                        "level": s["level"],
                        "weather_condition": s["weather_condition"],
                        "selected_mode": s["teacher"]["selected_mode"],
                        "mode_probabilities": s["teacher"]["mode_probabilities"],
                        "departure_time_shift_min": s["teacher"]["departure_time_shift_min"],
                        "confidence": s["teacher"]["confidence"],
                    }
                    for s in rs
                ],
                "mode_stats": mode_stats,
            }
        )
    return {"groups": per_group}


def _build_report(metrics: dict, attempted: int, valid: int, failed: int) -> str:
    lines = ["# Weather Semantic Re-Audit (v0.1.1)\n"]
    lines.append("weather_intensity is now defined as **rain intensity**:")
    lines.append("- intensity = 0.0 -> condition = clear")
    lines.append("- intensity > 0.0 -> condition = rain\n")
    lines.append(f"- attempted = {attempted}")
    lines.append(f"- valid = {valid}")
    lines.append(f"- failed = {failed}\n")

    for g in metrics["groups"]:
        lines.append(f"\n## Group {g['group_id']} (persona {g['persona_id']}, trip {g['trip_id']})\n")
        lines.append("| level | condition | selected | P(car) | P(pt) | P(bike) | P(walk) | shift | conf |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for lv in g["levels"]:
            probs = lv["mode_probabilities"]
            lines.append(
                f"| {lv['level']} | {lv['weather_condition']} | {lv['selected_mode']} | "
                f"{probs.get('car', '-')} | {probs.get('pt', '-')} | {probs.get('bike', '-')} | "
                f"{probs.get('walk', '-')} | {lv['departure_time_shift_min']} | {lv['confidence']} |"
            )
        lines.append("")
        for mode, m in g["mode_stats"].items():
            lines.append(
                f"- **{mode}**: spearman_rho = {m['spearman_rho']}, "
                f"direction_reversals = {m['direction_reversals']}, "
                f"delta_p = {m['delta_p_from_baseline']}"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
