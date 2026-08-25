#!/usr/bin/env python
"""Generate the K=3 aggregated teacher development dataset (cache-busted).

Each state is queried K times with ``cache: {"no-cache": true}`` (real model
inference, no gateway response reuse), then the K behavioral preference
distributions are averaged into one AggregatedTeacherTarget.

Usage:
    python scripts/generate_aggregated_teacher_dataset.py --dry-run
    python scripts/generate_aggregated_teacher_dataset.py --mock --max-states 12
    python scripts/generate_aggregated_teacher_dataset.py --num-personas 3 --num-trips 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from traveler_distillation.generators import (
    PersonaGenerator,
    TripGenerator,
    BaselineStateGenerator,
    CounterfactualContextGenerator,
)
from traveler_distillation.schemas.dataset import Perturbation
from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.dataset.aggregation import (
    AggregatedTeacherTarget,
    AggregationMetadata,
    aggregate_repeats,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def enumerate_states(
    gen_cfg: dict,
    num_personas: int,
    num_trips: int,
    seed: int,
    axes: list[str] | None = None,
    persona_offset: int = 0,
    trip_offset: int = 0,
    id_prefix: str = "S",
) -> list[dict]:
    personas = PersonaGenerator(seed=seed, config=gen_cfg).generate(num_personas, id_offset=persona_offset)
    trips = TripGenerator(seed=seed, config=gen_cfg).generate(num_trips, id_offset=trip_offset)
    perturbations = gen_cfg.get("perturbations", {})
    if axes is not None:
        unknown = [a for a in axes if a not in perturbations]
        if unknown:
            raise ValueError(f"unknown perturbation axes: {unknown}; available: {sorted(perturbations)}")
        perturbations = {a: perturbations[a] for a in axes}
    baseline_gen = BaselineStateGenerator(gen_cfg)
    cf_gen = CounterfactualContextGenerator(gen_cfg)

    states = []
    seq = 0
    cf_seq = 0
    for persona in personas:
        for trip in trips:
            # One split group per persona+trip pair: baseline + ALL its
            # counterfactual curves stay together in a single split, which is
            # required for counterfactual/elasticity evaluation.
            split_group_id = f"{persona.persona_id}::{trip.trip_id}"
            baseline = baseline_gen.generate(persona, trip)
            seq += 1
            sid = f"{id_prefix}{seq:06d}"
            states.append(
                {
                    "sample_id": sid,
                    "cf_group_id": None,
                    "baseline_sample_id": None,
                    "split_group_id": split_group_id,
                    "persona_group_id": persona.persona_id,
                    "perturbation": Perturbation(axis="baseline", level=0.0),
                    "state": baseline,
                }
            )
            baseline_id = sid
            for axis, spec in perturbations.items():
                cf_seq += 1
                gid = f"CF_{axis}_{cf_seq:06d}"
                for cs in cf_gen.generate(baseline, axis, spec.get("levels", [])):
                    seq += 1
                    sid = f"{id_prefix}{seq:06d}"
                    states.append(
                        {
                            "sample_id": sid,
                            "cf_group_id": gid,
                            "baseline_sample_id": baseline_id,
                            "split_group_id": split_group_id,
                            "persona_group_id": persona.persona_id,
                            "perturbation": Perturbation(axis=cs.axis, level=cs.level),
                            "state": cs.state,
                        }
                    )
    return states


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _process_state(
    st: dict,
    teacher,
    parser,
    validator,
    teacher_repeats: int,
    is_mock: bool,
    existing_valid: list[dict],
    t_section: dict,
    use_cache_bypass: bool = True,
    clip_departure: bool = False,
) -> dict:
    """Fetch/parse/validate K uncached repeats for one state and aggregate.

    Thread-safe: ``teacher``/``parser``/``validator`` carry only read-only
    config, and the DeepSeek client opens a fresh ``httpx.Client`` per call.
    Returns a dict with the aggregated target (if complete), any NEW repeat
    records, a status, and this state's local attempt/valid/failed counts.

    ``use_cache_bypass`` sends LiteLLM's ``cache: {"no-cache": true}`` request
    field — only valid against the LiteLLM gateway. For the official DeepSeek
    API it must be False (no gateway cache exists there anyway).
    """
    sid = st["sample_id"]
    valid_repeats = list(existing_valid)
    seen_completion_ids = {r.get("completion_id") for r in valid_repeats if r.get("completion_id")}
    new_records: list[dict] = []
    local = {"attempted": 0, "valid": 0, "failed": 0}
    attempts = 0

    while len(valid_repeats) < teacher_repeats and attempts < teacher_repeats + 6:
        attempts += 1
        local["attempted"] += 1
        try:
            if is_mock:
                raw_content = teacher.respond(st["state"])
                evidence = {
                    "completion_id": f"mock-{sid}-{len(valid_repeats)}",
                    "created": None,
                    "provider_cache_status": "mock",
                    "elapsed_seconds": 0.0,
                }
            else:
                raw = teacher.respond_with_evidence(st["state"], cache_bypass=use_cache_bypass)
                raw_content = raw["content"]
                evidence = raw["evidence"]
        except Exception as exc:
            local["failed"] += 1
            print(f"  [{sid}] repeat {len(valid_repeats)} FAILED: {exc}", flush=True)
            continue

        cid = evidence.get("completion_id")
        if (not is_mock) and cid in seen_completion_ids:
            # gateway returned a cached completion object -> do not count
            print(f"  [{sid}] cache-bypass failure (duplicate completion {str(cid)[:8]}), retrying", flush=True)
            local["failed"] += 1
            continue

        try:
            action = parser.parse(raw_content, clip_departure=clip_departure)
            res = validator.validate(st["state"], action)
            if not res.valid:
                raise ValueError(res.reason)
        except Exception as exc:
            local["failed"] += 1
            print(f"  [{sid}] repeat parse/validation failed: {exc}", flush=True)
            continue

        if cid:
            seen_completion_ids.add(cid)
        rec = {
            "sample_id": sid,
            "repeat_index": len(valid_repeats),
            "completion_id": cid,
            "created": evidence.get("created"),
            "provider_cache_status": evidence.get("response_headers", {}).get("llm_provider-eo-cache-status")
            if not is_mock else "mock",
            "elapsed_seconds": evidence.get("elapsed_seconds"),
            "usage": evidence.get("usage"),
            "action": action.model_dump(mode="json"),
        }
        valid_repeats.append(rec)
        new_records.append(rec)
        local["valid"] += 1

    result = {
        "sid": sid,
        "status": "incomplete" if len(valid_repeats) < teacher_repeats else "complete",
        "n_valid": len(valid_repeats),
        "local": local,
        "new_records": new_records,
        "target": None,
    }
    if result["status"] == "incomplete":
        return result

    actions = [UniversalTravelerAction.model_validate(r["action"]) for r in valid_repeats]
    agg = aggregate_repeats(actions)
    result["target"] = AggregatedTeacherTarget(
        sample_id=sid,
        counterfactual_group_id=st["cf_group_id"],
        persona_group_id=st.get("persona_group_id"),
        baseline_sample_id=st["baseline_sample_id"],
        split_group_id=st.get("split_group_id"),
        perturbation=st["perturbation"],
        state=st["state"],
        teacher_aggregate=agg,
        aggregation_metadata=AggregationMetadata(
            k=teacher_repeats,
            aggregation_method="mean_probability",
            prompt_version=t_section.get("prompt_version", "teacher_v0.1"),
            model=getattr(teacher, "model", "unknown"),
            source_completion_ids=[r["completion_id"] for r in valid_repeats if r.get("completion_id")],
        ),
        status="complete",
    )
    result["selected_mode"] = agg.selected_mode
    result["mode_probabilities"] = agg.mode_probabilities
    result["departure_shift"] = agg.departure_time_shift_min
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-personas", type=int, default=3)
    ap.add_argument("--num-trips", type=int, default=1)
    ap.add_argument("--persona-offset", type=int, default=0,
                    help="persona ids start at P{offset+1:06d} (population extension)")
    ap.add_argument("--trip-offset", type=int, default=0,
                    help="trip ids start at T{offset+1:06d} (population extension)")
    ap.add_argument("--id-prefix", type=str, default="S",
                    help="sample id prefix (population extension without collisions)")
    ap.add_argument("--max-states", type=int, default=None)
    ap.add_argument("--teacher-repeats", type=int, default=3)
    ap.add_argument("--axes", type=str, default=None,
                    help="comma-separated subset of perturbation axes to generate "
                         "(default: all axes in the config)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of concurrent state workers (1 = sequential)")
    ap.add_argument("--no-cache-bypass", action="store_true",
                    help="do NOT send LiteLLM cache-control metadata (use for the "
                         "official DeepSeek API; the gateway is the only target that "
                         "understands it and the only one that caches)")
    ap.add_argument("--output", default="data/student_v0_2_a")
    args = ap.parse_args()

    load_dotenv()
    gen_cfg = load_yaml(args.config)
    t_section = load_yaml(args.teacher_config).get("teacher", {})
    seed = args.seed if args.seed is not None else gen_cfg.get("generation", {}).get("seed", 42)

    parser = TeacherResponseParser()
    validator = TeacherResponseValidator(
        probability_tolerance=t_section.get("probability_tolerance", 1e-3),
        departure_shift_min=t_section.get("departure_shift_min", -60),
        departure_shift_max=t_section.get("departure_shift_max", 60),
    )

    axes = [a.strip() for a in args.axes.split(",") if a.strip()] if args.axes else None
    states = enumerate_states(
        gen_cfg, args.num_personas, args.num_trips, seed, axes=axes,
        persona_offset=args.persona_offset, trip_offset=args.trip_offset,
        id_prefix=args.id_prefix,
    )
    if args.max_states is not None:
        states = states[: args.max_states]

    if args.dry_run:
        print(f"DRY RUN: {len(states)} states x {args.teacher_repeats} repeats = "
              f"{len(states) * args.teacher_repeats} teacher calls")
        print(f"personas={args.num_personas} trips={args.num_trips} seed={seed} "
              f"axes={axes or sorted(gen_cfg.get('perturbations', {}))}")
        return 0

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = out / "aggregated_teacher_dataset.jsonl"
    repeats_path = out / "repeat_records.jsonl"
    incomplete_path = out / "incomplete_samples.jsonl"

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

    # resume state
    completed_ids = set()
    if args.resume:
        for rec in _load_jsonl(dataset_path):
            completed_ids.add(rec["sample_id"])
    existing_repeats = defaultdict(list)
    for rec in _load_jsonl(repeats_path):
        existing_repeats[rec["sample_id"]].append(rec)

    fmode = "a" if args.resume else "w"
    dataset_f = dataset_path.open(fmode, encoding="utf-8")
    repeats_f = repeats_path.open(fmode, encoding="utf-8")
    incomplete_f = incomplete_path.open(fmode, encoding="utf-8")
    stats = {"attempted": 0, "valid": 0, "failed": 0, "incomplete": 0, "aggregated": 0}
    write_lock = threading.Lock()

    pending = [st for st in states if st["sample_id"] not in completed_ids]

    def _write_result(res: dict) -> None:
        with write_lock:
            # persist any newly-fetched repeats even when the state is still
            # incomplete, so a later --resume sweep reuses them instead of
            # re-burning API calls
            for rec in res.get("new_records", []):
                repeats_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            repeats_f.flush()
            if res["status"] == "incomplete":
                stats["incomplete"] += 1
                incomplete_f.write(
                    json.dumps(
                        {
                            "sample_id": res["sid"],
                            "valid_repeats": res["n_valid"],
                            "required_repeats": args.teacher_repeats,
                            "status": "incomplete",
                            "timestamp": _now(),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
                incomplete_f.flush()
                print(f"  [{res['sid']}] INCOMPLETE ({res['n_valid']}/{args.teacher_repeats})", flush=True)
                return
            target = res["target"]
            dataset_f.write(json.dumps(json.loads(target.model_dump_json()), ensure_ascii=False) + "\n")
            dataset_f.flush()
            stats["aggregated"] += 1
            print(f"  [{res['sid']}] aggregated: {res['selected_mode']} {res['mode_probabilities']} "
                  f"shift={res['departure_shift']}", flush=True)

    use_cache_bypass = not args.no_cache_bypass
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    _process_state,
                    st,
                    teacher,
                    parser,
                    validator,
                    args.teacher_repeats,
                    args.mock,
                    [r for r in existing_repeats[st["sample_id"]] if r.get("action") is not None],
                    t_section,
                    use_cache_bypass,
                ): st
                for st in pending
            }
            for fut in as_completed(futures):
                res = fut.result()
                for k in ("attempted", "valid", "failed"):
                    stats[k] += res["local"][k]
                _write_result(res)
    else:
        for st in pending:
            existing_valid = [r for r in existing_repeats[st["sample_id"]] if r.get("action") is not None]
            res = _process_state(
                st, teacher, parser, validator, args.teacher_repeats, args.mock,
                existing_valid, t_section, use_cache_bypass,
            )
            for k in ("attempted", "valid", "failed"):
                stats[k] += res["local"][k]
            _write_result(res)

    dataset_f.close()
    repeats_f.close()
    incomplete_f.close()

    manifest = {
        "dataset_version": "student_v0_2_a_aggregated",
        "teacher_repeats": args.teacher_repeats,
        "aggregation_method": "mean_probability",
        "cache_bypass": use_cache_bypass,
        "endpoint": os.environ.get("DEEPSEEK_BASE_URL", "unknown"),
        "model": getattr(teacher, "model", "mock_teacher"),
        "seed": seed,
        "num_personas": args.num_personas,
        "num_trips": args.num_trips,
        "axes": axes or sorted(gen_cfg.get("perturbations", {})),
        "stats": stats,
        "timestamp": _now(),
    }
    (out / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
