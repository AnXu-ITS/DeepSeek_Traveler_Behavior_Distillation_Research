#!/usr/bin/env python
"""Extend an existing K=3 aggregated teacher dataset with counterfactual
samples for NEW perturbation axes, reusing the existing baselines so no
baseline call is re-burned (and the new counterfactuals link to the exact
baselines they belong to). Used for the PROGRESS.md S1/S2/S4 augmentation.

Unlike ``generate_aggregated_teacher_dataset.py`` (which always emits a fresh
baseline per persona+trip), this derives new counterfactual states directly
from each EXISTING baseline's ``state``, so split_group_id / persona_group_id /
baseline_sample_id stay consistent with the dataset being extended.

Usage:
    python scripts/extend_teacher_dataset.py \
        --dataset data/student_v0_3/aggregated_teacher_dataset.jsonl \
        --axes road_congestion \
        --output data/student_v0_3_s1 \
        --workers 4 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

# Reuse the battle-tested per-state fetch/aggregate loop from the primary
# generator (scripts/ is sys.path[0], so the sibling module resolves).
from generate_aggregated_teacher_dataset import _load_jsonl, _process_state  # noqa: E402

from traveler_distillation.config import load_dotenv, load_yaml  # noqa: E402
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget  # noqa: E402
from traveler_distillation.generators import CounterfactualContextGenerator  # noqa: E402
from traveler_distillation.schemas.dataset import Perturbation  # noqa: E402
from traveler_distillation.teacher import (  # noqa: E402
    DeepSeekTeacherClient,
    MockTeacherClient,
    TeacherResponseParser,
    TeacherResponseValidator,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_new_states(existing: list[dict], axes: list[str], perturbations: dict, gen_cfg: dict) -> list[dict]:
    """Derive counterfactual states for ``axes`` from every existing baseline."""
    cf_gen = CounterfactualContextGenerator(gen_cfg)
    baselines = [
        AggregatedTeacherTarget.model_validate(r)
        for r in existing
        if (r.get("perturbation") or {}).get("axis") == "baseline"
    ]
    if not baselines:
        raise ValueError("no baselines found in the existing dataset")

    states: list[dict] = []
    cf_gid_seq = 0
    sample_seq = 0
    for base in baselines:
        for axis in axes:
            levels = perturbations[axis]["levels"]
            cf_gid_seq += 1
            gid = f"EXT_{axis}_{cf_gid_seq:06d}"
            for cs in cf_gen.generate(base.state, axis, levels):
                sample_seq += 1
                states.append(
                    {
                        "sample_id": f"EXT_{axis}_{sample_seq:06d}",
                        "cf_group_id": gid,
                        "baseline_sample_id": base.sample_id,
                        "split_group_id": base.split_group_id,
                        "persona_group_id": base.persona_group_id,
                        "perturbation": Perturbation(axis=cs.axis, level=cs.level),
                        "state": cs.state,
                    }
                )
    return states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--axes", required=True,
                    help="comma-separated perturbation axes to add (levels from config)")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--teacher-repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-bypass", action="store_true",
                    help="opt-in LiteLLM gateway cache-bypass (official DeepSeek does NOT need it)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    gen_cfg = load_yaml(args.config)
    t_section = load_yaml(args.teacher_config).get("teacher", {})
    perturbations = gen_cfg.get("perturbations", {})

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    unknown = [a for a in axes if a not in perturbations]
    if unknown:
        print(f"[FAIL] unknown perturbation axes: {unknown}; available: {sorted(perturbations)}")
        return 1

    src = Path(args.dataset)
    if not src.exists():
        print(f"[FAIL] dataset not found: {src}")
        return 1
    existing = _load_jsonl(src)

    new_states = build_new_states(existing, axes, perturbations, gen_cfg)
    n_calls = len(new_states) * args.teacher_repeats
    print(f"extend: {len(existing)} existing -> +{len(new_states)} new counterfactuals "
          f"(x{args.teacher_repeats} = {n_calls} teacher calls) axes={axes}")

    if args.dry_run:
        per_axis: dict[str, int] = defaultdict(int)
        for st in new_states:
            per_axis[st["perturbation"].axis] += 1
        print("per-axis new states:", dict(per_axis))
        print("DRY RUN: no API calls made")
        return 0

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

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = out / "aggregated_teacher_dataset.jsonl"
    repeats_path = out / "repeat_records.jsonl"
    incomplete_path = out / "incomplete_samples.jsonl"

    # Seed the output with the existing dataset so the combined file is
    # self-contained (validators/trainers read a single dataset + repeats pair).
    if not dataset_path.exists() or dataset_path.stat().st_size == 0:
        dataset_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in existing) + ("\n" if existing else ""),
            encoding="utf-8",
        )
    src_repeats = Path(src).parent / "repeat_records.jsonl"
    if (not repeats_path.exists() or repeats_path.stat().st_size == 0) and src_repeats.exists():
        repeats_path.write_text(src_repeats.read_text(encoding="utf-8"), encoding="utf-8")

    # resume: skip already-completed new states + reuse already-fetched repeats
    done_ids = {r["sample_id"] for r in _load_jsonl(dataset_path)}
    existing_repeats: dict[str, list] = defaultdict(list)
    for r in _load_jsonl(repeats_path):
        existing_repeats[r["sample_id"]].append(r)

    pending = [st for st in new_states if st["sample_id"] not in done_ids]
    print(f"resume: {len(new_states) - len(pending)} already done, {len(pending)} pending")

    stats = {"attempted": 0, "valid": 0, "failed": 0, "incomplete": 0, "aggregated": 0}
    write_lock = threading.Lock()

    def _write_result(res: dict) -> None:
        with write_lock:
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
            dataset_f.write(json.dumps(json.loads(res["target"].model_dump_json()), ensure_ascii=False) + "\n")
            dataset_f.flush()
            stats["aggregated"] += 1
            print(f"  [{res['sid']}] aggregated: {res['selected_mode']} {res['mode_probabilities']} "
                  f"shift={res['departure_shift']}", flush=True)

    use_cache_bypass = args.cache_bypass
    with dataset_path.open("a", encoding="utf-8") as dataset_f, \
            repeats_path.open("a", encoding="utf-8") as repeats_f, \
            incomplete_path.open("a", encoding="utf-8") as incomplete_f:

        def _submit(st: dict):
            return _process_state(
                st,
                teacher,
                parser,
                validator,
                args.teacher_repeats,
                args.mock,
                [r for r in existing_repeats[st["sample_id"]] if r.get("action") is not None],
                t_section,
                use_cache_bypass,
            )

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_submit, st): st for st in pending}
                for fut in as_completed(futures):
                    res = fut.result()
                    for k in ("attempted", "valid", "failed"):
                        stats[k] += res["local"][k]
                    _write_result(res)
        else:
            for st in pending:
                res = _submit(st)
                for k in ("attempted", "valid", "failed"):
                    stats[k] += res["local"][k]
                _write_result(res)

    manifest = {
        "extension_version": "extend_teacher_dataset_v0.1",
        "source_dataset": str(src),
        "axes": axes,
        "teacher_repeats": args.teacher_repeats,
        "endpoint": _env("DEEPSEEK_BASE_URL"),
        "model": getattr(teacher, "model", "mock_teacher"),
        "new_states": len(new_states),
        "stats": stats,
        "timestamp": _now(),
    }
    (out / "extension_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote combined dataset to {out}")
    return 0


def _env(key: str) -> str:
    import os
    return os.environ.get(key, "unknown")


if __name__ == "__main__":
    raise SystemExit(main())
