#!/usr/bin/env python
"""S5 multi-axis / joint teacher dataset generator.

Derives joint (two-axis) counterfactual states from the EXISTING S3 baselines
(data/student_v0_3_s3/aggregated_teacher_dataset.jsonl), reusing each baseline's
split_group_id / persona_group_id / baseline_sample_id so the joint states stay
linked to their baseline AND to the single-axis counterparts already in S3 (the
interaction metric links joint -> baseline + two single-axis states by the same
split_group_id + axis + level).

K strategy (from configs/joint_sampling.yaml):
  - high-SNR combos (weather / road_disruption) : K = 3
  - low-SNR combos  (fare / congestion / delay)  : K = 5
  - K=7 causal subset (a deterministic slice of a low-SNR combo) : K = 7

Two aggregated views are written:
  - aggregated_teacher_dataset.jsonl     -> FULL-K target (M2 view; subset = K=7)
  - aggregated_teacher_dataset_k5.jsonl  -> base-K target (M1 view; subset states
    re-aggregated from the FIRST base-K of their 7 repeats)

Usage:
    python scripts/generate_joint_teacher_dataset.py --dry-run
    python scripts/generate_joint_teacher_dataset.py --combos rain_x_congestion --max-baselines 1 --override-k 1
    python scripts/generate_joint_teacher_dataset.py --workers 16
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

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from generate_aggregated_teacher_dataset import _load_jsonl, _process_state  # noqa: E402

from traveler_distillation.config import load_dotenv, load_yaml  # noqa: E402
from traveler_distillation.dataset.aggregation import (  # noqa: E402
    AggregatedTeacherTarget,
    AggregationMetadata,
    aggregate_repeats,
)
from traveler_distillation.generators import JointContextGenerator  # noqa: E402
from traveler_distillation.schemas.action import UniversalTravelerAction  # noqa: E402
from traveler_distillation.schemas.dataset import Perturbation  # noqa: E402
from traveler_distillation.schemas.state import UniversalTravelerState  # noqa: E402
from traveler_distillation.teacher import (  # noqa: E402
    DeepSeekTeacherClient,
    MockTeacherClient,
    TeacherResponseParser,
    TeacherResponseValidator,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_joint_states(existing: list[dict], combos: list[dict], k7: dict, gen_cfg: dict) -> list[dict]:
    """Derive joint states from every S3 baseline, with per-state effective K.

    Each returned dict carries ``teacher_repeats`` (effective K to sample) and
    ``base_repeats`` (the combination's base K). A state is a "K=7 subset"
    state when ``teacher_repeats > base_repeats``; M1 re-aggregates those from
    the first ``base_repeats`` repeats.
    """
    jgen = JointContextGenerator(gen_cfg)
    baselines = sorted(
        (r for r in existing if (r.get("perturbation") or {}).get("axis") == "baseline"),
        key=lambda r: r["sample_id"],
    )
    if not baselines:
        raise ValueError("no baselines found in the existing dataset")

    k7_combo = k7.get("combination_id")
    k7_n = int(k7.get("n_states", 0))

    states: list[dict] = []
    seq = 0
    k7_count = 0
    for base in baselines:
        base_state = UniversalTravelerState.model_validate(base["state"])
        for combo in combos:
            cid = combo["combination_id"]
            base_k = int(combo.get("teacher_repeats", 3))
            for js in jgen.generate(base_state, combo):
                effective_k = base_k
                # K=7 causal subset: deterministic slice (level_index==1) of the
                # named low-SNR combo, first ``k7_n`` states in baseline order.
                if cid == k7_combo and js.level_index == 1:
                    if k7_count < k7_n:
                        effective_k = 7
                    k7_count += 1
                seq += 1
                states.append(
                    {
                        "sample_id": f"S5J{seq:06d}",
                        "cf_group_id": f"JOINT_{cid}_{js.level_index}",
                        "baseline_sample_id": base["sample_id"],
                        "split_group_id": base.get("split_group_id"),
                        "persona_group_id": base.get("persona_group_id"),
                        "perturbation": Perturbation(
                            axis="joint",
                            level=0.0,
                            joint_axes=[{"axis": a, "level": l} for a, l in js.axes],
                        ),
                        "state": js.state,
                        "combination_id": cid,
                        "seen_in_training": bool(combo.get("seen_in_training", True)),
                        "joint_level_id": f"{cid}:{js.level_index}",
                        "teacher_repeats": effective_k,
                        "base_repeats": base_k,
                    }
                )
    return states


def _rebuild_target(
    state_dict: dict, records: list, k: int, model: str, prompt_version: str
) -> AggregatedTeacherTarget:
    """Rebuild an aggregated target from the first ``k`` repeat RECORDS (raw dicts).

    ``records`` are raw repeat-record dicts (each with ``action`` and
    ``completion_id``), ordered by ``repeat_index``. The first ``k`` are parsed
    and averaged into a base-K target (M1 view of a K=7 subset state).
    """
    records = sorted(records, key=lambda r: r.get("repeat_index", 0))[:k]
    actions = [UniversalTravelerAction.model_validate(r["action"]) for r in records]
    agg = aggregate_repeats(actions)
    return AggregatedTeacherTarget(
        sample_id=state_dict["sample_id"],
        counterfactual_group_id=state_dict.get("cf_group_id"),
        persona_group_id=state_dict.get("persona_group_id"),
        baseline_sample_id=state_dict.get("baseline_sample_id"),
        split_group_id=state_dict.get("split_group_id"),
        perturbation=state_dict["perturbation"],
        state=state_dict["state"],
        teacher_aggregate=agg,
        aggregation_metadata=AggregationMetadata(
            k=k,
            aggregation_method="mean_probability",
            prompt_version=prompt_version,
            model=model,
            source_completion_ids=[r.get("completion_id") for r in records if r.get("completion_id")],
        ),
        status="complete",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--output", default="data/student_s5_joint")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--combos", default=None, help="comma-separated combination ids to generate")
    ap.add_argument("--max-baselines", type=int, default=None, help="limit baselines (pilot)")
    ap.add_argument("--override-k", type=int, default=None, help="override ALL K (pilot only)")
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="teacher max_tokens (joint contexts trigger longer reasoning than single-axis)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    gen_cfg = load_yaml(args.config)
    j_cfg = load_yaml(args.joint_config)
    t_section = load_yaml(args.teacher_config).get("teacher", {})

    combos = list(j_cfg.get("joint_combinations", []))
    if args.combos:
        wanted = {c.strip() for c in args.combos.split(",") if c.strip()}
        combos = [c for c in combos if c["combination_id"] in wanted]
    k7 = j_cfg.get("k7_subset", {})

    src = Path(args.dataset)
    if not src.exists():
        print(f"[FAIL] dataset not found: {src}")
        return 1
    existing = _load_jsonl(src)

    if args.max_baselines is not None:
        bset = sorted(
            {r["sample_id"] for r in existing if (r.get("perturbation") or {}).get("axis") == "baseline"}
        )[: args.max_baselines]
        existing = [r for r in existing if r["sample_id"] in bset]

    states = build_joint_states(existing, combos, k7, gen_cfg)
    if args.override_k is not None:
        for st in states:
            st["teacher_repeats"] = args.override_k
            st["base_repeats"] = args.override_k

    n_calls = sum(st["teacher_repeats"] for st in states)
    per_combo = defaultdict(lambda: {"states": 0, "calls": 0})
    for st in states:
        per_combo[st["combination_id"]]["states"] += 1
        per_combo[st["combination_id"]]["calls"] += st["teacher_repeats"]
    print(f"joint states={len(states)}  total teacher calls={n_calls}")
    for cid, v in sorted(per_combo.items()):
        print(f"  {cid}: states={v['states']} calls={v['calls']}")

    if args.dry_run:
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
        model = "mock_teacher"
    else:
        teacher = DeepSeekTeacherClient(
            model=t_section.get("model"),
            temperature=t_section.get("temperature", 0.2),
            max_retries=t_section.get("max_retries", 2),
            timeout_seconds=t_section.get("timeout_seconds", 120),
            max_tokens=args.max_tokens,
        )
        model = getattr(teacher, "model", "unknown")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset_path = out / "aggregated_teacher_dataset.jsonl"       # full-K (M2 view)
    dataset_k5_path = out / "aggregated_teacher_dataset_k5.jsonl"  # base-K (M1 view)
    repeats_path = out / "repeat_records.jsonl"
    incomplete_path = out / "incomplete_samples.jsonl"

    done_ids = set()
    if args.resume:
        for r in _load_jsonl(dataset_path):
            done_ids.add(r["sample_id"])
    existing_repeats: dict[str, list] = defaultdict(list)
    for r in _load_jsonl(repeats_path):
        existing_repeats[r["sample_id"]].append(r)

    pending = [st for st in states if st["sample_id"] not in done_ids]
    print(f"resume: {len(states) - len(pending)} already done, {len(pending)} pending")

    stats = {"attempted": 0, "valid": 0, "failed": 0, "incomplete": 0, "aggregated": 0}
    lock = threading.Lock()

    def _submit(st: dict):
        return _process_state(
            st, teacher, parser, validator, st["teacher_repeats"], args.mock,
            [r for r in existing_repeats[st["sample_id"]] if r.get("action") is not None],
            t_section, False, clip_departure=True,
        )

    def _write(st: dict, res: dict):
        with lock:
            for rec in res.get("new_records", []):
                repeats_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            repeats_f.flush()
            if res["status"] == "incomplete":
                stats["incomplete"] += 1
                incomplete_f.write(json.dumps({
                    "sample_id": res["sid"], "valid_repeats": res["n_valid"],
                    "required_repeats": st["teacher_repeats"],
                    "status": "incomplete", "timestamp": _now(),
                }, ensure_ascii=False) + "\n")
                incomplete_f.flush()
                print(f"  [{res['sid']}] INCOMPLETE ({res['n_valid']}/{st['teacher_repeats']})", flush=True)
                return
            # full-K aggregate (M2 view)
            dataset_f.write(json.dumps(json.loads(res["target"].model_dump_json()), ensure_ascii=False) + "\n")
            dataset_f.flush()
            # base-K aggregate (M1 view)
            if st["teacher_repeats"] > st["base_repeats"]:
                all_records = [
                    r for r in (existing_repeats[st["sample_id"]] + res.get("new_records", []))
                    if r.get("action") is not None
                ]
                m1_target = _rebuild_target(
                    st, all_records, st["base_repeats"], model,
                    t_section.get("prompt_version", "teacher_v0.1"),
                )
                dataset_k5_f.write(json.dumps(json.loads(m1_target.model_dump_json()), ensure_ascii=False) + "\n")
            else:
                dataset_k5_f.write(json.dumps(json.loads(res["target"].model_dump_json()), ensure_ascii=False) + "\n")
            dataset_k5_f.flush()
            stats["aggregated"] += 1
            print(f"  [{res['sid']}] aggregated K={st['teacher_repeats']}: {res.get('selected_mode')} "
                  f"{res.get('mode_probabilities')} shift={res.get('departure_shift')}", flush=True)

    with dataset_path.open("a", encoding="utf-8") as dataset_f, \
            dataset_k5_path.open("a", encoding="utf-8") as dataset_k5_f, \
            repeats_path.open("a", encoding="utf-8") as repeats_f, \
            incomplete_path.open("a", encoding="utf-8") as incomplete_f:

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_submit, st): st for st in pending}
                for fut in as_completed(futures):
                    st = futures[fut]
                    res = fut.result()
                    for kk in ("attempted", "valid", "failed"):
                        stats[kk] += res["local"][kk]
                    _write(st, res)
        else:
            for st in pending:
                res = _submit(st)
                for kk in ("attempted", "valid", "failed"):
                    stats[kk] += res["local"][kk]
                _write(st, res)

    manifest = {
        "dataset_version": "student_s5_joint",
        "source_dataset": str(src),
        "joint_config": str(args.joint_config),
        "combos": [c["combination_id"] for c in combos],
        "k7_subset": k7,
        "n_joint_states": len(states),
        "n_teacher_calls": n_calls,
        "endpoint": os.environ.get("DEEPSEEK_BASE_URL", "unknown"),
        "model": model,
        "departure_clip": True,
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
