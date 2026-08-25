#!/usr/bin/env python
"""S6 causal-audit Teacher run: fill C/D teacher targets (K=5).

Reads the audit states (data/causal_audit/states.jsonl), queries the teacher K
times for every state whose ``teacher_aggregate`` is null (the broken-path and
mediator-only states; A/B already reuse S3 K=3 targets), aggregates the K
repeats, and writes the filled records to ``states_with_teacher.jsonl``.

Usage:
    python scripts/run_teacher_causal_audit.py --dry-run
    python scripts/run_teacher_causal_audit.py --workers 16 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from traveler_distillation.config import load_dotenv, load_yaml
from traveler_distillation.dataset.aggregation import (
    AggregatedTeacherTarget,
    AggregationMetadata,
    aggregate_repeats,
)
from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.teacher import (
    DeepSeekTeacherClient,
    TeacherResponseParser,
    TeacherResponseValidator,
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="data/causal_audit/states.jsonl")
    ap.add_argument("--output", default="data/causal_audit/states_with_teacher.jsonl")
    ap.add_argument("--audit-config", default="configs/causal_audit.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--max-states", type=int, default=None, help="limit new states (pilot)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    t_section = load_yaml(args.teacher_config).get("teacher", {})
    k = int(load_yaml(args.audit_config).get("teacher_repeats", 5))

    records = _load_jsonl(Path(args.states))
    pending = [r for r in records if r.get("teacher_aggregate") is None]
    if args.max_states is not None:
        pending = pending[: args.max_states]
    print(f"states={len(records)}  new (C/D, need teacher)={len(pending)}  x K={k} = {len(pending) * k} calls")

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
        from traveler_distillation.teacher import MockTeacherClient
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

    # resume: reuse incrementally-written progress (survives interruption)
    progress_path = Path(args.output).with_suffix(".progress.jsonl")
    done = {}
    for r in _load_jsonl(progress_path):
        done[r["sample_id"]] = r["teacher_aggregate"]
    if not done:
        for r in _load_jsonl(Path(args.output)):
            done[r["sample_id"]] = r["teacher_aggregate"]

    results: dict[str, dict] = {}
    lock = threading.Lock()
    progress_f = progress_path.open("a", encoding="utf-8")
    stats = {"attempted": 0, "valid": 0, "failed": 0, "incomplete": 0}

    def _fetch(r: dict):
        state = UniversalTravelerState.model_validate(r["state"])
        actions = []
        local = {"attempted": 0, "valid": 0, "failed": 0}
        attempts = 0
        while len(actions) < k and attempts < k + 6:
            attempts += 1
            local["attempted"] += 1
            try:
                if args.mock:
                    content = teacher.respond(state)
                else:
                    raw = teacher.respond_with_evidence(state, cache_bypass=False)
                    content = raw["content"]
            except Exception as exc:
                local["failed"] += 1
                print(f"  [{r['sample_id']}] FAILED: {exc}", flush=True)
                continue
            try:
                action = parser.parse(content, clip_departure=True)
                res = validator.validate(state, action)
                if not res.valid:
                    raise ValueError(res.reason)
            except Exception as exc:
                local["failed"] += 1
                continue
            actions.append(action)
            local["valid"] += 1
        if len(actions) < k:
            return r["sample_id"], None, local, "incomplete"
        agg = aggregate_repeats(actions)
        target = {
            "aggregate": {
                "mode_probabilities": agg.mode_probabilities,
                "selected_mode": agg.selected_mode,
                "departure_time_shift_min": agg.departure_time_shift_min,
                "confidence_mean": agg.confidence_mean,
                "confidence_std": agg.confidence_std,
            },
            "k": k,
            "model": model,
        }
        return r["sample_id"], target, local, "complete"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch, r): r for r in pending if r["sample_id"] not in done}
        for fut in as_completed(futures):
            sid, target, local, status = fut.result()
            with lock:
                for kk in ("attempted", "valid", "failed"):
                    stats[kk] += local[kk]
                if status == "complete":
                    results[sid] = target
                    # incremental progress write (crash-safe resume)
                    progress_f.write(json.dumps(
                        {"sample_id": sid, "teacher_aggregate": target["aggregate"], "teacher_k": k},
                        ensure_ascii=False) + "\n")
                    progress_f.flush()
                    print(f"  [{sid}] done ({len(results)}/{len(pending)})", flush=True)
                else:
                    stats["incomplete"] += 1
    progress_f.close()

    # merge
    for r in records:
        if r["sample_id"] in results:
            r["teacher_aggregate"] = {
                "mode_probabilities": results[r["sample_id"]]["aggregate"]["mode_probabilities"],
                "selected_mode": results[r["sample_id"]]["aggregate"]["selected_mode"],
                "departure_time_shift_min": results[r["sample_id"]]["aggregate"]["departure_time_shift_min"],
                "confidence_mean": results[r["sample_id"]]["aggregate"]["confidence_mean"],
                "confidence_std": results[r["sample_id"]]["aggregate"]["confidence_std"],
            }
            r["teacher_k"] = k
        elif r["sample_id"] in done:
            r["teacher_aggregate"] = done[r["sample_id"]]
            r["teacher_k"] = k

    out = Path(args.output)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
