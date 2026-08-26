#!/usr/bin/env python
"""S8 — teacher labeling of the Singapore real-supply accessibility dataset.

Fetches K teacher repeats per S8 state (K=3 base; K=5 for class-boundary
samples per S8 §13) using the S8 accessibility-aware prompt, aggregates them
into AggregatedTeacherTarget records, and writes:

    data/singapore_accessibility/states_with_teacher.jsonl
    data/singapore_accessibility/repeat_records.jsonl
    data/singapore_accessibility/generation_manifest.json
    data/singapore_accessibility/states_with_teacher.progress.jsonl  (resume)

Incremental + resumable: a killed run continues from the progress file.
The official DeepSeek endpoint has no gateway cache, so cache_bypass=False.

Usage:
    python scripts/label_s8_teacher.py [--workers 16] [--limit N] [--dry-run] [--mock] [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from traveler_distillation.config import load_dotenv, load_yaml  # noqa: E402
from traveler_distillation.dataset.aggregation import (  # noqa: E402
    AggregatedTeacherTarget,
    AggregationMetadata,
    aggregate_repeats,
)
from traveler_distillation.schemas.action import UniversalTravelerAction  # noqa: E402
from traveler_distillation.schemas.dataset import Perturbation  # noqa: E402
from traveler_distillation.schemas.state import UniversalTravelerState  # noqa: E402
from traveler_distillation.teacher import (  # noqa: E402
    DeepSeekTeacherClient,
    MockTeacherClient,
    TeacherResponseParser,
    TeacherResponseValidator,
)
from traveler_distillation.teacher.prompts_s8 import (  # noqa: E402
    S8_SYSTEM_PROMPT,
    TEACHER_S8_PROMPT_VERSION,
    build_s8_user_prompt,
)


class S8TeacherClient(DeepSeekTeacherClient):
    """DeepSeek client using the S8 accessibility-aware prompt."""

    def build_prompt(self, state: UniversalTravelerState) -> tuple[str, str]:
        return S8_SYSTEM_PROMPT, build_s8_user_prompt(state.model_dump_json(indent=2))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _process_state(
    st: dict, k: int, teacher, parser, validator, is_mock: bool,
    existing_valid: list[dict],
) -> dict:
    """K valid repeats for one state -> aggregated target (mirrors S5 flow)."""
    sid = st["sample_id"]
    valid_repeats = list(existing_valid)
    seen = {r.get("completion_id") for r in valid_repeats if r.get("completion_id")}
    new_records: list[dict] = []
    local = {"attempted": 0, "valid": 0, "failed": 0}
    attempts = 0
    state = UniversalTravelerState.model_validate(st["state"])

    while len(valid_repeats) < k and attempts < k + 6:
        attempts += 1
        local["attempted"] += 1
        try:
            if is_mock:
                raw_content = teacher.respond(state)
                evidence = {"completion_id": f"mock-{sid}-{len(valid_repeats)}",
                            "created": None, "provider_cache_status": "mock",
                            "elapsed_seconds": 0.0}
            else:
                raw = teacher.respond_with_evidence(state, cache_bypass=False)
                raw_content = raw["content"]
                evidence = raw["evidence"]
        except Exception as exc:
            local["failed"] += 1
            print(f"  [{sid}] repeat {len(valid_repeats)} FAILED: {exc}", flush=True)
            continue

        cid = evidence.get("completion_id")
        if (not is_mock) and cid in seen:
            local["failed"] += 1
            continue
        try:
            action = parser.parse(raw_content, clip_departure=True)
            res = validator.validate(state, action)
            if not res.valid:
                raise ValueError(res.reason)
        except Exception as exc:
            local["failed"] += 1
            print(f"  [{sid}] parse/validation failed: {exc}", flush=True)
            continue

        if cid:
            seen.add(cid)
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

    result = {"sid": sid, "status": "incomplete" if len(valid_repeats) < k else "complete",
              "n_valid": len(valid_repeats), "local": local,
              "new_records": new_records,
              "existing_records": existing_valid,
              "target": None}
    if result["status"] == "incomplete":
        return result
    actions = [UniversalTravelerAction.model_validate(r["action"]) for r in valid_repeats]
    agg = aggregate_repeats(actions)
    result["target"] = AggregatedTeacherTarget(
        sample_id=sid,
        counterfactual_group_id=st.get("curve_group"),
        persona_group_id=st.get("persona_id"),
        baseline_sample_id=None,
        split_group_id=st.get("curve_group"),
        perturbation=Perturbation(axis="accessibility", level=0.0),
        state=state,
        teacher_aggregate=agg,
        aggregation_metadata=AggregationMetadata(
            k=k,
            aggregation_method="mean_probability",
            prompt_version=TEACHER_S8_PROMPT_VERSION,
            model=getattr(teacher, "model", "unknown"),
            source_completion_ids=[r.get("completion_id") for r in valid_repeats if r.get("completion_id")],
        ),
        status="complete",
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/singapore_accessibility/records.jsonl")
    ap.add_argument("--boundary", default="data/singapore_accessibility/boundary_pairs.json")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--output", default="data/singapore_accessibility")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N states (pilot)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    t_section = load_yaml(args.teacher_config).get("teacher", {})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = _load_jsonl(Path(args.dataset))
    if args.limit:
        records = records[: args.limit]

    boundary_ids: set[str] = set()
    for bp in json.loads(Path(args.boundary).read_text(encoding="utf-8")):
        boundary_ids.update(bp["samples"])

    k_per_state = {r["sample_id"]: (5 if r["sample_id"] in boundary_ids else 3) for r in records}
    k_counts = Counter(k_per_state.values())
    n_calls = sum(k_per_state.values())
    print(f"states={len(records)}  k_distribution={dict(k_counts)}  total calls={n_calls}")
    if args.dry_run:
        print("DRY RUN: no API calls made")
        return 0

    progress_path = out / "states_with_teacher.progress.jsonl"
    progress = {r["sample_id"]: r for r in _load_jsonl(progress_path)} if args.resume else {}
    repeat_path = out / "repeat_records.jsonl"
    target_path = out / "states_with_teacher.jsonl"
    done = {json.loads(l)["sample_id"] for l in target_path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if target_path.exists() else set()

    parser = TeacherResponseParser()
    validator = TeacherResponseValidator(
        probability_tolerance=t_section.get("probability_tolerance", 1e-3),
    )
    if args.mock:
        teacher = MockTeacherClient()
        is_mock = True
    else:
        teacher = S8TeacherClient(max_tokens=args.max_tokens)
        is_mock = False

    pending = [r for r in records if r["sample_id"] not in done]
    print(f"pending states: {len(pending)}")
    t0 = time.time()
    stats = Counter()

    def work(st):
        sid = st["sample_id"]
        existing = [r for r in progress.get(sid, {}).get("repeats", [])]
        res = _process_state(st, k_per_state[sid], teacher, parser, validator,
                             is_mock, existing)
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, st): st for st in pending}
        for fut in as_completed(futs):
            res = fut.result()
            stats["attempted"] += res["local"]["attempted"]
            stats["valid"] += res["local"]["valid"]
            stats["failed"] += res["local"]["failed"]
            if res["status"] == "complete":
                stats["aggregated"] += 1
                _append_jsonl(target_path, res["target"].model_dump(mode="json"))
                for rec in res["existing_records"] + res["new_records"]:
                    _append_jsonl(repeat_path, rec)
            else:
                stats["incomplete"] += 1
                _append_jsonl(progress_path, {
                    "sample_id": res["sid"], "k": k_per_state[res["sid"]],
                    "n_valid": res["n_valid"], "repeats": res["new_records"],
                    "updated": datetime.now(timezone.utc).isoformat(),
                })
            if stats["attempted"] % 50 == 0:
                print(f"  progress: {dict(stats)} elapsed={time.time() - t0:.0f}s", flush=True)

    manifest = {
        "dataset_version": "s8_singapore_accessibility",
        "source": args.dataset,
        "prompt_version": TEACHER_S8_PROMPT_VERSION,
        "k_policy": {"base": 3, "boundary": 5,
                     "boundary_definition": "good<->medium, medium<->poor, poor<->infeasible class transitions within a curve group"},
        "k_distribution": dict(k_counts),
        "n_states": len(records),
        "n_calls_planned": n_calls,
        "stats": dict(stats),
        "endpoint": teacher.base_url if not is_mock else "mock",
        "model": teacher.model,
        "clip_departure": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out / "generation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print(f"\nstats: {dict(stats)} in {time.time() - t0:.0f}s")
    print(f"wrote {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
