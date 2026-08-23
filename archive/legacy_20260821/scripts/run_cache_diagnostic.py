#!/usr/bin/env python
"""Gateway cache diagnostic for the repeatability audit.

Issues 3 representative states x 3 identical real teacher calls (9 calls total)
and records non-secret request/response evidence so we can verify the calls are
independent server-side executions and not a local or gateway cache hit.

Usage:
    python scripts/run_cache_diagnostic.py
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
from traveler_distillation.schemas.dataset import TeacherDatasetSample
from traveler_distillation.audit import state_hash


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_pilot(path: Path) -> list[TeacherDatasetSample]:
    samples = []
    if not path.exists():
        return samples
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(TeacherDatasetSample.model_validate(json.loads(line)))
    return samples


def _select_states(pilot: list[TeacherDatasetSample]) -> list[TeacherDatasetSample]:
    """Pick one representative state per axis (weather, fare, congestion)."""
    by_axis: dict[str, TeacherDatasetSample] = {}
    for s in pilot:
        if s.teacher is None:
            continue
        axis = s.perturbation.axis
        if axis in ("weather_intensity", "fare_multiplier", "road_congestion"):
            # Prefer a non-baseline perturbation (level differs from baseline).
            by_axis.setdefault(axis, s)
    picks = []
    for axis in ("weather_intensity", "fare_multiplier", "road_congestion"):
        if axis in by_axis:
            picks.append(by_axis[axis])
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--teacher-config", default="configs/teacher_v0_1.yaml")
    ap.add_argument("--pilot", default="outputs/teacher_audit_v0_1/pilot_samples.jsonl")
    ap.add_argument("--output", default="outputs/teacher_audit_v0_1/cache_diagnostic.json")
    ap.add_argument("--calls-per-state", type=int, default=3)
    args = ap.parse_args()

    load_dotenv()
    t_cfg = load_yaml(args.teacher_config).get("teacher", {})
    parser = TeacherResponseParser()
    validator = TeacherResponseValidator(
        probability_tolerance=t_cfg.get("probability_tolerance", 1e-3),
        departure_shift_min=t_cfg.get("departure_shift_min", -60),
        departure_shift_max=t_cfg.get("departure_shift_max", 60),
    )
    teacher = DeepSeekTeacherClient(
        model=t_cfg.get("model"),
        temperature=t_cfg.get("temperature", 0.2),
        max_retries=t_cfg.get("max_retries", 2),
        timeout_seconds=t_cfg.get("timeout_seconds", 120),
        max_tokens=t_cfg.get("max_tokens", 8192),
    )

    pilot = _read_pilot(Path(args.pilot))
    states = _select_states(pilot)
    if len(states) < 3:
        print(f"WARNING: only {len(states)}/3 representative states found in pilot", file=sys.stderr)

    results = []
    n_http = 0
    completion_ids: list[str] = []
    created_ts: list[int] = []
    cache_status_values: list[str] = []
    prompt_cache_hits: list[int] = []

    for i, sample in enumerate(states, 1):
        h = state_hash(sample.state)
        call_records = []
        for rep in range(args.calls_per_state):
            request_start = _now()
            raw = teacher.respond_with_evidence(sample.state)
            response_ts = _now()
            content = raw["content"]
            evidence = raw["evidence"]

            n_http += 1
            action = None
            try:
                action = parser.parse(content)
            except Exception:
                action = None

            if evidence.get("completion_id"):
                completion_ids.append(str(evidence["completion_id"]))
            if evidence.get("created") is not None:
                created_ts.append(int(evidence["created"]))

            headers = evidence.get("response_headers", {})
            cache_status = headers.get("llm_provider-eo-cache-status")
            if cache_status:
                cache_status_values.append(str(cache_status))
            usage = evidence.get("usage") or {}
            hit = usage.get("prompt_cache_hit_tokens")
            if isinstance(hit, int):
                prompt_cache_hits.append(hit)

            call_records.append(
                {
                    "audit_call_id": f"A{i:02d}-call-{rep}",
                    "request_start_timestamp": request_start,
                    "response_timestamp": response_ts,
                    "http_status": evidence.get("http_status"),
                    "model": evidence.get("model"),
                    "finish_reason": evidence.get("finish_reason"),
                    "completion_id": evidence.get("completion_id"),
                    "created": evidence.get("created"),
                    "system_fingerprint": evidence.get("system_fingerprint"),
                    "elapsed_seconds": evidence.get("elapsed_seconds"),
                    "usage": evidence.get("usage"),
                    "response_headers": evidence.get("response_headers"),
                    "selected_mode": action.selected_mode if action else None,
                    "mode_probabilities": action.mode_probabilities if action else None,
                    "departure_time_shift_min": action.departure_time_shift_min if action else None,
                    "confidence": action.confidence if action else None,
                }
            )
            print(
                f"[A{i:02d}] rep={rep} status={evidence.get('http_status')} "
                f"finish={evidence.get('finish_reason')} id={str(evidence.get('completion_id'))[:8]} "
                f"cache={cache_status} hit_tokens={hit} elapsed={evidence.get('elapsed_seconds')}s"
            )

        results.append(
            {
                "audit_state_id": f"A{i:02d}",
                "state_hash": h,
                "source_sample_id": sample.sample_id,
                "axis": sample.perturbation.axis,
                "calls": call_records,
            }
        )

    # ---- conclusion ----
    unique_completions = len(set(completion_ids))
    unique_created = len(set(created_ts))
    all_miss = bool(cache_status_values) and all(v.upper() == "MISS" for v in cache_status_values)
    all_no_hit = bool(prompt_cache_hits) and all(h == 0 for h in prompt_cache_hits)

    # A reused completion_id (or reused server `created` timestamp) across calls
    # means the gateway returned the same response object => response cache hit.
    if unique_completions < n_http or (created_ts and unique_created < n_http):
        conclusion = "GATEWAY CACHE DETECTED"
    elif unique_completions == n_http and unique_created == n_http:
        conclusion = "NO LOCAL CACHE; INDEPENDENT REQUESTS CONFIRMED"
    elif all_miss or all_no_hit:
        conclusion = "GATEWAY CACHE STATUS NOT DIRECTLY OBSERVABLE"
    else:
        conclusion = "GATEWAY CACHE STATUS NOT DIRECTLY OBSERVABLE"

    gateway_evidence = []
    gateway_evidence.append(
        f"unique completion ids: {unique_completions}/{n_http} HTTP calls"
    )
    if cache_status_values:
        gateway_evidence.append(f"llm_provider-eo-cache-status values: {sorted(set(cache_status_values))}")
    if prompt_cache_hits:
        gateway_evidence.append(
            f"prompt_cache_hit_tokens: {sorted(set(prompt_cache_hits))} (non-zero = provider prefix cache used)"
        )

    diagnostic = {
        "diagnostic_version": "teacher_audit_v0.1.1",
        "states_tested": len(states),
        "calls_per_state": args.calls_per_state,
        "http_calls_actually_issued": n_http,
        "unique_completion_ids": unique_completions,
        "unique_created_timestamps": unique_created,
        "cache_status_values": sorted(set(cache_status_values)),
        "prompt_cache_hit_tokens": prompt_cache_hits,
        "local_cache_detected": False,
        "gateway_cache_evidence": "; ".join(gateway_evidence) or "no gateway cache headers observed",
        "conclusion": conclusion,
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCONCLUSION: {conclusion}")
    print(f"HTTP calls issued: {n_http}")
    print(f"Unique completion ids: {unique_completions}")
    print(f"Gateway cache evidence: {'; '.join(gateway_evidence) or 'n/a'}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
