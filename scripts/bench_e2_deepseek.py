#!/usr/bin/env python
"""E2 (TRC_AIT_5_EXPERIMENT_PLAN §E2) — DeepSeek direct-decision measurement.

Sequential single-worker API calls over the first N states of the frozen E2
state pool (K=1 per state, cache_bypass=True so every call is a real
inference). Latency/token evidence is appended per call to a JSONL file, so the
run is resumable. Budget gate: total attempts <= --max-attempts (design §1:
100 calls + <=20% retry margin).

Usage:
    python scripts/bench_e2_deepseek.py --pool outputs/e2_efficiency/states_100000.jsonl \
        --n 100 --out outputs/e2_efficiency/deepseek_calls.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traveler_distillation.config import load_dotenv, load_yaml  # noqa: E402
from traveler_distillation.schemas.state import UniversalTravelerState  # noqa: E402
from traveler_distillation.teacher.client import (  # noqa: E402
    DeepSeekTeacherClient,
    TeacherClientError,
)

TEACHER_CONFIG = "configs/teacher_v0_1.yaml"


def wait_for_pool(pool: Path, need: int, timeout_s: float = 3600.0) -> Path:
    """Return the pool file to read once it has >= need states.

    While a multiprocess pool build is running, the merged ``states_N.jsonl``
    does not exist yet; the first range file ``states_range_0_*.jsonl`` already
    contains the head of the sequence (ranges are written in order).
    """
    candidates = [pool]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for cand in candidates:
            if cand.exists():
                count = 0
                with cand.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            count += 1
                            if count >= need:
                                break
                if count >= need:
                    return cand
        candidates = [pool] + sorted(pool.parent.glob("states_range_0_*.jsonl"),
                                      key=lambda p: p.name)
        time.sleep(5.0)
    raise SystemExit(f"pool {pool} did not reach {need} states within {timeout_s}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="outputs/e2_efficiency/states_100000.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="outputs/e2_efficiency/deepseek_calls.jsonl")
    ap.add_argument("--max-attempts", type=int, default=120)
    ap.add_argument("--wait-pool", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    tcfg = load_yaml(str(ROOT / TEACHER_CONFIG))["teacher"]
    pool = Path(args.pool)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.wait_pool:
        pool = wait_for_pool(pool, args.n)

    with pool.open(encoding="utf-8") as fh:
        states = []
        for line in fh:
            if line.strip():
                states.append(UniversalTravelerState.model_validate_json(line))
            if len(states) >= args.n:
                break
    if len(states) < args.n:
        raise SystemExit(f"pool has {len(states)} states, need {args.n}")

    client = DeepSeekTeacherClient(
        model=tcfg.get("model"),
        temperature=tcfg.get("temperature", 0.2),
        max_retries=tcfg.get("max_retries", 2),
        timeout_seconds=tcfg.get("timeout_seconds", 120.0),
        max_tokens=tcfg.get("max_tokens", 8192),
        cache_bypass=True,
    )

    done = 0
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done += 1
    attempts = done
    print(f"[deepseek] endpoint={client.base_url} model={client.model} "
          f"resume_from={done}/{args.n} attempts_so_far={attempts}", flush=True)

    with out.open("a", encoding="utf-8") as fh:
        for i in range(done, args.n):
            state = states[i]
            rec = {
                "state_index": i,
                "persona_id": state.persona.persona_id,
                "trip_id": state.trip.trip_id,
                "cache_bypass": True,
                "ok": False,
            }
            t0 = time.perf_counter()
            try:
                ev = client.respond_with_evidence(state, cache_bypass=True)
                rec["ok"] = True
                rec["content_len"] = len(ev["content"])
                rec.update(ev["evidence"])
            except TeacherClientError as exc:
                rec["failure_type"] = exc.failure_type
                rec["error"] = str(exc)[:300]
            rec["wall_seconds"] = round(time.perf_counter() - t0, 4)
            rec["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            attempts += 1
            el = rec.get("elapsed_seconds")
            print(f"[deepseek] {i + 1}/{args.n} ok={rec['ok']} "
                  f"elapsed={el if el is not None else 'n/a'}s "
                  f"attempts={attempts}/{args.max_attempts}", flush=True)
            if attempts >= args.max_attempts:
                print(f"[deepseek] BUDGET GATE: reached {args.max_attempts} attempts — stopping", flush=True)
                break
            time.sleep(0.5)

    print(f"[deepseek] done: attempts={attempts}, records in {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
