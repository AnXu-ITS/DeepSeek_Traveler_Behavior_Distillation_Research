#!/usr/bin/env python
"""One-shot measurement: real per-call token usage, latency, and any cost fields.

Writes ``outputs/api_cost_probe.json`` so the numbers survive for the estimate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from traveler_distillation.config import load_dotenv, load_yaml
from traveler_distillation.teacher import DeepSeekTeacherClient
from traveler_distillation.generators import PersonaGenerator, TripGenerator, BaselineStateGenerator


def main() -> int:
    load_dotenv()
    cfg = load_yaml("configs/generation_v0_1.yaml")
    tc = load_yaml("configs/teacher_v0_1.yaml")["teacher"]
    persona = PersonaGenerator(seed=42, config=cfg).generate(1)[0]
    trip = TripGenerator(seed=42, config=cfg).generate(1)[0]
    state = BaselineStateGenerator(cfg).generate(persona, trip)

    client = DeepSeekTeacherClient(
        model=tc["model"],
        temperature=tc.get("temperature", 0.2),
        max_tokens=tc.get("max_tokens", 8192),
    )
    system_prompt, user_prompt = client.build_prompt(state)
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
        "cache": {"no-cache": True},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}",
    }
    t0 = time.time()
    resp = httpx.post(
        client.base_url + "/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    )
    elapsed = time.time() - t0

    out = {
        "http_status": resp.status_code,
        "elapsed_seconds": round(elapsed, 2),
        "content_type": resp.headers.get("content-type"),
        "body_head": resp.text[:600],
        "usage": None,
        "cost_fields_in_body": {},
        "cost_headers": {},
    }
    try:
        data = resp.json()
        out["usage"] = data.get("usage")
        out["cost_fields_in_body"] = {
            k: v for k, v in data.items() if any(w in k.lower() for w in ("cost", "price", "bill"))
        }
        out["model"] = data.get("model")
    except Exception:
        pass
    out["cost_headers"] = {
        k: v for k, v in resp.headers.items() if any(w in k.lower() for w in ("cost", "price", "bill"))
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    Path("outputs/api_cost_probe.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
