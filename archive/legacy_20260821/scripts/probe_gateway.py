#!/usr/bin/env python
"""Probe the DeepSeek gateway until it accepts chat completions again.

Minimal request (max_tokens=1). Exits 0 on first HTTP 200, 1 after max_attempts
failures. Used to detect recovery after a gateway outage so the long-running
dataset generation can be restarted as soon as the API is healthy again.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from traveler_distillation.config import load_dotenv


def main() -> int:
    load_dotenv()
    base = os.environ.get("DEEPSEEK_BASE_URL", "").rstrip("/")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    if not base or not key:
        print("missing DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY")
        return 1

    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    attempts = int(os.environ.get("PROBE_ATTEMPTS", "15"))
    interval = int(os.environ.get("PROBE_INTERVAL_S", "120"))

    for i in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=30, trust_env=False) as client:
                resp = client.post(url, json=payload, headers=headers)
            status = resp.status_code
            snippet = resp.text[:100].replace("\n", " ")
            print(f"[probe {i}/{attempts}] HTTP {status} {snippet}", flush=True)
            if status == 200:
                print("GATEWAY RECOVERED", flush=True)
                return 0
        except Exception as exc:
            print(f"[probe {i}/{attempts}] {type(exc).__name__}: {exc}", flush=True)
        if i < attempts:
            time.sleep(interval)

    print("gateway still down after all attempts", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
