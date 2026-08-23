"""Configuration loading utilities (YAML + minimal .env support)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader.

    Loads ``KEY=VALUE`` pairs from a file into ``os.environ`` for keys that are
    not already present. Never overwrites existing environment variables.
    """
    if path is None:
        # Walk up from CWD to find a .env
        candidates = [Path.cwd() / ".env"]
        p = Path.cwd()
        for _ in range(4):
            p = p.parent
            candidates.append(p / ".env")
        target = next((c for c in candidates if c.exists()), None)
    else:
        target = Path(path)

    if target is None or not target.exists():
        return

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict (empty dict if missing keys / empty file)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
