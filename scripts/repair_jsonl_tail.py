#!/usr/bin/env python
"""Repair a JSONL file whose trailing line was truncated (e.g. process killed
mid-write). Drops the final line if it does not parse as JSON; leaves valid
files untouched. Used before resuming dataset generation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _say(status: str, p: Path, extra: str = "") -> None:
    # console may be cp1252; never print paths containing non-ASCII chars
    print(f"{status} {p.name} {extra}")


def main() -> int:
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            _say("skip (missing):", p)
            continue
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or not lines[-1].strip():
            _say("ok (empty tail):", p)
            continue
        try:
            json.loads(lines[-1])
            _say("ok (last line valid):", p)
        except json.JSONDecodeError as exc:
            fixed = "\n".join(lines[:-1])
            if fixed:
                fixed += "\n"
            p.write_text(fixed, encoding="utf-8")
            _say("REPAIRED (dropped truncated last line):", p, str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
