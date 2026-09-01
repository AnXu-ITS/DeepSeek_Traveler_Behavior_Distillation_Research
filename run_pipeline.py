#!/usr/bin/env python
"""Convenience shim: python run_pipeline.py --config configs/reference_example.yaml

Delegates to scripts/run_reference_pipeline.py (the canonical entry point).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).resolve().parents[0] / "scripts" / "run_reference_pipeline.py")
with open(sys.argv[0], encoding="utf-8") as f:
    code = compile(f.read(), sys.argv[0], "exec")
exec(code, {"__name__": "__main__", "__file__": sys.argv[0]})
