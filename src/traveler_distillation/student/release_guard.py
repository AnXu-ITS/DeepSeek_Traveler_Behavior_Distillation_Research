"""Hard release protection: training outputs must never land inside a frozen release.

Per `S7_W3_BACKUP_FREEZE_INSTRUCTIONS.md` step 15, every S8+ training/eval script
MUST call :func:`assert_not_frozen_output` on its output directory before writing
anything. The guard is a hard assertion: violations raise ``RuntimeError`` and are
never silently ignored.

S8 may only load S7-W3 and save to a separate output/release directory.
"""
from __future__ import annotations

from pathlib import Path

FROZEN_RELEASE_DIRS = frozenset({"s7_w3_generic_core_v1"})
FROZEN_RELEASES: dict[str, str] = {
    "s7_w3_generic_core_v1": "S7-W3 Generic Behavioral Core v1.0 (FROZEN)"
}


def assert_not_frozen_output(path: str | Path) -> Path:
    """Return the resolved path, or raise if it resolves inside a frozen release."""
    p = Path(path).resolve()
    hit = FROZEN_RELEASE_DIRS & set(p.parts)
    if hit:
        name = next(iter(hit))
        raise RuntimeError(
            f"output path {path} resolves inside frozen release '{name}' "
            f"({FROZEN_RELEASES[name]}). S8+ runs must save to separate "
            "output/release directories; the frozen release must never be modified."
        )
    return p
