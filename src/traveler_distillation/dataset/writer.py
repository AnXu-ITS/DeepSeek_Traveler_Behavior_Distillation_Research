"""Streaming JSONL writer with overwrite protection."""
from __future__ import annotations

import json
from pathlib import Path


class JSONLWriter:
    """Append JSON objects as JSON lines.

    By default refuses to overwrite an existing file. ``overwrite=True``
    truncates; ``resume=True`` appends to an existing file.
    """

    def __init__(self, path: str | Path, overwrite: bool = False, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()

        if resume:
            mode = "a"
        elif overwrite:
            mode = "w"
        elif exists:
            raise FileExistsError(
                f"output already exists: {self.path}. Use --overwrite to replace "
                "or --resume to continue."
            )
        else:
            mode = "x"

        self._f = self.path.open(mode, encoding="utf-8")

    def append(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_existing_ids(path: str | Path) -> set[str]:
    """Read already-written sample_ids from a JSONL file (for resume)."""
    p = Path(path)
    ids: set[str] = set()
    if not p.exists():
        return ids
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = obj.get("sample_id")
        if sid:
            ids.add(sid)
    return ids
