"""MATSim execution helpers shared by the population/feedback-loop scripts."""
from __future__ import annotations

import subprocess
from pathlib import Path

# ASCII-only classpath (PowerShell->java argv corrupts non-ASCII paths): the
# MATSim release is reached via a Temp junction, and the preloaded-scenario
# launcher class lives in an ASCII Temp dir.
_TMP = "C:/Users/xuan1/AppData/Local/Temp"
MATSIM_CP = f"{_TMP}/matsim_run;{_TMP}/matsim_rel/matsim-2026.0.jar;{_TMP}/matsim_rel/libs/*"


def run_matsim(scenario_dir: Path, timeout: int = 1800) -> tuple[int, str]:
    """Run the preloaded-scenario launcher in scenario_dir; returns (code, log tail)."""
    proc = subprocess.run(
        ["java", "-Xmx2g", "-cp", MATSIM_CP, "RunMatsimPreloaded", "config.xml"],
        cwd=str(scenario_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    log_path = scenario_dir / "output" / "logfileWarningsErrors.log"
    tail = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-6:])
    return proc.returncode, tail


def parse_modestats(path: Path) -> dict:
    """MATSim modestats.csv: iteration;bike;car;pt;walk -> {mode: share}."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split(";")
    values = lines[1].split(";")
    return {h: float(v) for h, v in zip(header[1:], values[1:])}


def parse_traveldist(path: Path) -> dict:
    """MATSim traveldistancestats.csv: avg leg / avg trip distance (meters)."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split(";")
    values = lines[1].split(";")
    out = {}
    for h, v in zip(header[1:], values[1:]):
        key = h.strip().lower().replace(" ", "_").replace(".", "")
        if key.startswith("avg_average_leg"):
            out["avg_leg_distance_m"] = float(v)
        elif key.startswith("avg_average_trip"):
            out["avg_trip_distance_m"] = float(v)
    return out


def parse_linkstats(path: Path) -> dict:
    """MATSim link stats -> congestion summary.

    ``path`` may be MATSim 2026's ``ITERS/it.0/0.linkstats.txt.zst`` (TSV,
    per-hour min/avg/max triplets) or a legacy ``linkstats.csv.gz`` (CSV).

    Returns:
      mean_delay_ratio: capacity-weighted mean of (avg_travel_time -
                        free_travel_time) / free_travel_time over all
                        (link, hour) cells with positive travel time.
      congestion:       mean_delay_ratio / 0.5 clamped to [0, 1] — matches the
                        context semantics where road_congestion c scales travel
                        time by (1 + 0.5 * c).
    """
    import csv
    import gzip

    if not path.exists():
        return {"mean_delay_ratio": None, "congestion": None}

    # --- MATSim 2026 linkstats txt (zstd) ---
    if path.suffix == ".zst" and "linkstats" in path.name:
        try:
            import zstandard
            with open(path, "rb") as f:
                dctx = zstandard.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text = reader.read().decode("utf-8", "replace")
        except Exception:
            return {"mean_delay_ratio": None, "congestion": None}
        lines = text.splitlines()
        if len(lines) < 2:
            return {"mean_delay_ratio": None, "congestion": None}
        header = lines[0].split("\t")
        i_len = header.index("LENGTH") if "LENGTH" in header else None
        i_free = header.index("FREESPEED") if "FREESPEED" in header else None
        i_cap = header.index("CAPACITY") if "CAPACITY" in header else None
        avg_cols = [i for i, h in enumerate(header) if "avg" in h.lower()]

        weighted = 0.0
        total_w = 0.0
        n_cells = 0
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            try:
                length = float(parts[i_len])
                free = float(parts[i_free])
                cap = float(parts[i_cap])
            except (IndexError, ValueError, TypeError):
                continue
            if free <= 0 or length <= 0:
                continue
            free_tt = length / free
            w = max(cap, 0.1)
            for ci in avg_cols:
                try:
                    tt = float(parts[ci])
                except (IndexError, ValueError):
                    continue
                if tt <= 0:
                    continue
                ratio = max(0.0, (tt - free_tt) / free_tt)
                weighted += ratio * w
                total_w += w
                n_cells += 1
        if total_w <= 0:
            return {"mean_delay_ratio": None, "congestion": None}
        mean_ratio = weighted / total_w
        return {
            "mean_delay_ratio": round(mean_ratio, 5),
            "congestion": round(max(0.0, min(1.0, mean_ratio / 0.5)), 5),
            "n_cells": n_cells,
        }

    # --- legacy linkstats.csv.gz ---
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        if header is None:
            return {"mean_delay_ratio": None, "congestion": None}

        def col(*names):
            for i, h in enumerate(header):
                hu = h.strip().upper()
                if any(n.upper() in hu for n in names):
                    return i
            return None

        i_len = col("LENGTH")
        i_free = col("FREESPEED")
        i_tt = col("TRAVELTIME")
        i_cap = col("CAPACITY")

        weighted = 0.0
        total_w = 0.0
        n = 0
        for row in reader:
            try:
                length = float(row[i_len])
                free = float(row[i_free])
                tt = float(row[i_tt])
                cap = float(row[i_cap]) if i_cap is not None else 1.0
            except (IndexError, ValueError):
                continue
            if free <= 0 or length <= 0 or tt <= 0:
                continue
            free_tt = length / free
            ratio = max(0.0, (tt - free_tt) / free_tt)
            weighted += ratio * max(cap, 0.1)
            total_w += max(cap, 0.1)
            n += 1

    if total_w <= 0 or n == 0:
        return {"mean_delay_ratio": None, "congestion": None}

    mean_ratio = weighted / total_w
    congestion = max(0.0, min(1.0, mean_ratio / 0.5))
    return {
        "mean_delay_ratio": round(mean_ratio, 5),
        "congestion": round(congestion, 5),
        "n_links": n,
    }
