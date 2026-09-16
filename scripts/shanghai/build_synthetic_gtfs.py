#!/usr/bin/env python
"""Build a synthetic Shanghai Metro GTFS from real Amap stations + assumed headways.

Stations and line assignment are REAL (Amap POI). The schedule is SYNTHETIC:
typical Shanghai Metro headways (peak 3 min / off-peak 6 min / late 10 min),
first train 05:30, last 22:30, per-segment travel time from straight-line
distance at ~35 km/h (clamped 90-300 s). Station ordering along each line is a
geometric approximation (projection onto the line's principal axis) — a first
feasibility pass, clearly labeled; replace with the real Transitland feed
(`download_shanghai_gtfs.py`) for paper-grade supply.

Outputs GTFS text files and zips them to
``data/shanghai/gtfs/raw/shanghai-gtfs-synthetic.zip`` (consumed unchanged by
``build_shanghai_transit.py``).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WB / "src"))
from traveler_distillation.singapore.projection import utm51n  # noqa: E402

STATIONS_JSON = _WB / "data/shanghai/transit/amap_stations.json"
OUT_DIR = _WB / "data/shanghai/gtfs/synthetic"
ZIP_OUT = _WB / "data/shanghai/gtfs/raw/shanghai-gtfs-synthetic.zip"

FIRST_DEP = 5 * 3600 + 30 * 60   # 05:30
LAST_DEP = 22 * 3600 + 30 * 60   # 22:30
SPEED_MS = 35.0 / 3.6            # ~35 km/h metro
MIN_SEG, MAX_SEG = 90, 300       # seconds per inter-station segment
DWELL = 30                       # seconds dwell at each stop


def _hms(sec: int) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def headway(sec_of_day: int) -> int:
    """Peak 3 min (7-9:30, 17-19:30), late 10 min (>=21:00), else 6 min."""
    h = sec_of_day / 3600.0
    if (7.0 <= h <= 9.5) or (17.0 <= h <= 19.5):
        return 3 * 60
    if h >= 21.0:
        return 10 * 60
    return 6 * 60


def order_line(stations: list[dict]) -> list[dict]:
    """Geometric order: project onto the axis between the two farthest stations."""
    if len(stations) <= 2:
        return stations
    pts = {s["name"]: utm51n(s["lat"], s["lon"]) for s in stations}
    names = [s["name"] for s in stations]
    # farthest pair
    a, b, dmax = None, None, -1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (x1, y1), (x2, y2) = pts[names[i]], pts[names[j]]
            d = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if d > dmax:
                dmax, a, b = d, names[i], names[j]
    ax, ay = pts[a]
    dx, dy = pts[b][0] - ax, pts[b][1] - ay
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    ordered = sorted(stations, key=lambda s: (pts[s["name"]][0] - ax) * dx + (pts[s["name"]][1] - ay) * dy)
    return ordered


def main() -> int:
    stations = json.loads(STATIONS_JSON.read_text(encoding="utf-8"))
    by_id = {}
    for i, (amap_id, s) in enumerate(stations.items()):
        by_id[s["name"]] = {"stop_id": f"s{i}", "name": s["name"], "lat": s["lat"],
                            "lon": s["lon"], "lines": s["lines"]}

    # group physical stations by line
    lines: dict[str, list[dict]] = defaultdict(list)
    for s in by_id.values():
        for ln in s["lines"]:
            if ln.isdigit():
                lines[ln].append(s)
    lines = {ln: v for ln, v in sorted(lines.items(), key=lambda kv: int(kv[0])) if len(v) >= 2}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ---- stops.txt ----
    stops_rows = [{"stop_id": s["stop_id"], "stop_name": s["name"], "stop_lat": round(s["lat"], 6),
                   "stop_lon": round(s["lon"], 6), "location_type": 0} for s in by_id.values()]
    _write_csv(OUT_DIR / "stops.txt",
               ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"], stops_rows)

    # ---- routes.txt / trips.txt / stop_times.txt ----
    routes_rows, trips_rows, st_rows = [], [], []
    trip_no = 0
    for ln, sts in lines.items():
        ordered = order_line(sts)
        route_id = f"metro{ln}"
        routes_rows.append({"route_id": route_id, "agency_id": "shmetro", "route_short_name": f"{ln}号线",
                            "route_long_name": f"上海地铁{ln}号线", "route_type": 1})
        for direction, seq in ((0, ordered), (1, list(reversed(ordered)))):
            # per-segment travel times (seconds)
            seg = []
            for i in range(1, len(seq)):
                x1, y1 = utm51n(seq[i - 1]["lat"], seq[i - 1]["lon"])
                x2, y2 = utm51n(seq[i]["lat"], seq[i]["lon"])
                t = math.hypot(x2 - x1, y2 - y1) / SPEED_MS
                seg.append(min(MAX_SEG, max(MIN_SEG, t)))
            headsign = seq[-1]["name"]
            t = FIRST_DEP
            while t <= LAST_DEP:
                trip_id = f"{route_id}_{direction}_{trip_no}"
                trip_no += 1
                trips_rows.append({"route_id": route_id, "service_id": "WD", "trip_id": trip_id,
                                   "trip_headsign": headsign, "direction_id": direction})
                cur = t  # arrival at first station
                for i, s in enumerate(seq):
                    arr = cur
                    dep_i = arr + DWELL
                    st_rows.append({"trip_id": trip_id, "arrival_time": _hms(arr),
                                    "departure_time": _hms(dep_i), "stop_id": s["stop_id"],
                                    "stop_sequence": i + 1})
                    if i < len(seq) - 1:
                        cur = dep_i + seg[i]  # travel time to next station
                # advance by headway
                t += headway(t)

    _write_csv(OUT_DIR / "routes.txt",
               ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"], routes_rows)
    _write_csv(OUT_DIR / "trips.txt",
               ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id"], trips_rows)
    _write_csv(OUT_DIR / "stop_times.txt",
               ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], st_rows)

    # ---- calendar.txt / calendar_dates.txt (daily service, reference day 2026-09-09) ----
    _write_csv(OUT_DIR / "calendar.txt",
               ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "start_date", "end_date"],
               [{"service_id": "WD", "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
                 "friday": 1, "saturday": 1, "sunday": 1, "start_date": "20260901", "end_date": "20260930"}])
    _write_csv(OUT_DIR / "calendar_dates.txt", ["service_id", "date", "exception_type"], [])

    # ---- zip ----
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("stops.txt", "routes.txt", "trips.txt", "stop_times.txt", "calendar.txt", "calendar_dates.txt"):
            zf.write(OUT_DIR / name, name)

    print(f"lines: {len(lines)} (2+ stations each); trips: {trip_no}; stops: {len(by_id)}")
    for ln in sorted(lines, key=int):
        names = [s["name"] for s in order_line(lines[ln])]
        print(f"  线{ln:>2}: {' -> '.join(names)}")
    print(f"wrote {ZIP_OUT}")
    return 0


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())
