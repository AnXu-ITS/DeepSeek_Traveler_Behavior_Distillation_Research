"""Singapore GTFS -> region-filtered stops/trips (Phase A prep step).

Streams the (large) GTFS tables straight from the ZIP:

- calendar: pick the weekday service pattern with the most trips;
- trips: keep bus (route_type=3) and MRT (route_type=1) trips of that pattern;
- stop_times: keep only sequences of the kept trips (8.17M rows streamed once);
- regional filter: ALL stops of a kept trip must lie inside the study bbox
  (with margin) — trips crossing the boundary are excluded in Phase A;
- time window: first departure within [05:00, 23:00].

Outputs (data/singapore/transit/):
  prep_stops.jsonl   {stop_id, name, lat, lon, x, y}
  prep_trips.jsonl   {trip_id, route_id, short_name, mode, stop_sequence:
                      [{stop_id, arr_sec, dep_sec}], first_dep_sec}
  prep_meta.json     counts + provenance
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from .projection import utm48n

BBOX = {"lat_min": 1.330, "lat_max": 1.400, "lon_min": 103.900, "lon_max": 104.015}
MARGIN = 0.004  # ~450 m
TIME_WINDOW = (5 * 3600, 23 * 3600)
SERVICE_DAY = "weekday"  # picked from calendar (Monday-Friday)


def _hhmmss_to_sec(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def _stream_rows(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig")
        yield from csv.DictReader(text)


def prep_gtfs(
    zip_path: str | Path,
    out_dir: str | Path,
    bbox: dict | None = None,
    margin: float = MARGIN,
    time_window: tuple[int, int] = TIME_WINDOW,
) -> dict:
    bbox = bbox or BBOX
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(Path(zip_path)) as zf:
        # calendar: pick a weekday service id with the most trips
        service_trips: dict[str, int] = {}
        for row in _stream_rows(zf, "trips.txt"):
            service_trips[row["service_id"]] = service_trips.get(row["service_id"], 0) + 1
        calendar_rows = list(_stream_rows(zf, "calendar.txt"))
        weekday_ids = [
            r["service_id"] for r in calendar_rows
            if int(r.get("monday") or 0) == 1 and int(r.get("tuesday") or 0) == 1
        ]
        service_id = max(weekday_ids, key=lambda sid: service_trips.get(sid, 0)) if weekday_ids else \
            max(service_trips, key=service_trips.get)
        print(f"service_id={service_id} (trips={service_trips.get(service_id, 0)})")

        # routes: id -> (mode, short_name)
        route_mode = {}
        route_name = {}
        for row in _stream_rows(zf, "routes.txt"):
            rt = row.get("route_type")
            if rt in ("1", "3"):
                route_mode[row["route_id"]] = "rail" if rt == "1" else "bus"
                route_name[row["route_id"]] = row.get("route_short_name") or row.get("route_long_name") or row["route_id"]

        # trips of the chosen service
        trip_info = {}
        for row in _stream_rows(zf, "trips.txt"):
            if row["service_id"] == service_id and row["route_id"] in route_mode:
                trip_info[row["trip_id"]] = {
                    "route_id": row["route_id"],
                    "mode": route_mode[row["route_id"]],
                    "short_name": route_name[row["route_id"]],
                }

        # stop_times: stream once, keep only kept trips
        seq: dict[str, list[tuple[int, int, int]]] = {t: [] for t in trip_info}  # trip -> [(seq, arr, dep)]
        for row in _stream_rows(zf, "stop_times.txt"):
            tid = row["trip_id"]
            if tid in seq:
                seq[tid].append(
                    (int(row["stop_sequence"]), _hhmmss_to_sec(row["arrival_time"]),
                     _hhmmss_to_sec(row["departure_time"]), row["stop_id"])
                )

        # stops table
        stops = {r["stop_id"]: (float(r["stop_lat"]), float(r["stop_lon"]), r["stop_name"])
                 for r in _stream_rows(zf, "stops.txt")}

    # ---- filters ----
    lat_lo, lat_hi = bbox["lat_min"] - margin, bbox["lat_max"] + margin
    lon_lo, lon_hi = bbox["lon_min"] - margin, bbox["lon_max"] + margin

    def in_region(lat, lon):
        return lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi

    kept_trips = []
    n_out_of_region = 0
    n_bad_time = 0
    n_short = 0
    n_trimmed = 0
    used_stops: set[str] = set()
    for tid, rows in seq.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r[0])
        first_dep = rows_sorted[0][1]
        if not (time_window[0] <= first_dep <= time_window[1]):
            n_bad_time += 1
            continue
        stop_ids = [r[3] for r in rows_sorted]
        # supply cropping (plan §2.4): keep the LONGEST in-region consecutive
        # segment (>=2 stops); trips fully in region are kept whole.
        in_reg = [in_region(*stops[sid][:2]) for sid in stop_ids]
        runs = []
        i = 0
        while i < len(in_reg):
            if in_reg[i]:
                j = i
                while j + 1 < len(in_reg) and in_reg[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        if not runs:
            n_out_of_region += 1
            continue
        start, end = max(runs, key=lambda r: r[1] - r[0])
        if end - start + 1 < 2:
            n_short += 1
            continue
        trimmed = (start != 0) or (end != len(stop_ids) - 1)
        if trimmed:
            n_trimmed += 1
        info = trip_info[tid]
        seg_rows = rows_sorted[start:end + 1]
        seg_first_dep = seg_rows[0][1]
        seg_last_arr = seg_rows[-1][2]
        kept_trips.append({
            "trip_id": tid if not trimmed else f"{tid}_t{start}",
            "gtfs_trip_id": tid,
            "route_id": info["route_id"],
            "short_name": info["short_name"],
            "mode": info["mode"],
            "trimmed": trimmed,
            "first_dep_sec": seg_first_dep,
            "last_arr_sec": seg_last_arr,
            "stop_sequence": [
                {"stop_id": sid, "arr_sec": arr, "dep_sec": dep}
                for (_sq, arr, dep, sid) in seg_rows
            ],
        })
        used_stops.update(sid for (_sq, _a, _d, sid) in seg_rows)

    kept_trips.sort(key=lambda t: (t["first_dep_sec"], t["trip_id"]))
    stop_rows = []
    for sid in sorted(used_stops):
        lat, lon, name = stops[sid]
        x, y = utm48n(lat, lon)
        stop_rows.append({"stop_id": sid, "name": name, "lat": lat, "lon": lon, "x": round(x, 2), "y": round(y, 2)})

    with (out / "prep_stops.jsonl").open("w", encoding="utf-8") as f:
        for r in stop_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "prep_trips.jsonl").open("w", encoding="utf-8") as f:
        for r in kept_trips:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "zip": str(zip_path),
        "service_id": service_id,
        "bbox": bbox,
        "margin_deg": margin,
        "time_window": time_window,
        "supply_cropping": "longest in-region consecutive stop segment per trip (Phase A §2.4)",
        "trips_kept": len(kept_trips),
        "trips_trimmed": n_trimmed,
        "trips_dropped_no_region_run": n_out_of_region,
        "trips_dropped_time_window": n_bad_time,
        "trips_dropped_too_short": n_short,
        "stops_kept": len(stop_rows),
        "mode_counts": {"bus": sum(1 for t in kept_trips if t["mode"] == "bus"),
                        "rail": sum(1 for t in kept_trips if t["mode"] == "rail")},
    }
    (out / "prep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


if __name__ == "__main__":
    prep_gtfs("data/singapore/gtfs/raw/singapore-gtfs.zip", "data/singapore/transit")
