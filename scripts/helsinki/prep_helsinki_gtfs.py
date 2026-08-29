#!/usr/bin/env python
"""Helsinki (HSL) GTFS -> region-filtered stops/trips (E5 Phase A prep step).

HSL-specific adaptation of the Singapore prep (`src/traveler_distillation/
singapore/gtfs_prep.py`, which is NOT modified). Differences from Singapore:

- route_type mapping: HSL uses extended types — 0/900 tram, 1 metro,
  109 commuter rail -> "rail"; 3 and 700-716 bus types -> "bus"; 4 ferry and
  unknown types are EXCLUDED (counted in meta).
- service selection: HSL uses per-route service_ids with date-range + weekday
  suffixes. The Singapore heuristic ("pick the one service_id with most trips")
  does not apply. Standard GTFS service-day logic is used: reference day
  2026-09-09 (Wednesday, mid-validity), service_id active iff calendar covers
  the date with the weekday flag, or calendar_dates adds (type 1) / removes
  (type 2) it for that date.
- stops: keep location_type=0 platforms only (parent_station used for QA
  naming only); `parent_station=' '` treated as empty.
- stop_times include >=24h times (e.g. "25:10:00") — the Singapore
  `_hhmmss_to_sec` parse handles them as-is (int hours).

Everything else (bbox cropping with margin, longest in-region consecutive stop
segment per trip, time window [05:00, 23:00], output schema) mirrors the
Singapore prep verbatim so `build_transit.py` consumes the files unchanged.

Usage:
    python scripts/helsinki/prep_helsinki_gtfs.py \
        hsl/hsl.zip data/helsinki/transit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traveler_distillation.singapore.projection import utm35n  # noqa: E402

# E5 default subregion (design v0.1 §3.1)
BBOX = {"lat_min": 60.145, "lat_max": 60.230, "lon_min": 24.875, "lon_max": 25.060}
MARGIN = 0.004  # ~450 m, same as Singapore
TIME_WINDOW = (5 * 3600, 23 * 3600)
REFERENCE_DAY = "2026-09-09"  # Wednesday, mid-validity of the HSL feed

# HSL route_type -> output mode (rail/bus); None = excluded
_RAIL_TYPES = {"0", "1", "100", "109", "900"}


def route_mode(route_type: str) -> str | None:
    rt = (route_type or "").strip()
    if rt in _RAIL_TYPES:
        return "rail"
    if rt == "3":
        return "bus"
    if rt.isdigit() and 700 <= int(rt) <= 716:
        return "bus"
    return None  # ferry (4), unknown -> excluded


def _hhmmss_to_sec(s: str) -> int:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def _stream_rows(zf: zipfile.ZipFile, name: str):
    with zf.open(name) as f:
        text = io.TextIOWrapper(f, encoding="utf-8-sig")
        yield from csv.DictReader(text)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prep_gtfs(
    zip_path: str | Path,
    out_dir: str | Path,
    bbox: dict | None = None,
    margin: float = MARGIN,
    time_window: tuple[int, int] = TIME_WINDOW,
    reference_day: str = REFERENCE_DAY,
) -> dict:
    bbox = bbox or BBOX
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = date.fromisoformat(reference_day)
    weekday_col = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][d.weekday()]
    day_compact = reference_day.replace("-", "")
    print(f"reference day {reference_day} ({weekday_col}); start_date <= {day_compact} <= end_date")

    excluded = {"service_inactive": 0, "route_excluded": 0, "ferry_or_unknown": 0}
    with zipfile.ZipFile(Path(zip_path)) as zf:
        names = set(zf.namelist())
        assert "calendar.txt" in names and "trips.txt" in names and "stop_times.txt" in names

        # ---- active service ids (calendar + calendar_dates) ----
        active: set[str] = set()
        n_cal_rows = 0
        for row in _stream_rows(zf, "calendar.txt"):
            n_cal_rows += 1
            sd, ed = row.get("start_date") or "", row.get("end_date") or ""
            if sd <= day_compact <= ed and int(row.get(weekday_col) or 0) == 1:
                active.add(row["service_id"])
        n_cd_add = n_cd_rm = 0
        for row in _stream_rows(zf, "calendar_dates.txt"):
            if (row.get("date") or "") != day_compact:
                continue
            et = row.get("exception_type")
            if et == "1":
                active.add(row["service_id"])
                n_cd_add += 1
            elif et == "2":
                active.discard(row["service_id"])
                n_cd_rm += 1
        print(f"calendar rows={n_cal_rows} active service ids={len(active)} (cd +{n_cd_add}/-{n_cd_rm})")

        # ---- routes: id -> (mode, short_name); excluded counted ----
        route_mode_map: dict[str, str] = {}
        route_name: dict[str, str] = {}
        route_type_counts: dict[str, int] = {}
        for row in _stream_rows(zf, "routes.txt"):
            rt = row.get("route_type") or ""
            route_type_counts[rt] = route_type_counts.get(rt, 0) + 1
            m = route_mode(rt)
            if m is not None:
                route_mode_map[row["route_id"]] = m
                route_name[row["route_id"]] = row.get("route_short_name") or row.get("route_long_name") or row["route_id"]

        # ---- trips of the reference day + mapped routes ----
        trip_info: dict[str, dict] = {}
        for row in _stream_rows(zf, "trips.txt"):
            sid, rid = row["service_id"], row["route_id"]
            if sid not in active:
                excluded["service_inactive"] += 1
                continue
            if rid not in route_mode_map:
                excluded["route_excluded"] += 1
                continue
            trip_info[row["trip_id"]] = {
                "route_id": rid,
                "mode": route_mode_map[rid],
                "short_name": route_name[rid],
            }

        # ---- stop_times: stream once (976 MB), keep kept trips ----
        seq: dict[str, list[tuple[int, int, int, str]]] = {t: [] for t in trip_info}
        for row in _stream_rows(zf, "stop_times.txt"):
            tid = row["trip_id"]
            if tid in seq:
                seq[tid].append(
                    (int(row["stop_sequence"]), _hhmmss_to_sec(row["arrival_time"]),
                     _hhmmss_to_sec(row["departure_time"]), row["stop_id"])
                )

        # ---- stops: platforms only (location_type=0) ----
        stops: dict[str, tuple[float, float, str, str]] = {}
        n_parent_stations = 0
        for r in _stream_rows(zf, "stops.txt"):
            lt = (r.get("location_type") or "").strip()
            if lt == "1":
                n_parent_stations += 1
                continue
            if lt not in ("", "0"):
                continue
            stops[r["stop_id"]] = (float(r["stop_lat"]), float(r["stop_lon"]),
                                   r["stop_name"], (r.get("parent_station") or "").strip())

    # ---- filters (mirror Singapore) ----
    lat_lo, lat_hi = bbox["lat_min"] - margin, bbox["lat_max"] + margin
    lon_lo, lon_hi = bbox["lon_min"] - margin, bbox["lon_max"] + margin

    def in_region(lat, lon):
        return lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi

    kept_trips = []
    n_out_of_region = n_bad_time = n_short = n_trimmed = 0
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
        kept_trips.append({
            "trip_id": tid if not trimmed else f"{tid}_t{start}",
            "gtfs_trip_id": tid,
            "route_id": info["route_id"],
            "short_name": info["short_name"],
            "mode": info["mode"],
            "trimmed": trimmed,
            "first_dep_sec": seg_rows[0][1],
            "last_arr_sec": seg_rows[-1][2],
            "stop_sequence": [
                {"stop_id": sid, "arr_sec": arr, "dep_sec": dep}
                for (_sq, arr, dep, sid) in seg_rows
            ],
        })
        used_stops.update(sid for (_sq, _a, _d, sid) in seg_rows)

    kept_trips.sort(key=lambda t: (t["first_dep_sec"], t["trip_id"]))
    stop_rows = []
    for sid in sorted(used_stops):
        lat, lon, name, parent = stops[sid]
        x, y = utm35n(lat, lon)
        stop_rows.append({"stop_id": sid, "name": name, "lat": lat, "lon": lon,
                          "x": round(x, 2), "y": round(y, 2),
                          "parent_station": parent})

    with (out / "prep_stops.jsonl").open("w", encoding="utf-8") as f:
        for r in stop_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "prep_trips.jsonl").open("w", encoding="utf-8") as f:
        for r in kept_trips:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "city": "Helsinki",
        "zip": str(zip_path),
        "zip_sha256": _sha256(Path(zip_path)),
        "feed_version": None,
        "reference_day": reference_day,
        "weekday": weekday_col,
        "bbox": bbox,
        "margin_deg": margin,
        "time_window": time_window,
        "supply_cropping": "longest in-region consecutive stop segment per trip (Singapore Phase A §2.4 parity)",
        "route_type_counts": route_type_counts,
        "route_type_mapping": "0/900 tram, 1 metro, 100 rail, 109 commuter rail -> rail; 3, 700-716 -> bus; 4 ferry + unknown excluded",
        "active_service_ids": len(active),
        "excluded_counts": excluded,
        "n_parent_stations_skipped": n_parent_stations,
        "trips_kept": len(kept_trips),
        "trips_trimmed": n_trimmed,
        "trips_dropped_no_region_run": n_out_of_region,
        "trips_dropped_time_window": n_bad_time,
        "trips_dropped_too_short": n_short,
        "stops_kept": len(stop_rows),
        "mode_counts": {"bus": sum(1 for t in kept_trips if t["mode"] == "bus"),
                        "rail": sum(1 for t in kept_trips if t["mode"] == "rail")},
        "note_gt24h_times": "HSL stop_times contain >=24h timestamps; parsed as plain hh*3600 (Singapore-identical code)",
    }
    with zipfile.ZipFile(Path(zip_path)) as zf:
        if "feed_info.txt" in zf.namelist():
            for r in _stream_rows(zf, "feed_info.txt"):
                meta["feed_version"] = r.get("feed_version")
                meta["feed_start_date"] = r.get("feed_start_date")
                meta["feed_end_date"] = r.get("feed_end_date")
                break
    (out / "prep_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", default="hsl/hsl.zip")
    ap.add_argument("out_dir", default="data/helsinki/transit")
    ap.add_argument("--reference-day", default=REFERENCE_DAY)
    args = ap.parse_args()
    prep_gtfs(args.zip_path, args.out_dir, reference_day=args.reference_day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
