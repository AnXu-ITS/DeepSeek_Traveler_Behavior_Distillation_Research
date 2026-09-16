#!/usr/bin/env python
"""Fetch Shanghai metro stations + line assignment from Amap POI (Web服务 key).

Reads AMAP_KEY from the environment or the repo ``.env``. Paginates the Amap
POI text search (types=150500 地铁站, city=上海), then filters to the Xuhui
study bbox. Each Amap POI's ``address`` field carries the metro line numbers
the station serves (e.g. "10号线;17号线;2号线"), which gives us line topology.

Output: data/shanghai/transit/amap_stations.json
        {station_id: {name, lon, lat, district, lines: [..], address}}
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

_WB = Path(__file__).resolve().parents[2]
BBOX = {"lat_min": 31.15, "lat_max": 31.23, "lon_min": 121.40, "lon_max": 121.48}


def load_key() -> str:
    k = os.environ.get("AMAP_KEY", "")
    if k:
        return k
    env = _WB / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("AMAP_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def parse_lines(address: str) -> list[str]:
    """'10号线;17号线;2号线' -> ['10','17','2'] (line numbers, in feed order)."""
    out = []
    for tok in re.split(r"[;,、/，\s]+", address or ""):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(\d+)号线", tok)
        if m:
            out.append(m.group(1))
        elif "号线" in tok or "线" in tok:
            out.append(tok)
    # de-dup keep order
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def main() -> int:
    key = load_key()
    if not key:
        print("ERROR: no AMAP_KEY", file=sys.stderr)
        return 2

    base = "https://restapi.amap.com/v3/place/text"
    all_stations: dict[str, dict] = {}
    page = 1
    total = None
    while True:
        r = httpx.get(
            base,
            params={"types": "150500", "city": "上海", "offset": "25",
                    "page": str(page), "extensions": "all", "key": key},
            timeout=30,
        )
        d = r.json()
        if d.get("infocode") != "10000":
            print(f"Amap error page {page}: {d.get('info')} {d.get('infocode')}", file=sys.stderr)
            return 1
        if total is None:
            total = int(d.get("count", 0))
            print(f"Amap metro stations total: {total}")
        for p in d.get("pois", []):
            loc = (p.get("location") or "").split(",")
            if len(loc) != 2:
                continue
            lon, lat = float(loc[0]), float(loc[1])
            all_stations[p.get("id")] = {
                "name": (p.get("name") or "").replace("(地铁站)", ""),
                "lon": lon,
                "lat": lat,
                "district": p.get("adname", ""),
                "address": p.get("address", ""),
                "lines": parse_lines(p.get("address", "")),
            }
        if page * 25 >= total:
            break
        page += 1
        time.sleep(0.15)

    in_bbox = {
        sid: s for sid, s in all_stations.items()
        if BBOX["lat_min"] <= s["lat"] <= BBOX["lat_max"]
        and BBOX["lon_min"] <= s["lon"] <= BBOX["lon_max"]
    }
    out = _WB / "data/shanghai/transit/amap_stations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(in_bbox, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"total Shanghai: {len(all_stations)}; in Xuhui bbox: {len(in_bbox)}")
    lines = {}
    for s in in_bbox.values():
        for ln in s["lines"]:
            lines.setdefault(ln, []).append(s["name"])
    for ln in sorted(lines, key=lambda x: (len(x), x)):
        print(f"  线 {ln:>4} : {len(lines[ln])} 站")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
