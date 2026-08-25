"""Shared event-based scenario metrics for the Singapore runs (B.5B/B.5C/Phase C).

- ``load_link_attrs``: network.xml -> {link: (length, freespeed, capacity)} for car links
- ``collect_metrics``: stream it.0 events (.zst) -> trip/car/congestion metrics
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

_CAR = re.compile(r"^P\d+$")
_PERSON = re.compile(r"^P\d+$")


def load_link_attrs(network: str) -> dict[str, tuple[float, float, float]]:
    txt = open(network, encoding="utf-8").read()
    out = {}
    for m in re.finditer(
        r'<link id="([^"]+)" from="[^"]+" to="[^"]+" length="([\d.]+)" '
        r'freespeed="([\d.]+)" capacity="([\d.]+)"[^>]*modes="([^"]+)"',
        txt,
    ):
        if "car" in m.group(5).split(","):
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def collect_metrics(events_path: str | Path, links: dict) -> dict:
    """Event-based metrics from an it.0 events file (.zst)."""
    import zstandard
    ev = Path(events_path)
    if not ev.exists():
        return {"events_missing": True}

    dctx = zstandard.ZstdDecompressor()
    enter_t: dict[str, float] = {}
    car_dep: dict[str, float] = {}
    dep_stack: dict[str, list[float]] = defaultdict(list)
    n_dep = Counter()
    n_arr = Counter()
    pt_boardings = 0
    pt_alightings = 0
    stuck = 0
    waiting_pt = 0
    trip_time = 0.0
    n_trip_legs = 0
    car_time = 0.0
    n_car_legs = 0
    car_passages = 0
    car_delay_sum = 0.0
    car_slow_passages = 0
    car_vkt_m = 0.0
    link_delay: dict[str, list[float]] = defaultdict(list)

    with dctx.stream_reader(open(ev, "rb")) as f:
        for event, el in ET.iterparse(f, events=("end",)):
            if el.tag != "event":
                continue
            t = float(el.get("time"))
            etype = el.get("type")
            if etype == "departure":
                person = el.get("person")
                if person and _PERSON.match(person):
                    n_dep[person] += 1
                    dep_stack[person].append(t)
                    if el.get("legMode") == "car":
                        car_dep[person] = t
            elif etype == "arrival":
                person = el.get("person")
                if person and _PERSON.match(person):
                    n_arr[person] += 1
                    if dep_stack[person]:
                        trip_time += t - dep_stack[person].pop()
                        n_trip_legs += 1
                    if el.get("legMode") == "car" and person in car_dep:
                        car_time += t - car_dep[person]
                        n_car_legs += 1
                        del car_dep[person]
            elif etype == "PersonEntersPtVehicle":
                pt_boardings += 1
            elif etype == "PersonLeavesPtVehicle":
                pt_alightings += 1
            elif etype == "stuckAndAbort":
                stuck += 1
            elif etype == "waitingForPt":
                waiting_pt += 1
            elif etype == "entered link":
                vid = el.get("vehicle")
                if vid and _CAR.match(vid):
                    enter_t[vid] = t
            elif etype == "left link":
                vid = el.get("vehicle")
                if vid in enter_t:
                    lid = el.get("link")
                    attrs = links.get(lid)
                    if attrs is None:
                        del enter_t[vid]
                        continue
                    length, free, _cap = attrs
                    dt = t - enter_t[vid]
                    if dt <= 300.0 and free > 0 and length > 0:
                        freeflow = length / free
                        delay = max(0.0, dt - freeflow)
                        car_passages += 1
                        car_delay_sum += delay
                        car_vkt_m += length
                        link_delay[lid].append(delay)
                        if dt > freeflow + 15.0:
                            car_slow_passages += 1
                    del enter_t[vid]
            el.clear()

    failed = sum(max(0, n_dep[p] - n_arr[p]) for p in set(n_dep) | set(n_arr))
    mean_link_delays = sorted(sum(d) / len(d) for d in link_delay.values() if d)
    n_links_used = len(mean_link_delays)

    def _share_above(threshold):
        return round(sum(1 for d in mean_link_delays if d > threshold) / max(1, n_links_used), 4)

    return {
        "pt_boardings": pt_boardings,
        "pt_alightings": pt_alightings,
        "waiting_for_pt": waiting_pt,
        "stuck_and_abort": stuck,
        "failed_trips": failed,
        "mean_trip_time_min": round(trip_time / max(1, n_trip_legs) / 60.0, 2),
        "n_trip_legs": n_trip_legs,
        "car_mean_travel_time_min": round(car_time / max(1, n_car_legs) / 60.0, 2),
        "n_car_legs": n_car_legs,
        "road_delay_mean_s_per_passage": round(car_delay_sum / max(1, car_passages), 2),
        "network_congestion_slow_share": round(car_slow_passages / max(1, car_passages), 4),
        "car_link_passages": car_passages,
        "car_vkt_km": round(car_vkt_m / 1000.0, 1),
        "car_links_used": n_links_used,
        "share_links_delay_gt_15s": _share_above(15.0),
        "share_links_delay_gt_30s": _share_above(30.0),
        "share_links_delay_gt_60s": _share_above(60.0),
        "mean_link_delay_p50_s": round(mean_link_delays[n_links_used // 2], 2) if n_links_used else 0.0,
        "mean_link_delay_p90_s": round(mean_link_delays[int(n_links_used * 0.9)], 2) if n_links_used else 0.0,
    }
