"""Stop snapping + route routing + transitSchedule/transitVehicles XML.

Phase A supply step. Converts the region-filtered GTFS trips into an executable
MATSim scheduled-transit supply:

1. every stop is snapped to its NEAREST network node;
2. two artificial links per stop connect the node to the stop point
   (``ai_in_<stop>`` node->stop, ``ai_out_<stop>`` stop->node), so a transit
   stop always sits at the END of its reference link (MATSim convention);
3. each trip's stop sequence is routed over the car-allowed road graph with
   A* (free-flow travel time; admissible euclidean heuristic). Routing runs
   once per UNIQUE stop sequence and is shared by all trips of that pattern;
4. transitSchedule.xml gets one transitRoute per GTFS trip (segment) with its
   stop times as route-profile offsets and its first departure as the single
   departure; transitVehicles.xml gets one vehicle per trip.

MRT trips are treated as rail-on-road (explicit Phase A degradation: OSM
conversion carries no railway ways, so MRT routes follow the road graph;
transportMode is kept as "rail" and the degradation is recorded in the
routing report).

Outputs (data/singapore/transit/):
  transitSchedule.xml, transitVehicles.xml,
  network_with_transit.xml (road network + artificial access links),
  stop_snapping_report.json, route_routing_report.json,
  trips_by_stop.json (adapter pt-planning index),
  activity_nodes.json (candidate activity locations near stops).
"""
from __future__ import annotations

import json
import math
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from .osm_network import load_network_graph

_AI_FREESPEED = 11.1  # m/s, artificial access links (40 km/h)
_AI_CAPACITY = 9999.0
_MAX_SPEED = 25.0  # m/s, admissible heuristic bound


def _hms(sec: int) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _nearest_nodes(node_ids, xy, stops):
    """Vectorized nearest-network-node search for each stop (chunked)."""
    ids = list(node_ids)
    pts = np.array([xy[i] for i in ids], dtype=np.float64)
    stop_pts = np.array([[s["x"], s["y"]] for s in stops], dtype=np.float64)
    best = np.empty(len(stops), dtype=np.int64)
    best_d = np.empty(len(stops), dtype=np.float64)
    chunk = 100
    for i in range(0, len(stops), chunk):
        sp = stop_pts[i:i + chunk]  # (c, 2)
        d = ((sp[:, None, :] - pts[None, :, :]) ** 2).sum(-1)  # (c, N)
        best[i:i + chunk] = d.argmin(1)
        best_d[i:i + chunk] = d.min(1) ** 0.5
    return ids, best, best_d


def build_transit(
    network_path: str | Path,
    stops_path: str | Path,
    trips_path: str | Path,
    out_dir: str | Path,
    peak_window: tuple[int, int] | None = None,
    time_windows: list[tuple[int, int]] | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    stops = [json.loads(l) for l in Path(stops_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    trips = [json.loads(l) for l in Path(trips_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if time_windows is not None:
        trips = [t for t in trips if any(w[0] <= t["first_dep_sec"] <= w[1] for w in time_windows)]
    elif peak_window is not None:
        trips = [t for t in trips if peak_window[0] <= t["first_dep_sec"] <= peak_window[1]]
    stops_ids_used = {s for t in trips for s in (x["stop_id"] for x in t["stop_sequence"])}
    stops = [s for s in stops if s["stop_id"] in stops_ids_used]
    print(f"trips (windows)={len(trips)} stops={len(stops)}")
    g, xy = load_network_graph(network_path)

    # snapping is restricted to nodes of the LARGEST CAR component with at
    # least one LOCAL road edge (freespeed < 14 m/s): stops near the PIE/TPE
    # expressways must not snap onto the motorway itself (one-way chains make
    # motorway nodes directionally unreachable for bus routing).
    car_ug = nx.Graph()
    for u, v, d in g.edges(data=True):
        if "car" in d["modes"]:
            car_ug.add_edge(u, v)
    largest = max(nx.connected_components(car_ug), key=len)
    local_nodes = set()
    for u, v, d in g.edges(data=True):
        if "car" in d["modes"] and d.get("free", 0.0) < 14.0:
            local_nodes.add(u)
            local_nodes.add(v)
    snap_nodes = sorted(largest & local_nodes)
    print(f"car largest component: {len(largest)} nodes; local-road snap candidates: {len(snap_nodes)}")
    ids, nearest, snap_dist = _nearest_nodes(snap_nodes, xy, stops)

    snap_report = {"n_stops": len(stops),
                   "dist_m": {"mean": round(float(snap_dist.mean()), 1),
                              "p50": round(float(np.percentile(snap_dist, 50)), 1),
                              "p90": round(float(np.percentile(snap_dist, 90)), 1),
                              "max": round(float(snap_dist.max()), 1)},
                   "per_stop": []}
    for s, nid_idx, d in zip(stops, nearest, snap_dist):
        snap_report["per_stop"].append({"stop_id": s["stop_id"], "nearest_node": str(ids[nid_idx]), "dist_m": round(float(d), 1)})
    (out / "stop_snapping_report.json").write_text(json.dumps(snap_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"snapping: mean={snap_report['dist_m']['mean']}m p90={snap_report['dist_m']['p90']}m max={snap_report['dist_m']['max']}m")

    # artificial links node<->stop for every stop
    stop_node = {s["stop_id"]: ids[n] for s, n in zip(stops, nearest)}
    ai_links: list[dict] = []
    for s in stops:
        sid = s["stop_id"]
        nid = stop_node[sid]
        x0, y0 = xy[nid]
        length = max(math.hypot(s["x"] - x0, s["y"] - y0), 1.0)
        ai_links.append({"id": f"ai_in_{sid}", "from": nid, "to": f"stp_{sid}",
                         "x": s["x"], "y": s["y"], "length": length})
        ai_links.append({"id": f"ai_out_{sid}", "from": f"stp_{sid}", "to": nid,
                         "x": s["x"], "y": s["y"], "length": length})

    # routing graph: car modes only (bus drives like a car); one-way links get
    # a TRANSIT-ONLY reverse edge (documented Phase A approximation, same trick
    # as pt2matsim): GTFS bus routes must stay routable even where OSM one-way
    # modeling + snapping errors would otherwise break the chain. Car modes are
    # NOT affected (the reverse links only allow bus/rail).
    oneway_edges = {(u, v): d for u, v, d in g.edges(data=True)
                    if "car" in d["modes"] and not g.has_edge(v, u)}
    rg = nx.DiGraph()
    for u, v, d in g.edges(data=True):
        if "car" in d["modes"]:
            rg.add_edge(u, v, tt=d["length"] / max(d["free"], 0.1))
    for (u, v), d in oneway_edges.items():
        rg.add_edge(v, u, tt=d["length"] / max(d["free"], 0.1), bus_reverse=True)
    for nid, (x, y) in xy.items():
        rg.add_node(nid, x=x, y=y)
    for ai in ai_links:
        sid = ai["id"].split("_", 2)[-1]
        rg.add_node(ai["to"], x=ai["x"], y=ai["y"])
        rg.add_edge(ai["from"], ai["to"], tt=ai["length"] / _AI_FREESPEED)
        rg.add_edge(ai["to"], ai["from"], tt=ai["length"] / _AI_FREESPEED)

    # unique stop sequences -> shared routing; per-PAIR path cache
    seq_trips: dict[tuple, list[dict]] = defaultdict(list)
    for t in trips:
        seq = tuple(x["stop_id"] for x in t["stop_sequence"])
        if len(set(seq)) < 2:
            continue
        seq_trips[seq].append(t)

    route_failures = []
    route_by_seq: dict[tuple, list[str]] = {}
    pair_cache: dict[tuple[str, str], list[str] | None] = {}
    n_seq = len(seq_trips)

    def _pair_path(a: str, b: str) -> list[str] | None:
        """Link-id path from stop a to stop b (ai_out_a -> road -> ai_in_b)."""
        key = (a, b)
        if key in pair_cache:
            return pair_cache[key]
        try:
            path = nx.astar_path(
                rg, f"stp_{a}", f"stp_{b}",
                weight="tt",
                heuristic=lambda u, v: math.hypot(rg.nodes[u]["x"] - rg.nodes[v]["x"],
                                                  rg.nodes[u]["y"] - rg.nodes[v]["y"]) / _MAX_SPEED,
            )
        except nx.NetworkXNoPath:
            pair_cache[key] = None
            return None
        edge_ids = []
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            if u.startswith("stp_"):
                edge_ids.append(f"ai_out_{u[4:]}")
            elif v.startswith("stp_"):
                edge_ids.append(f"ai_in_{v[4:]}")
            elif g.has_edge(u, v):
                edge_ids.append(f"{u}_{v}")
            else:
                # transit-only reverse of a one-way link
                edge_ids.append(f"busr_{v}_{u}")
        pair_cache[key] = edge_ids
        return edge_ids

    for si, (seq, trip_group) in enumerate(seq_trips.items()):
        links: list[str] = []
        ok = True
        for i in range(len(seq) - 1):
            sub = _pair_path(seq[i], seq[i + 1])
            if sub is None:
                ok = False
                route_failures.append({"stops": [seq[i], seq[i + 1]], "reason": "no path",
                                       "n_trips": len(trip_group)})
                break
            links.extend(sub)
        if ok:
            # the route must START with the first stop's access link (ai_in): the
            # vehicle spawns before the first stop, drives into it, and only then
            # departs along ai_out. Without the leading ai_in the vehicle would be
            # one stop behind and die with "not yet at last stop".
            route_by_seq[seq] = [f"ai_in_{seq[0]}"] + links
        if si % 200 == 0:
            print(f"  routed {si}/{n_seq} sequences, cache={len(pair_cache)} ({time.time()-t0:.0f}s)")

    print(f"unique sequences={n_seq} routed={len(route_by_seq)} failed={len(route_failures)} ({time.time()-t0:.0f}s)")

    # ---- transitSchedule.xml ----
    sched = ET.Element("transitSchedule")
    stops_el = ET.SubElement(sched, "transitStops")
    for s in stops:
        ET.SubElement(
            stops_el, "stopFacility",
            id=s["stop_id"], x=f"{s['x']:.2f}", y=f"{s['y']:.2f}",
            linkRefId=f"ai_in_{s['stop_id']}", name=s["name"], isBlocking="false",
        )
    lines: dict[str, ET.Element] = {}
    for t in trips:
        seq = tuple(x["stop_id"] for x in t["stop_sequence"])
        if seq not in route_by_seq:
            continue
        rid = t["route_id"]
        if rid not in lines:
            lines[rid] = ET.SubElement(sched, "transitLine", id=rid, name=t["short_name"])
        line = lines[rid]
        tr = ET.SubElement(line, "transitRoute", id=t["trip_id"])
        ET.SubElement(tr, "transportMode").text = t["mode"]
        profile = ET.SubElement(tr, "routeProfile")
        first_dep = t["first_dep_sec"]
        seq_rows = t["stop_sequence"]
        for k, row in enumerate(seq_rows):
            attrs = {"refId": row["stop_id"]}
            if k == 0:
                attrs["departureOffset"] = "00:00:00"
                attrs["awaitDeparture"] = "true"
            elif k == len(seq_rows) - 1:
                attrs["arrivalOffset"] = _hms(row["arr_sec"] - first_dep)
                attrs["awaitDeparture"] = "false"
            else:
                attrs["arrivalOffset"] = _hms(row["arr_sec"] - first_dep)
                attrs["departureOffset"] = _hms(row["dep_sec"] - first_dep)
                attrs["awaitDeparture"] = "true"
            ET.SubElement(profile, "stop", **attrs)
        route_el = ET.SubElement(tr, "route")
        for lid in route_by_seq[seq]:
            ET.SubElement(route_el, "link", refId=lid)
        deps = ET.SubElement(tr, "departures")
        ET.SubElement(deps, "departure", id=f"dep_{t['trip_id']}",
                      departureTime=_hms(first_dep), vehicleRefId=f"veh_{t['trip_id']}")

    ET.ElementTree(sched).write(out / "transitSchedule.xml", encoding="utf-8", xml_declaration=True)
    _insert_doctype(out / "transitSchedule.xml",
                    '<!DOCTYPE transitSchedule SYSTEM "http://www.matsim.org/files/dtd/transitSchedule_v1.dtd">')

    # ---- transitVehicles.xml (MATSim 2026 XSD format) ----
    v_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.matsim.org/files/dtd '
        'http://www.matsim.org/files/dtd/vehicleDefinitions_v1.0.xsd">',
        ' <vehicleType id="busType">',
        '  <capacity><seats persons="40"/><standingRoom persons="20"/></capacity>',
        '  <length meter="12.0"/>',
        '  <accessTime secondsPerPerson="1.0"/>',
        '  <doorOperation mode="serial"/>',
        '  <passengerCarEquivalents pce="2.0"/>',
        ' </vehicleType>',
        ' <vehicleType id="railType">',
        '  <capacity><seats persons="200"/><standingRoom persons="300"/></capacity>',
        '  <length meter="138.0"/>',
        '  <accessTime secondsPerPerson="1.0"/>',
        '  <doorOperation mode="serial"/>',
        '  <passengerCarEquivalents pce="1.0"/>',
        ' </vehicleType>',
    ]
    for t in trips:
        seq = tuple(x["stop_id"] for x in t["stop_sequence"])
        if seq not in route_by_seq:
            continue
        v_lines.append(f' <vehicle id="veh_{t["trip_id"]}" type="{"railType" if t["mode"] == "rail" else "busType"}"/>')
    v_lines.append("</vehicleDefinitions>")
    (out / "transitVehicles.xml").write_text("\n".join(v_lines) + "\n", encoding="utf-8")

    # ---- network + artificial links + transit-only one-way reverses ----
    root = ET.parse(network_path).getroot()
    links_el = root.find("links")
    nodes_el = root.find("nodes")
    for ai in ai_links:
        if not ai["id"].startswith("ai_in_"):  # process each stop once (via its ai_in entry)
            continue
        ET.SubElement(nodes_el, "node", id=ai["to"], x=f"{ai['x']:.2f}", y=f"{ai['y']:.2f}")
        for lid, fr, to in ((ai["id"], ai["from"], ai["to"]), (ai["id"].replace("in_", "out_"), ai["to"], ai["from"])):
            ET.SubElement(
                links_el, "link", id=lid, **{"from": fr, "to": to,
                                              "length": f"{ai['length']:.2f}",
                                              "freespeed": str(_AI_FREESPEED),
                                              "capacity": str(_AI_CAPACITY),
                                              "permlanes": "1.0", "oneway": "1",
                                              "modes": "car,bike,walk,bus,rail"},
            )
    n_bus_reverse = 0
    for (u, v), d in oneway_edges.items():
        ET.SubElement(
            links_el, "link", id=f"busr_{u}_{v}", **{"from": v, "to": u,
                                                     "length": f"{d['length']:.2f}",
                                                     "freespeed": f"{d['free']:.2f}",
                                                     "capacity": f"{d['capacity']:.1f}",
                                                     "permlanes": "1.0", "oneway": "1",
                                                     "modes": "bus,rail"},
        )
        n_bus_reverse += 1
    # pedestrian/bike reverse links on one-way streets (walkers and cyclists
    # legally use both sides of a street; the car direction is one-way only)
    n_ped_reverse = 0
    for (u, v), d in oneway_edges.items():
        ped_modes = [m for m in ("walk", "bike") if m in d["modes"]]
        if not ped_modes:
            continue
        ET.SubElement(
            links_el, "link", id=f"pdr_{u}_{v}", **{"from": v, "to": u,
                                                    "length": f"{d['length']:.2f}",
                                                    "freespeed": f"{d['free']:.2f}",
                                                    "capacity": f"{d['capacity']:.1f}",
                                                    "permlanes": "1.0", "oneway": "1",
                                                    "modes": ",".join(ped_modes)},
        )
        n_ped_reverse += 1
    ET.ElementTree(root).write(out / "network_with_transit.xml", encoding="utf-8", xml_declaration=True)
    _insert_doctype(out / "network_with_transit.xml",
                    '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">')

    # ---- adapter indexes ----
    trips_by_stop: dict[str, list[dict]] = defaultdict(list)
    for t in trips:
        seq = tuple(x["stop_id"] for x in t["stop_sequence"])
        if seq not in route_by_seq:
            continue
        for k, row in enumerate(t["stop_sequence"]):
            trips_by_stop[row["stop_id"]].append({
                "trip_id": t["trip_id"], "route_id": t["route_id"],
                "mode": t["mode"], "seq_idx": k,
                "dep_sec": row["dep_sec"], "arr_sec": row["arr_sec"],
            })
    (out / "trips_by_stop.json").write_text(json.dumps(trips_by_stop, ensure_ascii=False), encoding="utf-8")

    # activity/home candidate nodes: real road nodes within 300 m of a stop,
    # with at least one incident link (isolated nodes cannot host activities)
    stop_pts = np.array([[s["x"], s["y"]] for s in stops])
    node_ids = [n for n in g.nodes if g.degree(n) > 0]
    node_pts = np.array([xy[n] for n in node_ids])
    candidates: set[str] = set()
    for i in range(0, len(stop_pts), 50):
        sp = stop_pts[i:i + 50]
        d = ((sp[:, None, :] - node_pts[None, :, :]) ** 2).sum(-1)
        idx = np.argwhere(d <= 300.0 ** 2)
        for r in idx:
            candidates.add(node_ids[int(r[1])])
    activity_nodes = [{"node": nid, "x": xy[nid][0], "y": xy[nid][1]} for nid in sorted(candidates)]
    (out / "activity_nodes.json").write_text(json.dumps(activity_nodes, ensure_ascii=False), encoding="utf-8")
    print(f"activity candidate nodes within 300m of stops: {len(activity_nodes)}")

    report = {
        "n_stops": len(stops),
        "n_trips": len(trips),
        "n_trips_routed": sum(1 for t in trips if tuple(x['stop_id'] for x in t['stop_sequence']) in route_by_seq),
        "n_unique_sequences": n_seq,
        "n_sequences_routed": len(route_by_seq),
        "routing_failures": route_failures[:50],
        "n_routing_failures": len(route_failures),
        "snapping_dist_m": snap_report["dist_m"],
        "n_transit_only_oneway_reverse_links": n_bus_reverse,
        "n_pedestrian_bike_reverse_links": n_ped_reverse,
        "runtime_s": round(time.time() - t0, 1),
        "rail_degradation": "MRT trips routed on the road graph (rail-on-road); OSM railway not converted in Phase A",
        "oneway_approximation": "transit routes may traverse one-way links against direction via "
                                "transit-only reverse links (modes=bus,rail); car traffic is unaffected — "
                                "documented Phase A approximation",
    }
    (out / "route_routing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "routing_failures"}, ensure_ascii=False, indent=2))
    return report


def _insert_doctype(path: Path, doctype: str) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out_lines = []
    inserted = False
    for line in lines:
        out_lines.append(line)
        if not inserted and line.strip().startswith("<?xml"):
            out_lines.append(doctype)
            inserted = True
    if not inserted:
        out_lines.insert(0, doctype)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_transit(
        "data/singapore/osm/network.xml",
        "data/singapore/transit/prep_stops.jsonl",
        "data/singapore/transit/prep_trips.jsonl",
        "data/singapore/transit",
    )
