"""S8 — real-supply transit accessibility computation (city-independent).

Computes, for an (origin, destination, departure time) triple on the Singapore
OSM + GTFS supply, the S8 transit-accessibility feature vector:

    pt_feasible, access/egress/wait/in-vehicle/transfer times, transfer count,
    door-to-door time, coverage ratio (D_vehicle / D_OD, clipped to [0,1]).

Only NUMERIC, city-independent attributes are produced. Stop ids, route ids and
network node ids are internal routing handles and must never be forwarded into
the student/teacher input (S8 instructions §3).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from ..singapore.osm_network import load_network_graph

# accessibility class labels (S8 instructions §9)
CLASS_A = "A_excellent"
CLASS_B = "B_good"
CLASS_C = "C_moderate"
CLASS_D = "D_poor"
CLASS_E = "E_infeasible"

# infeasible-state sentinels (documented convention; travel_time must be > 0 per schema)
INFEASIBLE_TRAVEL_TIME_MIN = 120.0
INFEASIBLE_ZERO_FIELDS = (
    "access_time_min", "egress_time_min", "wait_time_min",
    "in_vehicle_time_min", "transfer_time_min", "coverage_ratio",
)


class SupplyIndex:
    """Loaded Singapore supply + search indexes (shared across queries)."""

    def __init__(self, supply_paths: dict[str, str | Path]):
        sp = {k: Path(v) for k, v in supply_paths.items()}
        self.g, self.xy = load_network_graph(sp["network"])
        self.tt_graphs = {}
        for mode in ("car", "bike", "walk"):
            sg = nx.DiGraph()
            for u, v, d in self.g.edges(data=True):
                if mode in d["modes"]:
                    sg.add_edge(u, v, tt=d["length"] / max(d["free"], 0.1))
            for n, (x, y) in self.xy.items():
                sg.add_node(n, x=x, y=y)
            self.tt_graphs[mode] = sg

        self.stops = {r["stop_id"]: (r["x"], r["y"]) for r in _read_jsonl(sp["stops"])}
        snap = json.loads(sp["snapping"].read_text(encoding="utf-8"))
        self.stop_node = {r["stop_id"]: r["nearest_node"] for r in snap["per_stop"]}
        self.trips_by_stop = json.loads(sp["trips_by_stop"].read_text(encoding="utf-8"))
        self.trip_seq: dict[str, dict[str, tuple[int, int, int]]] = {}
        self.stop_sorted: dict[str, list[dict]] = {}
        self.stop_trip_map: dict[str, dict[str, dict]] = {}
        for sid, entries in self.trips_by_stop.items():
            self.stop_sorted[sid] = sorted(entries, key=lambda t: t["dep_sec"])
            self.stop_trip_map[sid] = {t["trip_id"]: t for t in entries}
            for t in entries:
                self.trip_seq.setdefault(t["trip_id"], {})[sid] = (t["seq_idx"], t["arr_sec"], t["dep_sec"])
        # trip -> sorted stop sequence [(seq_idx, stop_id, arr_sec, dep_sec)] for
        # downstream/segment lookups
        self.trip_stops_sorted: dict[str, list[tuple[int, str, int, int]]] = {}
        for trip_id, seqs in self.trip_seq.items():
            self.trip_stops_sorted[trip_id] = sorted(
                (s, sid, a, d) for sid, (s, a, d) in seqs.items()
            )
        self.activity_nodes = json.loads(sp["activity_nodes"].read_text(encoding="utf-8"))

        car_nodes = {u for u, v, d in self.g.edges(data=True) if "car" in d["modes"]}
        car_nodes |= {v for u, v, d in self.g.edges(data=True) if "car" in d["modes"]}
        self.activity_nodes = [a for a in self.activity_nodes if a["node"] in car_nodes]
        walk_und = nx.Graph()
        walk_und.add_edges_from(self.tt_graphs["walk"].edges())
        walk_largest = max(nx.connected_components(walk_und), key=len)
        self.walk_largest = walk_largest
        self.activity_nodes = [a for a in self.activity_nodes if a["node"] in walk_largest]

        # full OD-candidate node set (car-accessible AND walk-largest component)
        self.candidate_nodes = sorted(set(self.xy.keys()) & car_nodes & walk_largest)
        # nearest-stop distance bands for stratified OD sampling
        stop_xy = np.array([(x, y) for x, y in self.stops.values()])
        node_xy = np.array([self.xy[n] for n in self.candidate_nodes])
        dist = np.zeros(len(node_xy))
        for c0 in range(0, len(node_xy), 5000):
            chunk = node_xy[c0:c0 + 5000]
            d = np.hypot(chunk[:, None, 0] - stop_xy[None, :, 0],
                         chunk[:, None, 1] - stop_xy[None, :, 1])
            dist[c0:c0 + 5000] = d.min(axis=1)
        self.band_nodes = {
            "near": [n for n, dd in zip(self.candidate_nodes, dist) if dd < 300.0],
            "mid": [n for n, dd in zip(self.candidate_nodes, dist) if 300.0 <= dd < 700.0],
            "far": [n for n, dd in zip(self.candidate_nodes, dist) if dd >= 700.0],
        }

        self._walk_cache: dict[tuple[str, str], tuple[list[str], float] | None] = {}

    # ------------------------------------------------------------------ utils
    def walk_route(self, src: str, dst: str) -> tuple[list[str], float] | None:
        """Shortest walk (node path, minutes). Cached."""
        key = (src, dst)
        if key in self._walk_cache:
            return self._walk_cache[key]
        res = self._walk_route_uncached(src, dst)
        if len(self._walk_cache) >= 200_000:
            self._walk_cache.clear()
        self._walk_cache[key] = res
        return res

    def _walk_route_uncached(self, src: str, dst: str) -> tuple[list[str], float] | None:
        g = self.tt_graphs["walk"]
        if src not in g or dst not in g:
            return None
        try:
            length_tt, path = nx.bidirectional_dijkstra(g, src, dst, weight="tt")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return list(path), length_tt / 60.0

    def mode_travel_time(self, mode: str, src: str, dst: str) -> float | None:
        """Car/bike/walk travel time (minutes) on the real network."""
        g = self.tt_graphs.get(mode)
        if g is None or src not in g or dst not in g:
            return None
        try:
            length_tt, _path = nx.bidirectional_dijkstra(g, src, dst, weight="tt")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return length_tt / 60.0

    @staticmethod
    def euclid_m(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


def _read_jsonl(path: str | Path) -> list[dict]:
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            out.append(json.loads(ln))
    return out


def plan_accessibility(
    idx: SupplyIndex,
    origin: str,
    dest: str,
    dep_sec: float,
    access_max_m: float = 700.0,
    buffer_s: float = 300.0,
) -> dict:
    """Plan the PT itinerary and return the S8 accessibility vector.

    Mirrors MATSimAdapter._plan_pt routing rules (direct trip first, then one
    transfer with >= 180 s connection, extended 1.5 km egress), but returns a
    time decomposition instead of MATSim legs.

    Returned dict (city-independent only — no stop/route/line ids):
        pt_feasible, fallback_reason, access_time_min, egress_time_min,
        wait_time_min, in_vehicle_time_min, transfer_time_min, transfer_count,
        door_to_door_min, coverage_ratio
    """
    x0, y0 = idx.xy[origin]
    x1, y1 = idx.xy[dest]

    def _near(node: str, k: int = 12, radius: float | None = None) -> list[str]:
        r = radius if radius is not None else access_max_m
        nx0, ny0 = idx.xy[node]
        cands = []
        for sid, (sx, sy) in idx.stops.items():
            d = math.hypot(sx - nx0, sy - ny0)
            if d <= r:
                cands.append((d, sid))
        cands.sort()
        return [sid for _, sid in cands[:k]]

    access_cands = []
    for o_stop in _near(origin):
        o_node = idx.stop_node.get(o_stop)
        if o_node is None:
            continue
        route = idx.walk_route(origin, o_node)
        if route is not None:
            access_cands.append((o_stop, route))
    # NOTE: no wider access fallback in the S8 feature pipeline. The S8 routing
    # rule is "boarding stop within 700 m of the origin" (documented in the
    # data manifest); a 1.5 km fallback would produce temporally impossible
    # itineraries (access walk longer than the boarding window). Origins with
    # no stop within 700 m are Class E (no_stops_in_radius).
    if not access_cands:
        return _infeasible("no_stops_in_radius")

    egress_cands = []
    d_seq_by_stop: dict[str, dict[str, int]] = {}
    for d_stop in _near(dest):
        route = idx.walk_route(idx.stop_node[d_stop], dest) if d_stop in idx.stop_node else None
        if route is not None:
            egress_cands.append((d_stop, route))
        d_seq_by_stop[d_stop] = {t["trip_id"]: t["seq_idx"] for t in idx.trips_by_stop.get(d_stop, [])}

    def _egress(sid: str):
        nid = idx.stop_node.get(sid)
        if nid is None:
            return None
        return idx.walk_route(nid, dest)

    def _dist_to_dst(sid: str) -> float:
        sx, sy = idx.stops[sid]
        return math.hypot(sx - x1, sy - y1)

    def _vehicle_km(trip_id: str, seq_a: int, seq_b: int) -> float:
        stops_sorted = idx.trip_stops_sorted.get(trip_id)
        if not stops_sorted:
            return 0.0
        seg = [(s, sid) for (s, sid, a, d) in stops_sorted if seq_a <= s <= seq_b]
        total = 0.0
        for (_sa, sid_a), (_sb, sid_b) in zip(seg, seg[1:]):
            total += idx.euclid_m(idx.stops[sid_a], idx.stops[sid_b])
        return total / 1000.0

    def _downstream(trip_id: str, seq: int, k: int = 12) -> list[tuple[str, int, int, int]]:
        stops_sorted = idx.trip_stops_sorted.get(trip_id)
        if not stops_sorted:
            return []
        out = [(sid, s, a, d) for (s, sid, a, d) in stops_sorted if s > seq]
        return out[:k]

    def _trips_in_window(stop_id: str, lo: float, hi: float) -> list[dict]:
        arr = idx.stop_sorted.get(stop_id, [])
        out = []
        for t in arr:
            if t["dep_sec"] > hi:
                break
            if t["dep_sec"] >= lo:
                out.append(t)
        return out

    def _features(o_stop, access_min, d_stop, egress_min, t1, t2=None) -> dict:
        if t2 is None:
            in_veh = (t1["_alight_arr"] - t1["dep_sec"]) / 60.0
            transfer_min = 0.0
            veh_km = _vehicle_km(t1["trip_id"], t1["seq_idx"], t1["_alight_seq"])
            final_arr = t1["_alight_arr"]
        else:
            in_veh = ((t1["_x_arr"] - t1["dep_sec"]) + (t2["_alight_arr"] - t2["dep_sec"])) / 60.0
            transfer_min = (t2["dep_sec"] - t1["_x_arr"]) / 60.0  # alight t1 -> board t2 at x
            veh_km = (_vehicle_km(t1["trip_id"], t1["seq_idx"], t1["_x_seq"])
                      + _vehicle_km(t2["trip_id"], t2["seq_idx"], t2["_alight_seq"]))
            final_arr = t2["_alight_arr"]
        wait_min = max(0.0, (t1["dep_sec"] - dep_sec) / 60.0 - access_min)
        d2d = (final_arr - dep_sec) / 60.0 + egress_min
        od_m = idx.euclid_m((x0, y0), (x1, y1))
        coverage = min(1.0, max(0.0, (veh_km * 1000.0) / od_m)) if od_m > 1e-9 else 0.0
        return {
            "pt_feasible": 1.0,
            "fallback_reason": None,
            "access_time_min": round(access_min, 3),
            "egress_time_min": round(egress_min, 3),
            "wait_time_min": round(wait_min, 3),
            "in_vehicle_time_min": round(in_veh, 3),
            "transfer_time_min": round(transfer_min, 3),
            "transfer_count": 0 if t2 is None else 1,
            "door_to_door_min": round(d2d, 3),
            "coverage_ratio": round(coverage, 4),
        }

    def _board_lo(access_min: float) -> float:
        # boarding window is access-aware: the traveler must be able to reach
        # the stop before the vehicle departs (buffer 300 s or access+60 s).
        return dep_sec + max(buffer_s, access_min * 60.0 + 60.0)

    any_service_later = False
    dep_hi = dep_sec + 3600.0

    # ---- direct (incl. extended walk egress <= 1.5 km) ----
    for o_stop, (access_path, access_min) in access_cands:
        lo = _board_lo(access_min)
        o_trip_map = idx.stop_trip_map.get(o_stop, {})
        later = _trips_in_window(o_stop, lo, float("inf"))
        if later:
            any_service_later = True
        # exact direct: a trip departing o_stop in the window that serves d_stop
        best_direct = None  # (dep_sec, d_stop, egress_min, t, alight_seq, alight_arr)
        for d_stop, (egress_path, egress_min) in egress_cands:
            for trip_id, d_seq_idx in d_seq_by_stop.get(d_stop, {}).items():
                e = o_trip_map.get(trip_id)
                if e is None or e["seq_idx"] >= d_seq_idx or e["dep_sec"] < lo:
                    continue
                d_arr = idx.trip_seq[trip_id][d_stop][1]
                if best_direct is None or e["dep_sec"] < best_direct[0]:
                    best_direct = (e["dep_sec"], d_stop, egress_min, e, d_seq_idx, d_arr)
        if best_direct is not None:
            _dep, d_stop, egress_min, t, alight_seq, alight_arr = best_direct
            tc = dict(t)
            tc["_alight_seq"] = alight_seq
            tc["_alight_arr"] = alight_arr
            return _features(o_stop, access_min, d_stop, egress_min, tc)
        # extended egress: downstream stop of a later trip within 1.5 km
        for t in later:
            for sid, s, a, dd in _downstream(t["trip_id"], t["seq_idx"], 24):
                if _dist_to_dst(sid) > 1500.0:
                    continue
                route = _egress(sid)
                if route is None:
                    continue
                tc = dict(t)
                tc["_alight_seq"] = s
                tc["_alight_arr"] = a
                return _features(o_stop, access_min, sid, route[1], tc)

    # ---- one transfer (>=180 s connection, board within 1 h) ----
    best = None
    for o_stop, (access_path, access_min) in access_cands:
        lo = _board_lo(access_min)
        for t1 in _trips_in_window(o_stop, lo, dep_hi):
            for x_stop, _xs, x_arr, _xd in _downstream(t1["trip_id"], t1["seq_idx"], 12):
                for t2 in _trips_in_window(x_stop, x_arr + 180.0, x_arr + 2700.0):
                    if t2["trip_id"] == t1["trip_id"]:
                        continue
                    seq2 = idx.trip_seq.get(t2["trip_id"], {})
                    for d_stop, (egress_path, egress_min) in egress_cands:
                        d2 = seq2.get(d_stop)
                        if d2 is None or d2[0] <= t2["seq_idx"]:
                            continue
                        arr_final = d2[1]
                        if best is None or arr_final < best[0]:
                            t1c = dict(t1)
                            t1c["_x_seq"] = _xs
                            t1c["_x_arr"] = x_arr
                            t2c = dict(t2)
                            t2c["_x_stop"] = x_stop
                            t2c["_alight_seq"] = d2[0]
                            t2c["_alight_arr"] = d2[1]
                            best = (arr_final, (o_stop, access_min, d_stop, egress_min, t1c, t2c))
                    for sid, (s, a, dd) in seq2.items():
                        if s <= t2["seq_idx"] or _dist_to_dst(sid) > 1500.0:
                            continue
                        route = _egress(sid)
                        if route is None:
                            continue
                        arr_final = a
                        if best is None or arr_final < best[0]:
                            t1c = dict(t1)
                            t1c["_x_seq"] = _xs
                            t1c["_x_arr"] = x_arr
                            t2c = dict(t2)
                            t2c["_x_stop"] = x_stop
                            t2c["_alight_seq"] = s
                            t2c["_alight_arr"] = a
                            best = (arr_final, (o_stop, access_min, sid, route[1], t1c, t2c))
    if best is not None:
        o_stop, access_min, d_stop, egress_min, t1c, t2c = best[1]
        return _features(o_stop, access_min, d_stop, egress_min, t1c, t2c)

    if not any_service_later:
        return _infeasible("no_service_window")
    return _infeasible("no_direct_or_transfer")


def _infeasible(reason: str) -> dict:
    d = {
        "pt_feasible": 0.0,
        "fallback_reason": reason,
        "access_time_min": 0.0,
        "egress_time_min": 0.0,
        "wait_time_min": 0.0,
        "in_vehicle_time_min": 0.0,
        "transfer_time_min": 0.0,
        "transfer_count": 0,
        "door_to_door_min": 0.0,
        "coverage_ratio": 0.0,
    }
    return d


def classify_accessibility(acc: dict) -> str:
    """Map an accessibility vector to a Class A-E label (S8 instructions §9)."""
    if acc["pt_feasible"] == 0.0:
        return CLASS_E
    a, e = acc["access_time_min"], acc["egress_time_min"]
    cov, d2d, tr = acc["coverage_ratio"], acc["door_to_door_min"], acc["transfer_count"]
    if tr <= 1 and a <= 5.0 and e <= 5.0 and cov >= 0.85 and d2d <= 45.0:
        return CLASS_A
    if tr <= 1 and a <= 8.0 and e <= 8.0 and cov >= 0.7:
        return CLASS_B
    if tr <= 2 and a <= 12.0 and e <= 12.0 and cov >= 0.5:
        return CLASS_C
    return CLASS_D
