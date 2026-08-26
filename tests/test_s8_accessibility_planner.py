"""Tests for the S8 accessibility planner (synthetic supply)."""
from __future__ import annotations

import networkx as nx

from traveler_distillation.accessibility.gtfs_accessibility import (
    CLASS_A,
    CLASS_E,
    SupplyIndex,
    classify_accessibility,
    plan_accessibility,
)


def _fake_index() -> SupplyIndex:
    """Line network along x: O(0)-N1(600)-N2(2000)-N3(3000)-D(3100)-N5(5000).

    One trip T1: S1->S2->S3; one trip T2: X->S3. Walk speed 1.2 m/s.
    """
    idx = SupplyIndex.__new__(SupplyIndex)
    WALK = 1.2
    xy = {
        "O": (0.0, 0.0), "N1": (600.0, 0.0), "N2": (2000.0, 0.0),
        "N3": (3000.0, 0.0), "D": (3100.0, 0.0), "N5": (5000.0, 0.0),
    }
    idx.xy = xy
    g = nx.DiGraph()
    for a, b in (("O", "N1"), ("N1", "N2"), ("N2", "N3"), ("N3", "D"), ("D", "N5")):
        dist = xy[b][0] - xy[a][0]
        g.add_edge(a, b, tt=dist / WALK)
        g.add_edge(b, a, tt=dist / WALK)
    for n, (x, y) in xy.items():
        g.add_node(n, x=x, y=y)
    idx.tt_graphs = {"walk": g}
    idx.walk_largest = set(xy)
    idx._walk_cache = {}

    idx.stops = {"S1": (600.0, 0.0), "S2": (2000.0, 0.0), "S3": (3000.0, 0.0)}
    idx.stop_node = {"S1": "N1", "S2": "N2", "S3": "N3"}
    t1_dep = 30000.0
    t3_dep = 30600.0  # second departure at S1, 10 min after T1
    idx.trips_by_stop = {
        "S1": [{"trip_id": "T1", "route_id": "R1", "mode": "bus", "seq_idx": 0,
                "dep_sec": t1_dep, "arr_sec": t1_dep - 60},
               {"trip_id": "T3", "route_id": "R1", "mode": "bus", "seq_idx": 0,
                "dep_sec": t3_dep, "arr_sec": t3_dep - 60}],
        "S2": [{"trip_id": "T1", "route_id": "R1", "mode": "bus", "seq_idx": 1,
                "dep_sec": t1_dep + 320, "arr_sec": t1_dep + 300},
               {"trip_id": "T3", "route_id": "R1", "mode": "bus", "seq_idx": 1,
                "dep_sec": t3_dep + 320, "arr_sec": t3_dep + 300}],
        "S3": [{"trip_id": "T1", "route_id": "R1", "mode": "bus", "seq_idx": 2,
                "dep_sec": t1_dep + 620, "arr_sec": t1_dep + 600},
               {"trip_id": "T3", "route_id": "R1", "mode": "bus", "seq_idx": 2,
                "dep_sec": t3_dep + 620, "arr_sec": t3_dep + 600},
               {"trip_id": "T2", "route_id": "R2", "mode": "bus", "seq_idx": 1,
                "dep_sec": t1_dep + 900, "arr_sec": t1_dep + 900}],
    }
    idx.stop_sorted = {s: sorted(v, key=lambda t: t["dep_sec"]) for s, v in idx.trips_by_stop.items()}
    idx.stop_trip_map = {s: {t["trip_id"]: t for t in v} for s, v in idx.trips_by_stop.items()}
    idx.trip_seq = {
        "T1": {"S1": (0, t1_dep - 60, t1_dep), "S2": (1, t1_dep + 300, t1_dep + 320),
               "S3": (2, t1_dep + 600, t1_dep + 620)},
        "T3": {"S1": (0, t3_dep - 60, t3_dep), "S2": (1, t3_dep + 300, t3_dep + 320),
               "S3": (2, t3_dep + 600, t3_dep + 620)},
        "T2": {"S3": (1, t1_dep + 900, t1_dep + 900)},
    }
    idx.trip_stops_sorted = {
        tid: sorted((s, sid, a, d) for sid, (s, a, d) in seq.items())
        for tid, seq in idx.trip_seq.items()
    }
    idx.candidate_nodes = ["O", "N1", "N2", "N3", "D", "N5"]
    idx.band_nodes = {"near": ["O", "D"], "mid": [], "far": ["N5"]}
    idx.activity_nodes = [{"node": n, "x": xy[n][0], "y": xy[n][1]} for n in xy]
    return idx


def test_direct_itinerary_decomposition():
    idx = _fake_index()
    # dep 10 min before T1 departs: access 600m = 8.33 min, wait 1.67 min
    acc = plan_accessibility(idx, "O", "D", 30000.0 - 600.0)
    assert acc["pt_feasible"] == 1.0
    assert acc["transfer_count"] == 0
    assert abs(acc["access_time_min"] - 8.333) < 0.01
    assert abs(acc["wait_time_min"] - 1.667) < 0.01
    assert abs(acc["in_vehicle_time_min"] - 10.0) < 0.01
    assert abs(acc["egress_time_min"] - 1.389) < 0.01
    lhs = (acc["access_time_min"] + acc["egress_time_min"] + acc["wait_time_min"]
           + acc["in_vehicle_time_min"] + acc["transfer_time_min"])
    assert abs(lhs - acc["door_to_door_min"]) < 0.02
    assert 0.0 <= acc["coverage_ratio"] <= 1.0


def test_no_stops_in_radius_is_class_e():
    idx = _fake_index()
    acc = plan_accessibility(idx, "N5", "O", 30000.0)
    assert acc["pt_feasible"] == 0.0
    assert acc["fallback_reason"] == "no_stops_in_radius"
    assert classify_accessibility(acc) == CLASS_E


def test_access_aware_boarding_skips_unreachable_trip():
    idx = _fake_index()
    # dep 5 min before T1: access walk takes 8.33 min, so T1 (t+5min) is NOT
    # reachable; the access-aware window must board T3 (t+15min) instead.
    acc = plan_accessibility(idx, "O", "D", 30000.0 - 300.0)
    assert acc["pt_feasible"] == 1.0
    assert acc["wait_time_min"] >= 1.0 - 1e-6
    # wait = (30600 - 29700)/60 - 8.333 = 6.667
    assert abs(acc["wait_time_min"] - 6.667) < 0.01
    lhs = (acc["access_time_min"] + acc["egress_time_min"] + acc["wait_time_min"]
           + acc["in_vehicle_time_min"] + acc["transfer_time_min"])
    assert abs(lhs - acc["door_to_door_min"]) < 0.02


def test_classification_thresholds():
    base = {"pt_feasible": 1.0, "fallback_reason": None, "access_time_min": 3.0,
            "egress_time_min": 4.0, "wait_time_min": 4.0, "in_vehicle_time_min": 15.0,
            "transfer_time_min": 0.0, "transfer_count": 0,
            "door_to_door_min": 26.0, "coverage_ratio": 0.92}
    assert classify_accessibility(base) == CLASS_A
    base["access_time_min"] = 8.0
    base["coverage_ratio"] = 0.75
    assert classify_accessibility(base) == "B_good"
    base["coverage_ratio"] = 0.5
    assert classify_accessibility(base) == "C_moderate"
    base["pt_feasible"] = 0.0
    assert classify_accessibility(base) == CLASS_E
