"""MATSim adapter: distilled student decisions -> MATSim scenario files.

v0.1 scope (Phase 8 gate):

- loads a trained student checkpoint (model + fitted FeatureExtractor),
- builds a synthetic grid network,
- for each persona+trip, asks the student for its behavioral response under a
  given dynamic context and writes a MATSim daily plan (home -> activity ->
  home) whose outbound leg uses the student's chosen mode and departure shift,
- writes population.xml (with persona attributes) and a config.xml where
  walk/bike/pt run as teleported modes and car runs on the network,
- MATSim runs with ``lastIteration=0`` so its own replanning does NOT overwrite
  the distilled decisions (the student's plan is executed as-is).

Not in v0.1 (documented honestly): return-leg modeling (return uses the same
mode as the outbound leg), transit schedules (pt is teleported), within-day
adaptation.
"""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx
import torch

from ..schemas.context import DynamicContext
from ..schemas.state import UniversalTravelerState
from ..generators.alternative_generator import AlternativeGenerator
from ..singapore.osm_network import load_network_graph
from ..student import FeatureExtractor, TravelerStudent
from ..student.dataset import collate_batch

DEFAULT_MODE_SPEEDS = {
    # meters per second, teleported beeline speeds
    "walk": 1.39,   # ~5 km/h
    "bike": 3.9,    # ~14 km/h
    "pt": 5.0,      # ~18 km/h
}

_ACTIVITY_DURATIONS_MIN = {
    "work": 8 * 60,
    "school": 6 * 60,
    "shop": 2 * 60,
    "leisure": 3 * 60,
    "healthcare": 1 * 60,
    "other": 2 * 60,
}

_LEG_MIN = 30  # placeholder travel time for activity end times (teleported)


def _to_hms(minutes: float) -> str:
    total = max(0, int(round(minutes * 60)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _activity_link(g: nx.DiGraph, node: str, mode: str | None = None) -> str:
    """A link id usable as an activity location at ``node``.

    MATSim convention: an activity sits at the END of its link, and a leg
    starts there — so the link must END at the activity node (an INCOMING
    link). Prefers a real road link (not ``busr_*``/``ai_*``) that allows
    ``mode``; falls back to outgoing links.
    """
    def _ok(lid, d):
        return not lid.startswith(("busr_", "ai_")) and (mode is None or mode in d.get("modes", ()))

    preds = sorted((d.get("id", ""), p, d) for p, d in g.pred[node].items())
    for lid, p, d in preds:
        if _ok(lid, d):
            return lid
    succs = sorted((d.get("id", ""), s, d) for s, d in g.succ[node].items())
    for lid, s, d in succs:
        if _ok(lid, d):
            return lid
    for lid, p, d in preds:
        if not lid.startswith(("busr_", "ai_")):
            return lid
    if preds:
        return preds[0][0]
    if succs:
        return succs[0][0]
    raise ValueError(f"isolated node {node!r}: no link for activity")


class MATSimAdapter:
    """Student checkpoint -> MATSim network + population + config."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
        if "extractor_state" not in ckpt:
            raise ValueError(
                "checkpoint lacks extractor_state (trained with an older script); "
                "retrain or re-save with the current train scripts"
            )
        self.extractor = FeatureExtractor.from_state_dict(ckpt["extractor_state"])
        self.s_cfg = ckpt["config"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TravelerStudent(self.s_cfg, ckpt["feature_spec"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    # ------------------------------------------------------------- decisions
    @torch.no_grad()
    def decide(self, state: UniversalTravelerState) -> dict:
        """Student's behavioral response for one state."""
        feats = self.extractor.encode(state)
        batch = collate_batch([{
            "global_cat": torch.tensor(feats["global_cat"], dtype=torch.long),
            "global_num": torch.tensor(feats["global_num"], dtype=torch.float32),
            "alt_mode_idx": torch.tensor(feats["alt_mode_idx"], dtype=torch.long),
            "alt_num": torch.tensor(feats["alt_num"], dtype=torch.float32),
            "alt_mask": torch.tensor(feats["alt_available"], dtype=torch.float32),
        }])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(batch)
        probs_np = out["mode_probabilities"][0].cpu().numpy()
        probs = {
            alt.mode: float(probs_np[i])
            for i, alt in enumerate(state.alternatives)
            if alt.available
        }
        chosen = max(probs, key=probs.get)
        shift = float(out["departure_time_shift_min"][0].item())
        return {
            "mode": chosen,
            "mode_probabilities": probs,
            "departure_time_shift_min": shift,
        }

    # ---------------------------------------------------------------- network
    @staticmethod
    def build_grid_network(n: int, spacing_m: float, capacity: float = 2000.0) -> ET.Element:
        """n x n grid of nodes with bidirectional links (MATSim network root).

        ``capacity`` is the per-link hourly capacity (veh/h); small values make
        congestion emerge at modest population sizes (feedback-loop demos).
        """
        root = ET.Element("network", name="synthetic-grid")
        nodes = ET.SubElement(root, "nodes")
        for i in range(n):
            for j in range(n):
                ET.SubElement(
                    nodes, "node",
                    id=f"n_{i}_{j}",
                    x=str(round(i * spacing_m, 1)),
                    y=str(round(j * spacing_m, 1)),
                )
        links = ET.SubElement(root, "links")
        for i in range(n):
            for j in range(n):
                for di, dj, d in ((1, 0, "e"), (0, 1, "n")):
                    ni, nj = i + di, j + dj
                    if ni >= n or nj >= n:
                        continue
                    ET.SubElement(
                        links, "link",
                        id=f"l_{i}_{j}_{d}f",
                        **{"from": f"n_{i}_{j}", "to": f"n_{ni}_{nj}",
                           "length": str(spacing_m), "freespeed": "13.89",
                           "capacity": str(capacity), "permlanes": "1.0"},
                    )
                    ET.SubElement(
                        links, "link",
                        id=f"l_{i}_{j}_{d}r",
                        **{"from": f"n_{ni}_{nj}", "to": f"n_{i}_{j}",
                           "length": str(spacing_m), "freespeed": "13.89",
                           "capacity": str(capacity), "permlanes": "1.0"},
                    )
        return root

    @staticmethod
    def _snap(x: float, y: float, spacing_m: float, n: int) -> tuple[str, int, int, float, float]:
        i = max(0, min(n - 1, int(round(x / spacing_m))))
        j = max(0, min(n - 1, int(round(y / spacing_m))))
        return f"n_{i}_{j}", i, j, round(i * spacing_m, 1), round(j * spacing_m, 1)

    @staticmethod
    def _outgoing_link(i: int, j: int, n: int) -> str:
        """A link id leaving node (i, j) that is guaranteed to exist.

        MATSim activities must reference LINK ids (not node ids); cars park on
        the first link of their route. Reverse links are named after the
        FORWARD link's origin node, so the west-bound link leaving (i, j) is
        ``l_{i-1}_{j}_er``, not ``l_{i}_{j}_er``.
        """
        if i < n - 1:
            return f"l_{i}_{j}_ef"
        if j < n - 1:
            return f"l_{i}_{j}_nf"
        if i > 0:
            return f"l_{i-1}_{j}_er"  # reverse of the east link from (i-1, j)
        return f"l_{i}_{j}_nr"

    @staticmethod
    def _home_xy(persona_id: str, spacing_m: float, n: int) -> tuple[float, float]:
        h = int(hashlib.sha256(persona_id.encode()).hexdigest(), 16)
        x = (h % (n * 100)) / 100 * spacing_m * 0.9 + 0.05 * spacing_m
        y = ((h // 1000) % (n * 100)) / 100 * spacing_m * 0.9 + 0.05 * spacing_m
        return x, y

    # -------------------------------------------------------------- scenario
    def build_scenario(
        self,
        personas: list,
        trips: list,
        context: DynamicContext,
        output_dir: str | Path,
        grid_n: int = 20,
        spacing_m: float = 1000.0,
        mode_speeds: dict | None = None,
        trips_per_persona: list[list] | None = None,
        link_capacity: float = 2000.0,
    ) -> dict:
        """Generate network.xml, population.xml, config.xml under output_dir.

        ``trips`` is the trip list used for every persona. When
        ``trips_per_persona`` is given, persona ``i`` instead uses
        ``trips_per_persona[i]`` (each persona has its own trips).

        Returns a manifest with per-person decisions for analysis.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        speeds = {**DEFAULT_MODE_SPEEDS, **(mode_speeds or {})}
        alt_gen = AlternativeGenerator({})

        network_root = self.build_grid_network(grid_n, spacing_m, capacity=link_capacity)

        pop_root = ET.Element("population")
        manifest = []
        for p_idx, persona in enumerate(personas):
            person_trips = trips if trips_per_persona is None else trips_per_persona[p_idx]
            home_x, home_y = self._home_xy(persona.persona_id, spacing_m, grid_n)
            home_node, home_i, home_j, home_sx, home_sy = self._snap(
                home_x, home_y, spacing_m, grid_n
            )
            home_link = self._outgoing_link(home_i, home_j, grid_n)

            person = ET.SubElement(pop_root, "person", id=persona.persona_id)
            attrs = ET.SubElement(person, "attributes")
            for key in (
                "age_group", "income_group", "occupation", "household_size",
                "has_children", "car_ownership", "driving_license",
                "bike_ownership", "transit_pass", "habitual_mode",
                "schedule_flexibility", "mobility_limitation",
            ):
                # population_v6 format: value is TEXT CONTENT, class attribute
                # selects the Java type (NOT type="...").
                el = ET.SubElement(
                    attrs, "attribute", name=key, **{"class": "java.lang.String"}
                )
                el.text = str(getattr(persona, key))

            plan = ET.SubElement(person, "plan", selected="yes")
            for trip in person_trips:
                state = UniversalTravelerState(
                    persona=persona, trip=trip, context=context,
                    alternatives=alt_gen.generate(persona, trip, context),
                )
                decision = self.decide(state)
                dep_min = trip.desired_departure_min + decision["departure_time_shift_min"]

                dest_x = home_x + trip.distance_km * 1000.0
                dest_y = home_y
                dest_node, dest_i, dest_j, dest_sx, dest_sy = self._snap(
                    dest_x, dest_y, spacing_m, grid_n
                )
                dest_link = self._outgoing_link(dest_i, dest_j, grid_n)
                dur_min = _ACTIVITY_DURATIONS_MIN.get(trip.destination_type, 2 * 60)

                ET.SubElement(
                    plan, "activity", type="home", link=home_link,
                    x=str(home_sx), y=str(home_sy), z="0.0", end_time=_to_hms(dep_min),
                )
                ET.SubElement(plan, "leg", mode=decision["mode"])
                ET.SubElement(
                    plan, "activity", type=trip.destination_type, link=dest_link,
                    x=str(dest_sx), y=str(dest_sy), z="0.0",
                    end_time=_to_hms(dep_min + _LEG_MIN + dur_min),
                )
                ET.SubElement(plan, "leg", mode=decision["mode"])  # return, same mode
                # after the return leg the traveler is back home

                manifest.append(
                    {
                        "persona_id": persona.persona_id,
                        "trip_id": trip.trip_id,
                        "purpose": trip.purpose,
                        "destination_type": trip.destination_type,
                        "distance_km": trip.distance_km,
                        "student_mode": decision["mode"],
                        "student_mode_probabilities": decision["mode_probabilities"],
                        "departure_shift_min": round(decision["departure_time_shift_min"], 2),
                        "departure_min": round(dep_min, 2),
                        "home_link": home_link,
                        "dest_link": dest_link,
                    }
                )

            ET.SubElement(
                plan, "activity", type="home", link=home_link,
                x=str(home_sx), y=str(home_sy), z="0.0",
            )

        ET.ElementTree(network_root).write(
            out / "network.xml", encoding="utf-8", xml_declaration=True
        )
        ET.ElementTree(pop_root).write(
            out / "population.xml", encoding="utf-8", xml_declaration=True
        )
        self._insert_doctype(
            out / "network.xml",
            '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">',
        )
        self._insert_doctype(
            out / "population.xml",
            '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">',
        )
        self._write_config(out / "config.xml", speeds)

        (out / "adapter_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    # ------------------------------------------------------- real network (Phase A)
    def build_real_scenario(
        self,
        personas: list,
        trips: list,
        context: DynamicContext,
        output_dir: str | Path,
        network_path: str | Path,
        schedule_path: str | Path,
        vehicles_path: str | Path,
        stops_path: str | Path,
        snap_report_path: str | Path,
        trips_by_stop_path: str | Path,
        activity_nodes_path: str | Path,
        access_max_m: float = 700.0,
        boarding_buffer_s: float = 300.0,
        walk_speed_ms: float = 1.2,
        trips_per_persona: list[list] | None = None,
        flow_capacity_factor: float | None = None,
        storage_capacity_factor: float | None = None,
    ) -> dict:
        """Real-network scenario: student decisions -> population/config on the
        Singapore supply (OSM network + scheduled PT).

        All leg routes are computed HERE and written explicitly into the plans
        (car/bike/walk as link sequences, pt as a ``default_pt`` route
        description + access/egress walk legs), so the run needs no router
        binding beyond the transit QSim engine. PT legs use DIRECT trips
        (no transfers) found via the trips_by_stop index; when no direct trip
        exists the leg falls back to walk and the fallback is recorded.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        alt_gen = AlternativeGenerator({})

        # ---- supply graphs ----
        g, xy = load_network_graph(network_path)
        tt_graphs = {}
        for mode in ("car", "bike", "walk"):
            sg = nx.DiGraph()
            for u, v, d in g.edges(data=True):
                if mode in d["modes"]:
                    sg.add_edge(u, v, tt=d["length"] / max(d["free"], 0.1), id=d.get("id", f"{u}_{v}"))
            for n, (x, y) in xy.items():
                sg.add_node(n, x=x, y=y)
            tt_graphs[mode] = sg

        # shortest-path cache: per-(graph, src, dst) with a hard cap. Leg
        # itineraries repeat massively across agents (activity nodes are
        # shared), which makes large-N scenario builds feasible.
        _base_shortest = self._shortest
        _sp_cache: dict[tuple, list[str] | None] = {}
        _SP_CACHE_CAP = 300_000

        def _shortest_cached(sg: nx.DiGraph, src: str, dst: str) -> list[str] | None:
            key = (id(sg), src, dst)
            if key in _sp_cache:
                return _sp_cache[key]
            res = _base_shortest(sg, src, dst)
            if len(_sp_cache) >= _SP_CACHE_CAP:
                _sp_cache.clear()
            _sp_cache[key] = res
            return res

        self._shortest = _shortest_cached  # instance-level shadow of the staticmethod

        stops = {r["stop_id"]: (r["x"], r["y"]) for r in _read_jsonl(stops_path)}
        snap_rows = json.loads(Path(snap_report_path).read_text(encoding="utf-8"))["per_stop"]
        stop_node = {r["stop_id"]: r["nearest_node"] for r in snap_rows}
        trips_by_stop = json.loads(Path(trips_by_stop_path).read_text(encoding="utf-8"))
        activity_nodes = json.loads(Path(activity_nodes_path).read_text(encoding="utf-8"))
        # trip -> {stop: (seq_idx, arr_sec, dep_sec)} for transfer search
        trip_seq: dict[str, dict[str, tuple[int, int, int]]] = {}
        for sid, entries in trips_by_stop.items():
            for t in entries:
                trip_seq.setdefault(t["trip_id"], {})[sid] = (t["seq_idx"], t["arr_sec"], t["dep_sec"])
        # activity/home nodes restricted to car-accessible road nodes, so
        # car legs always have a legal departure/arrival link (walk-only
        # footway nodes would otherwise force car->walk fallbacks)
        car_nodes = set()
        for u, v, d in g.edges(data=True):
            if "car" in d["modes"]:
                car_nodes.add(u)
                car_nodes.add(v)
        activity_nodes = [a for a in activity_nodes if a["node"] in car_nodes]
        # safety net: only nodes with an incident edge in the walk graph AND in
        # its LARGEST connected component (so every pair of activity nodes has a
        # walk path; leg routes can never be "no path" inside this set)
        walk_g = tt_graphs["walk"]
        walk_und = nx.Graph()
        for u, v in walk_g.edges():
            walk_und.add_edge(u, v)
        walk_largest = max(nx.connected_components(walk_und), key=len)
        activity_nodes = [a for a in activity_nodes if a["node"] in walk_largest]
        self._access_max_m = access_max_m

        pop_root = ET.Element("population")
        manifest = []
        fallback_counts = {"pt_fallback_walk": 0, "no_route_car": 0, "no_route_bike": 0}

        for p_idx, persona in enumerate(personas):
            h = int(hashlib.sha256(persona.persona_id.encode()).hexdigest(), 16)
            home = activity_nodes[h % len(activity_nodes)]["node"]
            home_x, home_y = xy[home]

            person = ET.SubElement(pop_root, "person", id=persona.persona_id)
            attrs = ET.SubElement(person, "attributes")
            for key in (
                "age_group", "income_group", "occupation", "household_size",
                "has_children", "car_ownership", "driving_license",
                "bike_ownership", "transit_pass", "habitual_mode",
                "schedule_flexibility", "mobility_limitation",
            ):
                el = ET.SubElement(attrs, "attribute", name=key, **{"class": "java.lang.String"})
                el.text = str(getattr(persona, key))

            plan = ET.SubElement(person, "plan", selected="yes")
            person_trips = trips if trips_per_persona is None else trips_per_persona[p_idx]
            last_home_link = None
            for t_idx, trip in enumerate(person_trips):
                state = UniversalTravelerState(
                    persona=persona, trip=trip, context=context,
                    alternatives=alt_gen.generate(persona, trip, context),
                )
                decision = self.decide(state)
                dep_min = trip.desired_departure_min + decision["departure_time_shift_min"]
                dest = activity_nodes[(h + 7919 * (t_idx + 1)) % len(activity_nodes)]["node"]
                dest_x, dest_y = xy[dest]
                dur_min = _ACTIVITY_DURATIONS_MIN.get(trip.destination_type, 2 * 60)

                # build both leg chains first: the activity links are DERIVED
                # from the chains (origin activity link starts the outbound
                # route; the destination activity link ends it, per MATSim's
                # leg-route convention)
                home_link = _activity_link(g, home, decision["mode"] if decision["mode"] != "pt" else "walk")
                legs_out, info_out, mode_out, dest_link = self._build_legs(
                    decision["mode"], home, dest, dep_min * 60.0, home_link,
                    trips_by_stop, stop_node, stops, tt_graphs, boarding_buffer_s, fallback_counts, trip_seq,
                )
                ret_dep_s = (dep_min + _LEG_MIN + dur_min) * 60.0
                legs_ret, info_ret, mode_ret, home_link_final = self._build_legs(
                    mode_out, dest, home, ret_dep_s, dest_link,
                    trips_by_stop, stop_node, stops, tt_graphs, boarding_buffer_s, fallback_counts, trip_seq,
                )
                last_home_link = home_link_final

                ET.SubElement(plan, "activity", type="home", link=home_link,
                              x=f"{home_x:.2f}", y=f"{home_y:.2f}", z="0.0", end_time=_to_hms(dep_min))
                for lmode, rtype, rtext, extra in legs_out:
                    leg_el = ET.SubElement(plan, "leg", mode=lmode)
                    if rtext:
                        ET.SubElement(leg_el, "route", type=rtype, **extra).text = rtext
                ET.SubElement(plan, "activity", type=trip.destination_type, link=dest_link,
                              x=f"{dest_x:.2f}", y=f"{dest_y:.2f}", z="0.0",
                              end_time=_to_hms(dep_min + _LEG_MIN + dur_min))
                for lmode, rtype, rtext, extra in legs_ret:
                    leg_el = ET.SubElement(plan, "leg", mode=lmode)
                    if rtext:
                        ET.SubElement(leg_el, "route", type=rtype, **extra).text = rtext

                manifest.append({
                    "persona_id": persona.persona_id, "trip_id": trip.trip_id,
                    "student_mode": decision["mode"],
                    "outbound_mode": mode_out, "return_mode": mode_ret,
                    "departure_shift_min": round(decision["departure_time_shift_min"], 2),
                    "departure_min": round(dep_min, 2),
                    "home_node": home, "dest_node": dest,
                    "outbound": info_out, "return": info_ret,
                })

            ET.SubElement(plan, "activity", type="home", link=last_home_link,
                          x=f"{home_x:.2f}", y=f"{home_y:.2f}", z="0.0")

        ET.ElementTree(pop_root).write(out / "population.xml", encoding="utf-8", xml_declaration=True)
        self._insert_doctype(out / "population.xml",
                             '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">')
        self._write_real_config(out / "config.xml", network_path, schedule_path, vehicles_path,
                                flow_capacity_factor, storage_capacity_factor)
        (out / "adapter_manifest.json").write_text(
            json.dumps({"decisions": manifest, "fallback_counts": fallback_counts}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return manifest

    @staticmethod
    def _shortest(sg: nx.DiGraph, src: str, dst: str) -> list[str] | None:
        """Bidirectional Dijkstra on a mode graph; returns MATSim link ids.

        Link ids come from the graph edge attribute ``id`` (never rebuilt from
        node ids, which would be wrong for busr_/ai_ artificial links).
        """
        try:
            _len, path = nx.bidirectional_dijkstra(sg, src, dst, weight="tt")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return [sg.edges[path[i], path[i + 1]]["id"] for i in range(len(path) - 1)]

    def _build_legs(
        self, mode: str, src: str, dst: str, dep_sec: float, start_link: str,
        trips_by_stop: dict, stop_node: dict, stops: dict,
        tt_graphs: dict, buffer_s: float, fallback_counts: dict, trip_seq: dict,
    ) -> tuple[list[tuple[str, str, str, dict]], dict, str, str]:
        """Legs for one trip leg from src to dst.

        MATSim convention: a leg's route must START with the link the agent is
        currently on (the origin activity's link). Returns
        ``(legs, info, executed_mode, end_link)`` where ``end_link`` is the
        last network link of the chain (the destination activity's link).
        """
        if mode in ("car", "bike"):
            route = self._shortest(tt_graphs[mode], src, dst)
            if route is not None:
                full = [start_link] + route
                return [(mode, "links", " ".join(full), {})], \
                    {"mode": mode, "route_links": len(full)}, mode, full[-1]
            fallback_counts[f"no_route_{mode}"] += 1
            mode = "walk"
        if mode == "pt":
            legs, info, end_link, reason = self._plan_pt(src, dst, dep_sec, start_link, trips_by_stop, stop_node, stops, tt_graphs, buffer_s, trip_seq)
            if legs is not None:
                return legs, info, "pt", end_link
            fallback_counts["pt_fallback_walk"] = fallback_counts.get("pt_fallback_walk", 0) + 1
            if reason:
                key = f"pt_fallback_{reason}"
                fallback_counts[key] = fallback_counts.get(key, 0) + 1
            route = self._shortest(tt_graphs["walk"], src, dst) or []
            full = [start_link] + route
            return [("walk", "links", " ".join(full), {})], \
                {"mode": "walk", "pt_fallback": True, "pt_reason": reason, "route_links": len(full)}, "walk", full[-1]
        # walk (or anything else) -> walk
        route = self._shortest(tt_graphs["walk"], src, dst) or []
        full = [start_link] + route
        return [("walk", "links", " ".join(full), {})], {"mode": "walk", "route_links": len(full)}, "walk", full[-1]

    def _plan_pt(
        self, src: str, dst: str, dep_sec: float, start_link: str,
        trips_by_stop: dict, stop_node: dict, stops: dict,
        tt_graphs: dict, buffer_s: float, trip_seq: dict,
    ) -> tuple[list[tuple[str, str, str, dict]] | None, dict, str | None, str]:
        """PT plan: direct trip first, then one transfer.

        Transfers are represented as SEPARATE pt legs joined by a short
        transfer-walk leg (NOT a chainedRoute: the chained relocation
        machinery requires umlauf vehicle chaining, which single-departure
        vehicles do not have — see docs/MATSIM_2026_REAL_SUPPLY_INTEGRATION.md
        §5). Returns ``(legs, info, end_link, fallback_reason)``.
        """
        def _near(node: str, k: int = 12, radius: float | None = None) -> list[str]:
            r = radius if radius is not None else self._access_max_m
            x0, y0 = tt_graphs["walk"].nodes[node]["x"], tt_graphs["walk"].nodes[node]["y"]
            out_list = []
            for sid, (sx, sy) in stops.items():
                d = math.hypot(sx - x0, sy - y0)
                if d <= r:
                    out_list.append((d, sid))
            out_list.sort()
            return [sid for _, sid in out_list[:k]]

        access_cands = []
        for o_stop in _near(src, 12, self._access_max_m):
            o_node = stop_node.get(o_stop)
            if o_node is None:
                continue
            route = self._shortest(tt_graphs["walk"], src, o_node)
            if route is not None:
                access_cands.append((o_stop, route))
        if not access_cands:  # wider access fallback (1.5 km)
            for o_stop in _near(src, 12, 1500.0):
                o_node = stop_node.get(o_stop)
                if o_node is None:
                    continue
                route = self._shortest(tt_graphs["walk"], src, o_node)
                if route is not None:
                    access_cands.append((o_stop, route))
        dst_x = tt_graphs["walk"].nodes[dst]["x"]
        dst_y = tt_graphs["walk"].nodes[dst]["y"]
        egress_cache: dict[str, list[str] | None] = {}

        def _egress(sid: str) -> list[str] | None:
            if sid not in egress_cache:
                nid = stop_node.get(sid)
                egress_cache[sid] = self._shortest(tt_graphs["walk"], nid, dst) if nid else None
            return egress_cache[sid]

        def _dist_to_dst(sid: str) -> float:
            sx, sy = stops[sid]
            return math.hypot(sx - dst_x, sy - dst_y)

        egress_cands = []
        d_seq_by_stop: dict[str, dict[str, int]] = {}
        for d_stop in _near(dst, 12, self._access_max_m):
            route = _egress(d_stop)
            if route is not None:
                egress_cands.append((d_stop, route))
            d_seq_by_stop[d_stop] = {t["trip_id"]: t["seq_idx"] for t in trips_by_stop.get(d_stop, [])}

        if not access_cands:
            return None, {}, None, "no_stops_in_radius"

        def _pt_leg(access_stop, egress_stop, t, boarding_sec):
            desc = json.dumps({
                "accessFacilityId": access_stop, "egressFacilityId": egress_stop,
                "transitLineId": t["route_id"], "transitRouteId": t["trip_id"],
                "boardingTime": _to_hms(boarding_sec / 60.0),
            })
            return ("pt", "default_pt", desc,
                    {"start_link": f"ai_in_{access_stop}", "end_link": f"ai_in_{egress_stop}"})

        def _legs(o_stop, access, d_stop, egress, t1, t2=None):
            info = {"mode": "pt", "board_stop": o_stop, "alight_stop": d_stop,
                    "trip_id": t1["trip_id"], "line": t1["route_id"],
                    "boarding_time": _to_hms(t1["dep_sec"] / 60.0), "transfers": 0 if t2 is None else 1}
            access_route = [start_link] + access + [f"ai_in_{o_stop}"]
            egress_route = [f"ai_in_{d_stop}", f"ai_out_{d_stop}"] + egress
            if t2 is None:
                legs = [
                    ("walk", "links", " ".join(access_route), {}),
                    _pt_leg(o_stop, d_stop, t1, t1["dep_sec"]),
                    ("walk", "links", " ".join(egress_route), {}),
                ]
            else:
                x = t2["_x_stop"]
                transfer_route = [f"ai_in_{x}", f"ai_out_{x}", f"ai_in_{x}"]
                legs = [
                    ("walk", "links", " ".join(access_route), {}),
                    _pt_leg(o_stop, x, t1, t1["dep_sec"]),
                    ("walk", "links", " ".join(transfer_route), {}),
                    _pt_leg(x, d_stop, t2, t2["dep_sec"]),
                    ("walk", "links", " ".join(egress_route), {}),
                ]
                info.update({"transfer_stop": x, "trip_id_2": t2["trip_id"],
                             "line_2": t2["route_id"]})
            return legs, info, egress_route[-1]

        # ---- direct (incl. extended walk egress: alight up to 1.5 km from dest) ----
        dep_lo = dep_sec + buffer_s
        any_service_later = False
        for o_stop, access in access_cands:
            o_trips = trips_by_stop.get(o_stop, [])
            later = [t for t in o_trips if t["dep_sec"] >= dep_lo]
            if later:
                any_service_later = True
            for d_stop, egress in egress_cands:
                d_seq = d_seq_by_stop.get(d_stop, {})
                if not d_seq:
                    continue
                cands = [t for t in o_trips if t["trip_id"] in d_seq
                         and t["seq_idx"] < d_seq[t["trip_id"]]
                         and t["dep_sec"] >= dep_lo]
                if not cands:
                    continue
                t = min(cands, key=lambda x: x["dep_sec"])
                return _legs(o_stop, access, d_stop, egress, t) + ("direct",)
            # extended egress: any downstream stop of an o-trip within 1.5 km
            for t in later:
                seq1 = trip_seq.get(t["trip_id"], {})
                for sid, (s, a, dd) in seq1.items():
                    if s <= t["seq_idx"] or _dist_to_dst(sid) > 1500.0:
                        continue
                    egress = _egress(sid)
                    if egress is None:
                        continue
                    return _legs(o_stop, access, sid, egress, t) + ("direct",)

        # ---- one transfer (separate pt legs; extended walk egress allowed) ----
        best = None  # (final arrival, (legs, info, end_link))
        dep_hi = dep_sec + 3600.0  # board the first vehicle within 1 h
        for o_stop, access in access_cands:
            for t1 in trips_by_stop.get(o_stop, []):
                if not (dep_lo <= t1["dep_sec"] <= dep_hi):
                    continue
                seq1 = trip_seq.get(t1["trip_id"], {})
                downstream = sorted(
                    ((sid, s, a, dd) for sid, (s, a, dd) in seq1.items() if s > t1["seq_idx"]),
                    key=lambda r: r[1],
                )[:12]
                for x_stop, _xs, x_arr, _xd in downstream:
                    for t2 in trips_by_stop.get(x_stop, []):
                        if t2["trip_id"] == t1["trip_id"]:
                            continue
                        if not (x_arr + 180 <= t2["dep_sec"] <= x_arr + 2700):
                            continue
                        seq2 = trip_seq.get(t2["trip_id"], {})
                        for d_stop, egress in egress_cands:
                            d2 = seq2.get(d_stop)
                            if d2 is None or d2[0] <= t2["seq_idx"]:
                                continue
                            arr_final = d2[1]
                            if best is None or arr_final < best[0]:
                                t2c = dict(t2)
                                t2c["_x_stop"] = x_stop
                                best = (arr_final, _legs(o_stop, access, d_stop, egress, t1, t2c))
                        for sid, (s, a, dd) in seq2.items():
                            if s <= t2["seq_idx"] or _dist_to_dst(sid) > 1500.0:
                                continue
                            egress = _egress(sid)
                            if egress is None:
                                continue
                            arr_final = a
                            if best is None or arr_final < best[0]:
                                t2c = dict(t2)
                                t2c["_x_stop"] = x_stop
                                best = (arr_final, _legs(o_stop, access, sid, egress, t1, t2c))
        if best is not None:
            return best[1] + ("transfer",)

        if not any_service_later:
            return None, {}, None, "no_service_window"
        return None, {}, None, "no_direct_or_transfer"

    @staticmethod
    def _write_real_config(path: Path, network_path, schedule_path, vehicles_path,
                           flow_capacity_factor=None, storage_capacity_factor=None) -> None:
        lines = [
            '<?xml version="1.0" ?>',
            '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">',
            "<config>",
            '\t<module name="global">',
            '\t\t<param name="randomSeed" value="4711" />',
            '\t\t<param name="coordinateSystem" value="EPSG:32648" />',
            "\t</module>",
            '\t<module name="network">',
            f'\t\t<param name="inputNetworkFile" value="{network_path}" />',
            "\t</module>",
            '\t<module name="plans">',
            '\t\t<param name="inputPlansFile" value="population.xml" />',
            '\t\t<param name="handlingOfPlansWithoutRoutingMode" value="useMainModeIdentifier" />',
            "\t</module>",
            '\t<module name="transit">',
            '\t\t<param name="useTransit" value="true" />',
            f'\t\t<param name="transitScheduleFile" value="{schedule_path}" />',
            f'\t\t<param name="vehiclesFile" value="{vehicles_path}" />',
            '\t\t<param name="transitModes" value="pt" />',
            "\t</module>",
            '\t<module name="controller">',
            '\t\t<param name="outputDirectory" value="./output" />',
            '\t\t<param name="overwriteFiles" value="deleteDirectoryIfExists" />',
            '\t\t<param name="writePlansInterval" value="0" />',
            '\t\t<param name="firstIteration" value="0" />',
            '\t\t<param name="lastIteration" value="0" />',
            "\t</module>",
            '\t<module name="qsim">',
            '\t\t<param name="endTime" value="30:00:00" />',
            '\t\t<param name="mainMode" value="car,bike,walk" />',
            '\t\t<param name="trafficDynamics" value="queue" />',
        ]
        if flow_capacity_factor is not None:
            lines.append(f'\t\t<param name="flowCapacityFactor" value="{flow_capacity_factor}" />')
        if storage_capacity_factor is not None:
            lines.append(f'\t\t<param name="storageCapacityFactor" value="{storage_capacity_factor}" />')
        lines += [
            "\t</module>",
            '\t<module name="linkStats">',
            '\t\t<param name="writeLinkStatsInterval" value="1" />',
            "\t</module>",
            '\t<module name="routing">',
            '\t\t<param name="networkModes" value="car,bike,walk" />',
            '\t\t<param name="clearDefaultTeleportedModeParams" value="true" />',
            '\t\t<param name="networkRouteConsistencyCheck" value="disable" />',
            '\t\t<param name="routingRandomness" value="0.0" />',
            '\t\t<parameterset type="teleportedModeParameters">',
            '\t\t\t<param name="mode" value="non_network_walk" />',
            '\t\t\t<param name="beelineDistanceFactor" value="1.3" />',
            '\t\t\t<param name="teleportedModeSpeed" value="1.39" />',
            '\t\t</parameterset>',
            "\t</module>",
            '\t<module name="swissRailRaptor">',
            '\t\t<parameterset type="modeMapping">',
            '\t\t\t<param name="routeMode" value="bus" />',
            '\t\t\t<param name="passengerMode" value="pt" />',
            '\t\t</parameterset>',
            '\t\t<parameterset type="modeMapping">',
            '\t\t\t<param name="routeMode" value="rail" />',
            '\t\t\t<param name="passengerMode" value="pt" />',
            '\t\t</parameterset>',
            "\t</module>",
            '\t<module name="scoring">',
            '\t\t<parameterset type="scoringParameters">',
        ]
        for mode in ("car", "walk", "bike", "pt"):
            lines += [f'\t\t\t<parameterset type="modeParams"><param name="mode" value="{mode}" /></parameterset>']
        for act_type, dur in [
            ("home", "12:00:00"), ("work", "08:00:00"), ("school", "06:00:00"),
            ("shop", "02:00:00"), ("leisure", "03:00:00"), ("healthcare", "01:00:00"),
            ("other", "02:00:00"),
        ]:
            lines += [
                '\t\t\t<parameterset type="activityParams">',
                f'\t\t\t\t<param name="activityType" value="{act_type}" />',
                f'\t\t\t\t<param name="typicalDuration" value="{dur}" />',
                "\t\t\t</parameterset>",
            ]
        lines += [
            "\t\t</parameterset>",
            "\t</module>",
            "</config>",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _insert_doctype(path: Path, doctype: str) -> None:
        """Insert a DOCTYPE line after the XML declaration.

        MATSim's XML readers use the DOCTYPE to select the parser version
        (v1/v2 network, population_v5/v6); without it the reader's delegate is
        never set and parsing fails with a NullPointerException.
        """
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        # ElementTree writes: <?xml version='1.0' encoding='utf-8'?> first
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

    @staticmethod
    def _write_config(path: Path, speeds: dict) -> None:
        lines = [
            '<?xml version="1.0" ?>',
            '<!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd">',
            "<config>",
            '\t<module name="global">',
            '\t\t<param name="randomSeed" value="4711" />',
            '\t\t<param name="coordinateSystem" value="Atlantis" />',
            "\t</module>",
            '\t<module name="network">',
            '\t\t<param name="inputNetworkFile" value="network.xml" />',
            "\t</module>",
            '\t<module name="plans">',
            '\t\t<param name="inputPlansFile" value="population.xml" />',
            "\t</module>",
            '\t<module name="controller">',
            '\t\t<param name="outputDirectory" value="./output" />',
            '\t\t<param name="firstIteration" value="0" />',
            '\t\t<param name="lastIteration" value="0" />',
            "\t</module>",
            '\t<module name="linkStats">',
            '\t\t<param name="writeLinkStatsInterval" value="1" />',
            '\t\t<param name="averageLinkStatsOverIterations" value="1" />',
            "\t</module>",
            '\t<module name="qsim">',
            '\t\t<param name="endTime" value="36:00:00" />',
            '\t\t<param name="mainMode" value="car" />',
            '\t\t<param name="trafficDynamics" value="queue" />',
            "\t</module>",
            '\t<module name="routing">',
            '\t\t<param name="networkModes" value="car" />',
        ]
        for mode, speed in sorted(speeds.items()):
            lines += [
                '\t\t<parameterset type="teleportedModeParameters">',
                f'\t\t\t<param name="mode" value="{mode}" />',
                '\t\t\t<param name="beelineDistanceFactor" value="1.3" />',
                f'\t\t\t<param name="teleportedModeSpeed" value="{speed}" />',
                "\t\t</parameterset>",
            ]
        lines += [
            "\t</module>",
            '\t<module name="scoring">',
            '\t\t<parameterset type="scoringParameters">',
        ]
        for mode in ("car", "walk", "bike", "pt"):
            lines += [
                f'\t\t\t<parameterset type="modeParams"><param name="mode" value="{mode}" /></parameterset>',
            ]
        # every activity type that can appear in the population must have
        # utility parameters, otherwise scoring aborts at runtime
        for act_type, dur in [
            ("home", "12:00:00"), ("work", "08:00:00"), ("school", "06:00:00"),
            ("shop", "02:00:00"), ("leisure", "03:00:00"), ("healthcare", "01:00:00"),
            ("other", "02:00:00"),
        ]:
            lines += [
                '\t\t\t<parameterset type="activityParams">',
                f'\t\t\t\t<param name="activityType" value="{act_type}" />',
                f'\t\t\t\t<param name="typicalDuration" value="{dur}" />',
                "\t\t\t</parameterset>",
            ]
        lines += [
            "\t\t</parameterset>",
            "\t</module>",
            "</config>",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
