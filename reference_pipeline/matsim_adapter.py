"""MATSim adapter (reference): decisions -> population.xml / config.xml, plus
the MATSim Java launch + light event parsing.

Plan construction reuses the PRODUCTION building blocks byte-for-byte:

- OD derivation: production deterministic hash rule (or validated explicit OD);
- alternatives: production ``plan_accessibility`` + ``build_real_alternatives``
  (+ Phase C scenario effects), possibly served from the persistent cache;
- decisions: :class:`StudentAdapter` batch inference (decision-identical to the
  per-state production ``decide()``);
- legs: production ``MATSimAdapter._build_legs`` / ``_plan_pt`` with the same
  in-process SP cache (or the persistent cache);
- XML/config: identical ElementTree calls, ``_insert_doctype`` and
  ``_write_real_config`` from the production adapter — so for the same input
  the emitted population.xml / config.xml / adapter_manifest.json are
  byte-identical to the production path (verified in the validation suite).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from traveler_distillation.matsim.adapter import (
    _ACTIVITY_DURATIONS_MIN,
    _LEG_MIN,
    _activity_link,
    _read_jsonl,
    _to_hms,
)
from traveler_distillation.schemas.context import DynamicContext
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.singapore.osm_network import load_network_graph
from traveler_distillation.student.release_guard import assert_not_frozen_output

from .feature_adapter import derive_od, make_alt_factory
from .route_cache import RouteCache
from .student_adapter import StudentAdapter

_PERSON = re.compile(r"^P\d+$")


def build_supply_view(network_path: str | Path, activity_nodes_path: str | Path):
    """Load the network graph and the production-filtered activity-node pool.

    Mirrors ``MATSimAdapter.build_real_scenario`` exactly: car/bike/walk
    travel-time graphs (car = free-flow, walk 1.34 m/s, bike 4.17 m/s — the S9
    fix), car-accessible node set, walk largest connected component.
    Returns (g, xy, tt_graphs, activity_nodes).
    """
    g, xy = load_network_graph(network_path)
    tt_graphs = {}
    for mode, speed in (("car", None), ("bike", 4.17), ("walk", 1.34)):
        sg = nx.DiGraph()
        for u, v, d in g.edges(data=True):
            if mode in d["modes"]:
                tt = d["length"] / max(d["free"], 0.1) if speed is None else d["length"] / speed
                sg.add_edge(u, v, tt=tt, id=d.get("id", f"{u}_{v}"))
        for n, (x, y) in xy.items():
            sg.add_node(n, x=x, y=y)
        tt_graphs[mode] = sg

    car_nodes = set()
    for u, v, d in g.edges(data=True):
        if "car" in d["modes"]:
            car_nodes.add(u)
            car_nodes.add(v)
    walk_g = tt_graphs["walk"]
    walk_und = nx.Graph()
    walk_und.add_edges_from(walk_g.edges())
    walk_largest = max(nx.connected_components(walk_und), key=len)

    activity_nodes = json.loads(Path(activity_nodes_path).read_text(encoding="utf-8"))
    activity_nodes = [a for a in activity_nodes if a["node"] in car_nodes]
    activity_nodes = [a for a in activity_nodes if a["node"] in walk_largest]
    return g, xy, tt_graphs, activity_nodes


def validate_explicit_od(explicit_od: dict, valid_nodes: set) -> None:
    """Explicit origin/dest node ids must be usable activity nodes."""
    for (pid, tid), (o, d) in explicit_od.items():
        for label, node in (("origin", o), ("dest", d)):
            if node not in valid_nodes:
                raise ValueError(
                    f"explicit {label}_node {node!r} for ({pid}, {tid}) is not a valid "
                    "activity node (must be car-accessible and in the walk largest "
                    "connected component of the network)"
                )


def build_scenario(
    adapter: StudentAdapter,
    personas: list,
    trips_per_persona: list[list],
    context: DynamicContext,
    output_dir: str | Path,
    supply_paths: dict[str, str | Path],
    supply_view,
    batch_size: int = 256,
    cache: RouteCache | None = None,
    flow_capacity_factor: float | None = None,
    storage_capacity_factor: float | None = None,
    explicit_od: dict | None = None,
    encoder=None,
) -> dict:
    """Reference build: batch decisions + cached routing -> population.xml /
    config.xml / adapter_manifest.json (production formats)."""
    out = assert_not_frozen_output(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    explicit_od = explicit_od or {}
    g, xy, tt_graphs, activity_nodes = supply_view
    validate_explicit_od(explicit_od, {a["node"] for a in activity_nodes})

    from traveler_distillation.accessibility.gtfs_accessibility import SupplyIndex
    idx = SupplyIndex(supply_paths)
    alt_factory = make_alt_factory(idx, context, cache=cache)

    # ---- production SP cache wrapper (runtime optimization; routes identical) ----
    base_shortest = adapter._adapter._shortest
    _sp_cache: dict[tuple, list | None] = {}
    _SP_CACHE_CAP = 300_000
    _mode_by_graph = {id(tt_graphs[m]): m for m in tt_graphs}

    def _shortest(sg, src, dst):
        key = (id(sg), src, dst)
        if key in _sp_cache:
            return _sp_cache[key]
        if cache is None:
            res = base_shortest(sg, src, dst)
        else:
            mode = _mode_by_graph.get(id(sg))
            if mode is None:
                res = base_shortest(sg, src, dst)
            else:
                res = cache.shortest_path(mode, src, dst, lambda: base_shortest(sg, src, dst))
        if len(_sp_cache) >= _SP_CACHE_CAP:
            _sp_cache.clear()
        _sp_cache[key] = res
        return res

    adapter._adapter._shortest = _shortest
    adapter._adapter._access_max_m = 700.0

    # ---- states (production order) ----
    t0 = time.perf_counter()
    records = []  # (p_idx, persona, person_trips, home, home_xy)
    states = []
    od_by_state = []  # dest node per state (parallel to states)
    for p_idx, persona in enumerate(personas):
        h = int(hashlib.sha256(persona.persona_id.encode()).hexdigest(), 16)
        home = activity_nodes[h % len(activity_nodes)]["node"]
        home_xy = xy[home]
        person_trips = trips_per_persona[p_idx]
        records.append((p_idx, persona, person_trips, home, home_xy))
        for t_idx, trip in enumerate(person_trips):
            od = explicit_od.get((persona.persona_id, trip.trip_id))
            if od is not None:
                dest = od[1]
            else:
                dest = derive_od(persona.persona_id, t_idx, activity_nodes)[1]
            alternatives = alt_factory(persona, trip, context, home, dest)
            states.append(UniversalTravelerState(
                persona=persona, trip=trip, context=context, alternatives=alternatives,
            ))
            od_by_state.append((home, dest))
    t_feature = time.perf_counter() - t0

    # ---- batch inference ----
    t0 = time.perf_counter()
    decisions = adapter.predict(states, batch_size=batch_size, encoder=encoder)
    t_inference = time.perf_counter() - t0

    # ---- plan construction (production ET structure) ----
    t0 = time.perf_counter()
    pop_root = ET.Element("population")
    manifest = []
    fallback_counts = {"pt_fallback_walk": 0, "no_route_car": 0, "no_route_bike": 0}

    stops = {r["stop_id"]: (r["x"], r["y"]) for r in _read_jsonl(supply_paths["stops"])}
    snap_rows = json.loads(Path(supply_paths["snapping"]).read_text(encoding="utf-8"))["per_stop"]
    stop_node = {r["stop_id"]: r["nearest_node"] for r in snap_rows}
    trips_by_stop = json.loads(Path(supply_paths["trips_by_stop"]).read_text(encoding="utf-8"))
    trip_seq: dict[str, dict[str, tuple[int, int, int]]] = {}
    for sid, entries in trips_by_stop.items():
        for t in entries:
            trip_seq.setdefault(t["trip_id"], {})[sid] = (t["seq_idx"], t["arr_sec"], t["dep_sec"])

    state_iter = iter(zip(states, decisions, od_by_state))
    for p_idx, persona, person_trips, home, home_xy in records:
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
        home_x, home_y = home_xy
        last_home_link = None
        for t_idx, trip in enumerate(person_trips):
            state, decision, (_home_node, dest) = next(state_iter)
            dest_x, dest_y = xy[dest]
            dep_min = trip.desired_departure_min + decision["departure_time_shift_min"]
            dur_min = _ACTIVITY_DURATIONS_MIN.get(trip.destination_type, 2 * 60)

            home_link = _activity_link(g, home, decision["mode"] if decision["mode"] != "pt" else "walk")
            legs_out, info_out, mode_out, dest_link = adapter._adapter._build_legs(
                decision["mode"], home, dest, dep_min * 60.0, home_link,
                trips_by_stop, stop_node, stops, tt_graphs, 300.0, fallback_counts, trip_seq,
            )
            ret_dep_s = (dep_min + _LEG_MIN + dur_min) * 60.0
            legs_ret, info_ret, mode_ret, home_link_final = adapter._adapter._build_legs(
                mode_out, dest, home, ret_dep_s, dest_link,
                trips_by_stop, stop_node, stops, tt_graphs, 300.0, fallback_counts, trip_seq,
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
    adapter._adapter._insert_doctype(
        out / "population.xml",
        '<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">')
    adapter._adapter._write_real_config(
        out / "config.xml", supply_paths["network"], supply_paths["schedule"],
        supply_paths["vehicles"], flow_capacity_factor, storage_capacity_factor)
    (out / "adapter_manifest.json").write_text(
        json.dumps({"decisions": manifest, "fallback_counts": fallback_counts},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    t_plan_xml = time.perf_counter() - t0

    return {
        "manifest": manifest,
        "decisions": decisions,
        "fallback_counts": fallback_counts,
        "n_states": len(states),
        "timings": {
            "feature_preparation_s": round(t_feature, 3),
            "student_inference_s": round(t_inference, 3),
            "plan_and_xml_s": round(t_plan_xml, 3),
            "total_build_s": round(t_feature + t_inference + t_plan_xml, 3),
        },
    }


# ---------------------------------------------------------------------------
# MATSim launch (production RunMatsimPreloaded launcher, ASCII-safe classpath)
# ---------------------------------------------------------------------------
def _resolve_matsim_env(matsim_dir: str | Path, launcher_java: str | Path):
    """Return (classpath, run_dir) for ``java -cp``.

    Windows: the repo path may contain non-ASCII characters (PowerShell ->
    java argv corruption, same reason the production scripts use a Temp
    junction) — so the MATSim release is reached via a Temp junction and the
    launcher class is compiled into an ASCII Temp dir. POSIX: direct paths.
    """
    matsim_dir = Path(matsim_dir)
    jar = matsim_dir / "matsim-2026.0.jar"
    libs = matsim_dir / "libs"
    if not jar.exists():
        raise FileNotFoundError(
            f"MATSim jar not found: {jar}. Set matsim.matsim_dir to a directory "
            "containing matsim-2026.0.jar (see README: download the official "
            "matsim-libs release)."
        )
    launcher_src = Path(launcher_java)
    if not launcher_src.exists():
        raise FileNotFoundError(f"MATSim launcher source not found: {launcher_src}")

    is_windows = os.name == "nt"
    tmp = Path(os.environ.get("TEMP") or __import__("tempfile").gettempdir())
    if is_windows:
        target = tmp / "reference_matsim_rel"
        try:
            ascii_safe = matsim_dir.as_posix().encode("ascii")
            need_junction = False
        except UnicodeEncodeError:
            need_junction = True
        if need_junction or not target.exists():
            if target.exists():
                # stale junction: remove only if it is a junction
                try:
                    target.rmdir()
                except OSError:
                    pass
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(target), str(matsim_dir)],
                check=True, capture_output=True,
            )
        release_root = target
        sep = ";"
    else:
        release_root = matsim_dir
        sep = ":"

    run_dir = tmp / "reference_matsim_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    class_file = run_dir / "RunMatsimPreloaded.class"
    if not class_file.exists() or launcher_src.stat().st_mtime > class_file.stat().st_mtime:
        compile_cp = f"{release_root / 'matsim-2026.0.jar'}{sep}{release_root / 'libs'}{sep}*"
        proc = subprocess.run(
            ["javac", "-cp", compile_cp, "-d", str(run_dir), str(launcher_src)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to compile {launcher_src}:\n{proc.stdout}\n{proc.stderr}"
            )

    classpath = f"{run_dir}{sep}{release_root / 'matsim-2026.0.jar'}{sep}{release_root / 'libs'}{sep}*"
    return classpath, run_dir


def run_matsim(scenario_dir: str | Path, cfg) -> tuple[int, str]:
    """Run MATSim with the production preloaded-scenario launcher.

    ``cfg`` is the reference MatsimConfig. Returns (exit_code, log_tail)."""
    sdir = Path(scenario_dir)
    config_xml = sdir / ("config.xml" if not cfg.config_file else cfg.config_file)
    if not config_xml.exists():
        raise FileNotFoundError(f"MATSim config not found: {config_xml}")
    if cfg.classpath_override:
        cp = cfg.classpath_override
    else:
        cp, _ = _resolve_matsim_env(cfg.matsim_dir, cfg.launcher_java)
    with open(sdir / "java_run.log", "w", encoding="utf-8", errors="replace") as out_f:
        proc = subprocess.run(
            [cfg.java, f"-Xmx{cfg.java_xmx}", "-cp", cp, "RunMatsimPreloaded", "config.xml"],
            cwd=str(sdir), stdout=out_f, stderr=subprocess.STDOUT, timeout=cfg.timeout_s,
        )
    tail = ""
    log = sdir / "output" / "logfileWarningsErrors.log"
    if log.exists():
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:])
    return proc.returncode, tail


def parse_events(sdir: str | Path) -> dict:
    """Light single-pass event metrics (mirrors run_phase_c / smoke parsers)."""
    import zstandard

    out = {
        "leg_departures": Counter(), "pt_boardings": 0, "pt_alightings": 0,
        "stuck_persons": 0, "stuck_transit_vehicles": 0, "n_events": 0,
        "failed_trips": 0, "mean_trip_time_min": 0.0, "n_trip_legs": 0,
    }
    ev = Path(sdir) / "output" / "ITERS" / "it.0" / "0.events.xml.zst"
    if not ev.exists():
        out["events_missing"] = True
        return out
    dep_stack: dict[str, list[float]] = defaultdict(list)
    n_dep: Counter = Counter()
    n_arr: Counter = Counter()
    trip_time = 0.0
    n_trip_legs = 0

    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(open(ev, "rb")) as f:
        for event, el in ET.iterparse(f, events=("end",)):
            if el.tag != "event":
                continue
            t = float(el.get("time"))
            etype = el.get("type")
            out["n_events"] += 1
            if etype == "departure":
                person = el.get("person")
                if person and _PERSON.match(person):
                    n_dep[person] += 1
                    dep_stack[person].append(t)
                    out["leg_departures"][el.get("legMode")] += 1
            elif etype == "arrival":
                person = el.get("person")
                if person and _PERSON.match(person):
                    n_arr[person] += 1
                    if dep_stack[person]:
                        trip_time += t - dep_stack[person].pop()
                        n_trip_legs += 1
            elif etype == "PersonEntersPtVehicle":
                out["pt_boardings"] += 1
            elif etype == "PersonLeavesPtVehicle":
                out["pt_alightings"] += 1
            elif etype == "stuckAndAbort":
                person = el.get("person") or ""
                if person.startswith("pt_veh"):
                    out["stuck_transit_vehicles"] += 1
                elif _PERSON.match(person):
                    out["stuck_persons"] += 1
            el.clear()

    out["failed_trips"] = sum(max(0, n_dep[p] - n_arr[p]) for p in set(n_dep) | set(n_arr))
    out["mean_trip_time_min"] = round(trip_time / max(1, n_trip_legs) / 60.0, 2)
    out["n_trip_legs"] = n_trip_legs
    out["leg_departures"] = dict(out["leg_departures"])
    return out
