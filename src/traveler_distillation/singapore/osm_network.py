"""OSM extract -> MATSim network.xml (Phase A supply step).

Streams the Overpass OSM XML (102k+ nodes / 30k+ highway ways), keeps ways by
highway class, projects node coordinates WGS84 -> UTM 48N, and writes a MATSim
v1 network with per-highway-class freespeed / capacity / allowed modes:

- car  : motorway..service (+ residential/road), NOT on footway/cycleway/path/steps;
- bike : additionally on cycleway/footway/path/pedestrian (not steps, not expressways);
- walk : additionally on steps/pedestrian/footway/path/cycleway (not expressways).

One-way handling follows OSM tags (yes/1/-1), everything else is bidirectional
(two directed links). QA output: per-mode connectivity (largest component),
link/nodes counts, length/speed/capacity distributions, bbox — so the network
is auditable before it ever reaches MATSim.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import networkx as nx

from .projection import utm48n

# highway class -> (freespeed m/s, default lanes, capacity veh/h per lane, car, bike, walk)
_HIGHWAY = {
    "motorway":       (25.0, 2, 2000, True,  False, False),
    "motorway_link":  (17.0, 1, 1500, True,  False, False),
    "trunk":          (22.0, 2, 1800, True,  False, False),
    "trunk_link":     (15.0, 1, 1400, True,  False, False),
    "primary":        (18.0, 2, 1500, True,  True,  True),
    "primary_link":   (14.0, 1, 1200, True,  True,  True),
    "secondary":      (15.0, 2, 1000, True,  True,  True),
    "secondary_link": (12.0, 1, 900,  True,  True,  True),
    "tertiary":       (13.0, 2, 800,  True,  True,  True),
    "tertiary_link":  (11.0, 1, 700,  True,  True,  True),
    "unclassified":   (11.0, 1, 600,  True,  True,  True),
    "residential":    (8.3,  1, 600,  True,  True,  True),
    "living_street":  (5.5,  1, 300,  True,  True,  True),
    "service":        (7.0,  1, 400,  True,  True,  True),
    "road":           (10.0, 1, 600,  True,  True,  True),
    "footway":        (1.4,  1, 1000, False, True,  True),
    "cycleway":       (3.9,  1, 1000, False, True,  True),
    "path":           (1.4,  1, 600,  False, True,  True),
    "pedestrian":     (1.4,  1, 1000, False, True,  True),
    "steps":          (0.8,  1, 600,  False, False, True),
}

_KEEP = set(_HIGHWAY)


def _parse_speed(raw: str | None) -> float | None:
    """OSM maxspeed tag -> m/s (None if unparseable)."""
    if not raw:
        return None
    s = raw.strip().lower().replace("km/h", "").replace("kph", "").replace(" ", "")
    try:
        return float(s) / 3.6
    except ValueError:
        return None


def _parse_int(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _iter_elements(path: Path):
    """Stream (tag, attrs-dict, clear-fn) events; callers must clear."""
    for event, elem in ET.iterparse(path, events=("start", "end")):
        yield event, elem


def _collect_ways_and_nodes(osm_path: Path):
    """Two streaming passes: ways first (node-id demand), then node coords."""
    ways = []           # (way_id, [node ids], tags dict)
    kept_nodes: set[int] = set()

    for event, elem in _iter_elements(osm_path):
        if event == "end" and elem.tag == "way":
            tags = {}
            nds = []
            for child in elem:
                if child.tag == "nd":
                    nds.append(int(child.get("ref")))
                elif child.tag == "tag":
                    tags[child.get("k")] = child.get("v")
            hw = tags.get("highway")
            if hw in _KEEP and len(nds) >= 2:
                ways.append((int(elem.get("id")), nds, tags))
                kept_nodes.update(nds)
            elem.clear()

    node_coords: dict[int, tuple[float, float]] = {}
    for event, elem in _iter_elements(osm_path):
        if event == "end" and elem.tag == "node":
            nid = int(elem.get("id"))
            if nid in kept_nodes:
                node_coords[nid] = (float(elem.get("lat")), float(elem.get("lon")))
            elem.clear()
    return ways, node_coords


def build_network(
    osm_path: str | Path,
    out_network: str | Path,
    out_stats: str | Path,
) -> dict:
    """Convert OSM extract to MATSim network.xml; write QA stats JSON."""
    osm_path = Path(osm_path)
    ways, node_coords = _collect_ways_and_nodes(osm_path)

    # project all kept nodes once
    projected = {nid: utm48n(lat, lon) for nid, (lat, lon) in node_coords.items()}

    nodes_el = ET.Element("nodes")
    for nid, (x, y) in projected.items():
        ET.SubElement(nodes_el, "node", id=str(nid), x=f"{x:.3f}", y=f"{y:.3f}")

    links_el = ET.Element(
        "links", capperiod="01:00:00", effectivecellsize="7.5", effectivelanewidth="3.75"
    )
    n_links = 0
    skipped_short = 0
    len_hist: list[float] = []
    speed_hist: list[float] = []
    cap_hist: list[float] = []
    mode_counter: Counter = Counter()
    g_mode = {m: nx.Graph() for m in ("car", "bike", "walk")}
    # multiple OSM ways can share the same consecutive node pair -> the MATSim
    # link id (from_to) must be unique; merge duplicates (union modes, max
    # capacity, conservative freespeed).
    link_merge: dict[tuple[str, str], dict] = {}

    def _emit(fr: int, to: int, length: float, free: float, cap: float, modes: str):
        nonlocal n_links
        if length < 0.5:
            return
        key = (str(fr), str(to))
        if key in link_merge:
            old = link_merge[key]
            old["modes"] = ",".join(sorted(set(old["modes"].split(",")) | set(modes.split(","))))
            old["capacity"] = max(old["capacity"], cap)
            old["freespeed"] = min(old["freespeed"], free)
            return
        link_merge[key] = {"length": length, "freespeed": free, "capacity": cap, "modes": modes}

    def _flush():
        nonlocal n_links
        for (fr, to), d in sorted(link_merge.items()):
            ET.SubElement(
                links_el, "link",
                id=f"{fr}_{to}",
                **{"from": fr, "to": to, "length": f"{d['length']:.3f}",
                   "freespeed": f"{d['freespeed']:.3f}", "capacity": f"{d['capacity']:.1f}",
                   "permlanes": "1.0", "oneway": "1", "modes": d["modes"]},
            )
            n_links += 1
            len_hist.append(d["length"])
            speed_hist.append(d["freespeed"])
            cap_hist.append(d["capacity"])
            for m in d["modes"].split(","):
                g_mode[m].add_edge(fr, to, length=d["length"], free=d["freespeed"])
                mode_counter[m] += 1

    for way_id, nds, tags in ways:
        free0, lanes0, cap0, car_ok, bike_ok, walk_ok = _HIGHWAY[tags.get("highway")]
        free = _parse_speed(tags.get("maxspeed")) or free0
        lanes = _parse_int(tags.get("lanes")) or lanes0
        cap = cap0 * max(1, lanes)
        oneway = (tags.get("oneway") or "").strip().lower()
        modes = ",".join(
            m for m, ok in (("car", car_ok), ("bike", bike_ok), ("walk", walk_ok)) if ok
        )
        pts = [projected[nid] for nid in nds]
        if oneway in ("yes", "true", "1"):
            for i in range(len(nds) - 1):
                (x1, y1), (x2, y2) = pts[i], pts[i + 1]
                _emit(nds[i], nds[i + 1], math.hypot(x2 - x1, y2 - y1), free, cap, modes)
        elif oneway == "-1":
            for i in range(len(nds) - 1):
                (x1, y1), (x2, y2) = pts[i], pts[i + 1]
                _emit(nds[i + 1], nds[i], math.hypot(x2 - x1, y2 - y1), free, cap, modes)
        else:
            for i in range(len(nds) - 1):
                (x1, y1), (x2, y2) = pts[i], pts[i + 1]
                length = math.hypot(x2 - x1, y2 - y1)
                _emit(nds[i], nds[i + 1], length, free, cap, modes)
                _emit(nds[i + 1], nds[i], length, free, cap, modes)

    network_root = ET.Element("network", name="tampines-pasir-ris-osm")
    network_root.append(nodes_el)
    network_root.append(links_el)
    _flush()
    out = Path(out_network)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(network_root).write(out, encoding="utf-8", xml_declaration=True)
    _insert_doctype(out, '<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v1.dtd">')

    def _q(vals):
        s = sorted(vals)
        return {
            "n": len(s),
            "min": round(s[0], 2) if s else None,
            "p50": round(s[len(s) // 2], 2) if s else None,
            "max": round(s[-1], 2) if s else None,
        }

    stats = {
        "source": str(osm_path),
        "ways_kept": len(ways),
        "nodes": len(projected),
        "links": n_links,
        "mode_link_counts": dict(mode_counter),
        "length_m": _q(len_hist),
        "freespeed_ms": _q(speed_hist),
        "capacity_veh_h": _q(cap_hist),
        "total_network_km": round(sum(len_hist) / 1000.0, 2),
        "connectivity": {},
    }
    for m, g in g_mode.items():
        comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
        stats["connectivity"][m] = {
            "components": len(comps),
            "largest_component_nodes": comps[0] if comps else 0,
            "fraction_in_largest": round(comps[0] / g.number_of_nodes(), 4) if g.number_of_nodes() else None,
        }
    Path(out_stats).parent.mkdir(parents=True, exist_ok=True)
    Path(out_stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


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


def load_network_graph(network_path: str | Path, modes: str = "car,bike,walk") -> nx.DiGraph:
    """Load a MATSim network.xml into a directed graph with coords attached.

    Returns (graph, nodes_xy) where graph nodes are MATSim node ids and edges
    carry ``length`` / ``free`` / ``modes`` / ``capacity``.
    """
    g = nx.DiGraph()
    xy: dict[str, tuple[float, float]] = {}
    root = ET.parse(Path(network_path)).getroot()
    for node in root.find("nodes"):
        nid = node.get("id")
        x, y = float(node.get("x")), float(node.get("y"))
        g.add_node(nid, x=x, y=y)
        xy[nid] = (x, y)
    for link in root.find("links"):
        modes_set = set((link.get("modes") or "car").split(","))
        g.add_edge(
            link.get("from"), link.get("to"),
            length=float(link.get("length")), free=float(link.get("freespeed")),
            capacity=float(link.get("capacity")), modes=modes_set,
            id=link.get("id"),
        )
    return g, xy


if __name__ == "__main__":
    build_network(
        "data/singapore/osm/tampines_pasir_ris.osm",
        "data/singapore/osm/network.xml",
        "data/singapore/osm/network_stats.json",
    )
