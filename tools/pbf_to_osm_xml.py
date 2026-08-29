#!/usr/bin/env python
"""OSM PBF -> OSM XML (bbox-cropped), pure Python, zero dependencies.

Converts an OpenStreetMap .pbf (Geofabrik-style, DenseNodes encoding) into an
Overpass-like .osm XML limited to a bounding box, so the existing
``src/traveler_distillation/singapore/osm_network.py::build_network`` can
consume it unchanged (E5 Helsinki supply step; the venv has no osmium/pyosm).

Implementation notes (PBF spec: fileformat.proto / osmformat.proto):
- File framing: 4-byte big-endian BlobHeader length, then BlobHeader proto,
  then Blob proto (size = BlobHeader.datasize — the Blob itself is NOT
  length-prefixed).
- Blobs are zlib-compressed (field 3) or raw (field 1).
- PrimitiveBlock: stringtable + primitive groups; granularity/lat_offset/
  lon_offset per block (defaults 100 / 0 / 0). Node lat/lon = offset +
  granularity * raw (in nanodegrees, then /1e9).
- sint64 fields (node id, lat, lon, dense deltas, way refs) are zigzag-coded;
  way id is int64 (plain varint); keys/vals are uint32 (plain).
- DenseNodes id/lat/lon are delta-coded cumulative; keys_vals is a flat
  int32 stream of (key_idx, val_idx) pairs terminated by 0.
- Relations and node tags are skipped (osm_network only needs node id+coords
  and way refs+tags).

Semantics: Overpass-equivalent to the Singapore extraction
(``way["highway"](bbox); >; out body;``) — only ways carrying a highway tag
(with at least one node inside the bbox) and ALL of their member nodes are
emitted (members may lie outside the bbox). Non-highway ways (buildings,
landuse, ...) and nodes not referenced by any kept highway way are dropped.

Three passes over the file:
  pass 1: collect ids of nodes inside the bbox (set of ints);
  pass 2: collect highway ways with >=1 in-bbox node (cache in memory) and
          their full node refs (wanted node set);
  pass 3: emit wanted nodes from the stream + emit cached ways.

Usage:
    python tools/pbf_to_osm_xml.py hsl/hsl.osm.pbf data/helsinki/osm/helsinki_sub.osm \
        --lat-min 60.145 --lat-max 60.230 --lon-min 24.875 --lon-max 25.060 --margin 0.004
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from traveler_distillation.singapore.osm_network import _KEEP as _HIGHWAY_KEEP  # noqa: E402


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Plain (non-zigzag) protobuf varint -> (value, new_pos)."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _zigzag(v: int) -> int:
    return (v >> 1) ^ -(v & 1)


def _iter_fields(buf: bytes):
    """Yield (field_number, wire_type, value|bytes) over a proto message."""
    pos = 0
    n = len(buf)
    while pos < n:
        key, pos = _read_varint(buf, pos)
        fnum, wtype = key >> 3, key & 7
        if wtype == 0:
            v, pos = _read_varint(buf, pos)
            yield fnum, wtype, v
        elif wtype == 1:
            v = struct.unpack("<q", buf[pos:pos + 8])[0]
            pos += 8
            yield fnum, wtype, v
        elif wtype == 2:
            ln, pos = _read_varint(buf, pos)
            yield fnum, wtype, buf[pos:pos + ln]
            pos += ln
        elif wtype == 5:
            v = struct.unpack("<i", buf[pos:pos + 4])[0]
            pos += 4
            yield fnum, wtype, v
        else:
            raise ValueError(f"unsupported wire type {wtype} at byte {pos}")


def _packed_varints(buf: bytes) -> list[int]:
    out = []
    pos = 0
    n = len(buf)
    while pos < n:
        v, pos = _read_varint(buf, pos)
        out.append(v)
    return out


def _decode_dense(buf: bytes) -> tuple[list[int], list[int], list[int], list[int]]:
    """Decode a DenseNodes message -> (ids, lats_raw, lons_raw, keys_vals)."""
    ids_raw: bytes | None = None
    lats_raw: bytes | None = None
    lons_raw: bytes | None = None
    keys_vals: list[int] = []
    for fnum, _w, v in _iter_fields(buf):
        if fnum == 1:
            ids_raw = v
        elif fnum == 8:
            lats_raw = v
        elif fnum == 9:
            lons_raw = v
        elif fnum == 10:
            keys_vals = _packed_varints(v)
    ids = _cum_zigzag(_packed_varints(ids_raw)) if ids_raw is not None else []
    lats = _cum_zigzag(_packed_varints(lats_raw)) if lats_raw is not None else []
    lons = _cum_zigzag(_packed_varints(lons_raw)) if lons_raw is not None else []
    return ids, lats, lons, keys_vals


def _cum_zigzag(vals: list[int]) -> list[int]:
    acc = 0
    out = []
    for v in vals:
        acc += _zigzag(v)
        out.append(acc)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _blob_iter(fin):
    """Yield (blob_type, payload_bytes) over the PBF framing."""
    while True:
        hl_raw = fin.read(4)
        if not hl_raw:
            return
        hl = struct.unpack(">I", hl_raw)[0]
        hdr_blob = fin.read(hl)
        hdr_type = ""
        datasize = 0
        for fnum, _w, v in _iter_fields(hdr_blob):
            if fnum == 1:
                hdr_type = v.decode("utf-8", "replace")
            elif fnum == 3:
                datasize = v
        blob = fin.read(datasize)
        payload = None
        for fnum, _w, v in _iter_fields(blob):
            if fnum == 3:
                payload = zlib.decompress(v)
            elif fnum == 1 and payload is None:
                payload = v
        if payload is not None:
            yield hdr_type, payload


def _iter_nodes(block_payload: bytes):
    """Yield (node_id, lat_deg, lon_deg) from a PrimitiveBlock payload."""
    granularity, lat_off, lon_off = 100, 0, 0
    groups: list[bytes] = []
    for fnum, _w, v in _iter_fields(block_payload):
        if fnum == 17:
            granularity = v
        elif fnum == 19:
            lat_off = v
        elif fnum == 20:
            lon_off = v
        elif fnum == 2:
            groups.append(v)
    for g in groups:
        for gfn, _gw, gv in _iter_fields(g):
            if gfn == 2:  # DenseNodes
                ids, lats, lons, _kv = _decode_dense(gv)
                for nid, lat_raw, lon_raw in zip(ids, lats, lons):
                    yield nid, (lat_off + granularity * lat_raw) * 1e-9, (lon_off + granularity * lon_raw) * 1e-9
            elif gfn == 1:  # regular Node
                nid = lat_raw = lon_raw = None
                for nfn, _nw, nv in _iter_fields(gv):
                    if nfn == 1:
                        nid = _zigzag(nv)
                    elif nfn == 8:
                        lat_raw = _zigzag(nv)
                    elif nfn == 9:
                        lon_raw = _zigzag(nv)
                if nid is not None and lat_raw is not None and lon_raw is not None:
                    yield nid, (lat_off + granularity * lat_raw) * 1e-9, (lon_off + granularity * lon_raw) * 1e-9


def _iter_ways(block_payload: bytes):
    """Yield (way_id, refs, tags) from a PrimitiveBlock payload."""
    stringtable: list[bytes] = []
    groups: list[bytes] = []
    for fnum, _w, v in _iter_fields(block_payload):
        if fnum == 1:
            for sfn, _sw, sv in _iter_fields(v):
                if sfn == 1:
                    stringtable.append(sv)
        elif fnum == 2:
            groups.append(v)
    for g in groups:
        for gfn, _gw, gv in _iter_fields(g):
            if gfn != 3:
                continue
            wid = None
            refs: list[int] = []
            keys: list[int] = []
            vals_raw: list[int] = []
            for wfn, _ww, wv in _iter_fields(gv):
                if wfn == 1:
                    wid = wv  # int64, plain varint (positive)
                elif wfn == 8:
                    refs = _cum_zigzag(_packed_varints(wv))
                elif wfn == 2:
                    keys = _packed_varints(wv)
                elif wfn == 3:
                    vals_raw = _packed_varints(wv)
            tags = [
                (
                    stringtable[ki].decode("utf-8", "replace") if ki < len(stringtable) else "",
                    stringtable[vi].decode("utf-8", "replace") if vi < len(stringtable) else "",
                )
                for ki, vi in zip(keys, vals_raw)
            ]
            if wid is not None:
                yield wid, refs, tags


def convert(src: str | Path, dst: str | Path, bbox: dict, margin: float = 0.0) -> dict:
    src, dst = Path(src), Path(dst)
    lat_lo, lat_hi = bbox["lat_min"] - margin, bbox["lat_max"] + margin
    lon_lo, lon_hi = bbox["lon_min"] - margin, bbox["lon_max"] + margin
    t0 = time.time()

    def in_bbox(lat: float, lon: float) -> bool:
        return lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi

    # ---------------------------------------------------------------- pass 1
    bbox_nodes: set[int] = set()
    header_meta: dict = {}
    with src.open("rb") as fin:
        for hdr_type, payload in _blob_iter(fin):
            if hdr_type == "OSMHeader":
                for fnum, _w, v in _iter_fields(payload):
                    if fnum == 1:  # bbox message
                        bb = {}
                        for bfn, _bw, bv in _iter_fields(v):
                            bb[bfn] = bv
                        header_meta["bbox_lat_lo"] = _zigzag(bb.get(4, 0)) * 1e-9
                        header_meta["bbox_lat_hi"] = _zigzag(bb.get(3, 0)) * 1e-9
                        header_meta["bbox_lon_lo"] = _zigzag(bb.get(1, 0)) * 1e-9
                        header_meta["bbox_lon_hi"] = _zigzag(bb.get(2, 0)) * 1e-9
            elif hdr_type == "OSMData":
                for nid, lat, lon in _iter_nodes(payload):
                    if in_bbox(lat, lon):
                        bbox_nodes.add(nid)

    pass1_s = time.time() - t0
    print(f"pass 1: nodes in bbox = {len(bbox_nodes)} ({pass1_s:.0f}s)", flush=True)

    # ---------------------------------------------------------------- pass 2
    wanted_ways: list[tuple[int, list[int], list[tuple[str, str]]]] = []
    wanted_nodes: set[int] = set()
    n_ways_seen = 0
    with src.open("rb") as fin:
        for hdr_type, payload in _blob_iter(fin):
            if hdr_type != "OSMData":
                continue
            for wid, refs, tags in _iter_ways(payload):
                n_ways_seen += 1
                tag_map = dict(tags)
                if tag_map.get("highway") not in _HIGHWAY_KEEP:
                    continue
                if len(refs) < 2:
                    continue
                if not any(r in bbox_nodes for r in refs):
                    continue
                wanted_ways.append((wid, refs, tags))
                wanted_nodes.update(refs)

    pass2_s = time.time() - t0 - pass1_s
    print(f"pass 2: highway ways kept = {len(wanted_ways)} (of {n_ways_seen}); "
          f"wanted nodes = {len(wanted_nodes)} ({pass2_s:.0f}s)", flush=True)

    # ---------------------------------------------------------------- pass 3
    dst.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    emitted_nodes: set[int] = set()

    def _emit_node(f, nid: int, lat: float, lon: float):
        f.write(f'  <node id="{nid}" lat="{lat:.7f}" lon="{lon:.7f}"/>\n')
        emitted_nodes.add(nid)

    with src.open("rb") as fin, dst.open("w", encoding="utf-8") as fout:
        fout.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fout.write('<osm version="0.6" generator="tools/pbf_to_osm_xml.py">\n')
        for hdr_type, payload in _blob_iter(fin):
            if hdr_type != "OSMData":
                continue
            for nid, lat, lon in _iter_nodes(payload):
                if nid in wanted_nodes:
                    _emit_node(fout, nid, lat, lon)
        for wid, refs, tags in wanted_ways:
            fout.write(f'  <way id="{wid}">\n')
            for r in refs:
                fout.write(f'    <nd ref="{r}"/>\n')
            for k, v in tags:
                kk = k.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
                vv = v.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
                fout.write(f'    <tag k="{kk}" v="{vv}"/>\n')
            fout.write("  </way>\n")
        fout.write("</osm>\n")

    pass3_s = time.time() - t1
    n_way_refs = sum(len(r) for _w, r, _t in wanted_ways)
    missing_refs = [r for _w, refs, _t in wanted_ways for r in refs if r not in emitted_nodes]
    meta = {
        "source": str(src),
        "source_sha256": _sha256(src),
        "bbox": bbox,
        "margin_deg": margin,
        "source_header_bbox": header_meta,
        "semantics": "Overpass-equivalent: way[highway] with >=1 node in bbox; all member nodes emitted (may lie outside bbox)",
        "nodes_in_bbox": len(bbox_nodes),
        "highway_ways_kept": len(wanted_ways),
        "nodes_emitted": len(emitted_nodes),
        "way_refs_total": n_way_refs,
        "way_refs_missing": len(missing_refs),
        "pass1_s": round(pass1_s, 1),
        "pass2_s": round(pass2_s, 1),
        "pass3_s": round(pass3_s, 1),
        "total_s": round(time.time() - t0, 1),
    }
    dst_meta = dst.with_suffix(".meta.json")
    dst_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if len(emitted_nodes) == 0:
        print("ERROR: zero nodes emitted — bbox likely outside the source file", file=sys.stderr)
    if missing_refs:
        print(f"ERROR: {len(missing_refs)} way refs lack node coordinates", file=sys.stderr)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--lat-min", type=float, default=60.145)
    ap.add_argument("--lat-max", type=float, default=60.230)
    ap.add_argument("--lon-min", type=float, default=24.875)
    ap.add_argument("--lon-max", type=float, default=25.060)
    ap.add_argument("--margin", type=float, default=0.004)
    args = ap.parse_args()
    bbox = {"lat_min": args.lat_min, "lat_max": args.lat_max,
            "lon_min": args.lon_min, "lon_max": args.lon_max}
    convert(args.src, args.dst, bbox, args.margin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
