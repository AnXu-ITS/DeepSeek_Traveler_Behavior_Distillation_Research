"""Persistent route / accessibility cache with scenario-bound keys.

The cache stores results of the PRODUCTION routing functions
(``plan_accessibility``, ``SupplyIndex.mode_travel_time`` and the leg
shortest-path lookups). Every cache file carries a metadata header that binds
it to the exact supply and routing state it was built from:

- network version   (network XML SHA256)
- schedule version  (transitSchedule / vehicles / stops / snapping /
                     trips_by_stop SHA256)
- routing state     (access radius, boarding buffer, transfer rules,
                     walk/bike speed constants, extended egress)

A cache is loaded ONLY when its metadata matches the current run. If the
network, schedule or routing regime changed, the old cache is ignored (and,
with ``rebuild=true``, overwritten). There is no cross-scenario reuse of
incompatible caches. Scenario context (weather / fare / delay) never enters
the cache keys — like the production in-process cache, the stored quantities
depend only on (OD, departure, supply).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path

SCHEMA_VERSION = 1

# routing constants that define the "routing state" of the production pipeline
# (S9 fix: walk 1.34 m/s, bike 4.17 m/s — see releases/s9_supply_aware_v2/
# provenance/data_fix.json). Any change to these MUST bump the cache metadata.
ROUTING_STATE = {
    "access_max_m": 700.0,
    "boarding_buffer_s": 300.0,
    "max_transfers": 1,
    "connection_window_s": [180.0, 2700.0],
    "departure_window_s": 3600.0,
    "extended_egress_m": 1500.0,
    "walk_speed_ms": 1.34,
    "bike_speed_ms": 4.17,
    "car_speed": "length/freespeed",
}


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class RouteCache:
    """Disk cache for accessibility vectors, mode travel times, leg routes."""

    def __init__(
        self,
        cache_path: str | Path,
        supply_paths: dict[str, str | Path],
        routing_state: dict | None = None,
        reuse: bool = True,
        rebuild: bool = False,
    ):
        self.path = Path(cache_path)
        self.reuse = reuse
        self.rebuild = rebuild
        self.routing_state = dict(routing_state if routing_state is not None else ROUTING_STATE)
        self.supply_hashes = {
            "network": _sha256(supply_paths["network"]),
            "schedule": _sha256(supply_paths["schedule"]),
            "vehicles": _sha256(supply_paths["vehicles"]),
            "stops": _sha256(supply_paths["stops"]),
            "snapping": _sha256(supply_paths["snapping"]),
            "trips_by_stop": _sha256(supply_paths["trips_by_stop"]),
        }
        self.meta = {
            "schema_version": SCHEMA_VERSION,
            "created_at": None,
            "git_commit": None,
            "supply_hashes": self.supply_hashes,
            "routing_state": self.routing_state,
        }
        # entries: "acc" -> {(o, d, dep_sec): dict}
        #          "tt"  -> {(mode, src, dst): float | None}
        #          "sp"  -> {(mode, src, dst): list[str] | None}
        self._acc: dict = {}
        self._tt: dict = {}
        self._sp: dict = {}
        self.stats = {
            "acc_hits": 0, "acc_misses": 0, "acc_compute_s": 0.0,
            "tt_hits": 0, "tt_misses": 0, "tt_compute_s": 0.0,
            "sp_hits": 0, "sp_misses": 0, "sp_compute_s": 0.0,
        }
        self._persisted_avg: dict[str, float | None] = {"acc": None, "tt": None, "sp": None}
        self._loaded = self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> bool:
        if self.rebuild or not self.reuse or not self.path.exists():
            return False
        try:
            with self.path.open("rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[route_cache] could not read cache {self.path}: {e} — starting fresh")
            return False
        meta = data.get("meta", {})
        if meta.get("schema_version") != SCHEMA_VERSION:
            print(f"[route_cache] cache schema mismatch — starting fresh ({self.path})")
            return False
        if meta.get("supply_hashes", {}) != self.supply_hashes:
            print("[route_cache] SUPPLY CHANGED since cache was built — cache invalidated")
            print(f"[route_cache]   cached: {meta.get('supply_hashes')}")
            print(f"[route_cache]   current: {self.supply_hashes}")
            return False
        if meta.get("routing_state", {}) != self.routing_state:
            print("[route_cache] ROUTING STATE CHANGED since cache was built — cache invalidated")
            return False
        self._acc = data.get("acc", {})
        self._tt = data.get("tt", {})
        self._sp = data.get("sp", {})
        self._persisted_avg = data.get("avg_compute_s", {"acc": None, "tt": None, "sp": None})
        return True

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------- accessors
    def plan_accessibility(self, idx, origin: str, dest: str, dep_sec: float) -> dict:
        key = (origin, dest, float(dep_sec))
        if key in self._acc:
            self.stats["acc_hits"] += 1
            return self._acc[key]
        from traveler_distillation.accessibility.gtfs_accessibility import plan_accessibility
        t0 = time.perf_counter()
        res = plan_accessibility(idx, origin, dest, dep_sec)
        self.stats["acc_compute_s"] += time.perf_counter() - t0
        self.stats["acc_misses"] += 1
        self._acc[key] = res
        return res

    def mode_travel_time(self, mode: str, src: str, dst: str, compute_fn):
        key = (mode, src, dst)
        if key in self._tt:
            self.stats["tt_hits"] += 1
            return self._tt[key]
        t0 = time.perf_counter()
        res = compute_fn()
        self.stats["tt_compute_s"] += time.perf_counter() - t0
        self.stats["tt_misses"] += 1
        self._tt[key] = res
        return res

    def shortest_path(self, mode: str, src: str, dst: str, compute_fn):
        """Cached leg shortest path. ``compute_fn`` is the production
        ``_shortest``-style callable returning a link list or None."""
        key = (mode, src, dst)
        if key in self._sp:
            self.stats["sp_hits"] += 1
            return self._sp[key]
        t0 = time.perf_counter()
        res = compute_fn()
        self.stats["sp_compute_s"] += time.perf_counter() - t0
        self.stats["sp_misses"] += 1
        self._sp[key] = res
        return res

    # ---------------------------------------------------------------- stats
    @property
    def total_calls(self) -> int:
        s = self.stats
        return (s["acc_hits"] + s["acc_misses"] + s["tt_hits"] + s["tt_misses"]
                + s["sp_hits"] + s["sp_misses"])

    @property
    def total_hits(self) -> int:
        s = self.stats
        return s["acc_hits"] + s["tt_hits"] + s["sp_hits"]

    @property
    def hit_rate(self) -> float:
        return self.total_hits / self.total_calls if self.total_calls else 0.0

    @property
    def compute_time_s(self) -> float:
        s = self.stats
        return s["acc_compute_s"] + s["tt_compute_s"] + s["sp_compute_s"]

    @property
    def time_saved_s(self) -> float:
        """Estimated routing time saved by hits. Uses the average compute cost
        per call type measured in THIS run when there were misses; otherwise
        falls back to the averages persisted in the cache file."""
        s = self.stats
        saved = 0.0

        def avg(kind: str) -> float | None:
            misses = s[f"{kind}_misses"]
            if misses:
                return s[f"{kind}_compute_s"] / misses
            return self._persisted_avg.get(kind)

        for kind in ("acc", "tt", "sp"):
            a = avg(kind)
            if a:
                saved += s[f"{kind}_hits"] * a
        return saved

    @property
    def entry_counts(self) -> dict:
        return {"accessibility": len(self._acc), "travel_time": len(self._tt),
                "shortest_path": len(self._sp)}

    def summary(self) -> dict:
        s = self.stats
        return {
            "loaded_from_disk": self._loaded,
            "cache_file": str(self.path),
            "hit_rate": round(self.hit_rate, 6),
            "hits": self.total_hits,
            "misses": self.total_calls - self.total_hits,
            "entries": self.entry_counts,
            "compute_time_s": round(self.compute_time_s, 3),
            "estimated_routing_time_saved_s": round(self.time_saved_s, 3),
            "per_kind": {
                "acc": {"hits": s["acc_hits"], "misses": s["acc_misses"]},
                "tt": {"hits": s["tt_hits"], "misses": s["tt_misses"]},
                "sp": {"hits": s["sp_hits"], "misses": s["sp_misses"]},
            },
        }

    # ---------------------------------------------------------------- flush
    def flush(self) -> None:
        if not (self._acc or self._tt or self._sp):
            return
        self.meta["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.meta["git_commit"] = _git_commit()
        avg_compute = {}
        for kind in ("acc", "tt", "sp"):
            misses = self.stats[f"{kind}_misses"]
            if misses:
                avg_compute[kind] = self.stats[f"{kind}_compute_s"] / misses
            else:
                avg_compute[kind] = self._persisted_avg.get(kind)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump({
                "meta": self.meta,
                "avg_compute_s": avg_compute,
                "acc": self._acc,
                "tt": self._tt,
                "sp": self._sp,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta_sidecar = self.path.with_suffix(".meta.json")
        meta_sidecar.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0
