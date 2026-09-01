#!/usr/bin/env python
"""Student deployment pipeline benchmark (audit companion — does NOT modify any
existing pipeline code; imports the production modules read-only).

Ground truth used here:
  * MATSim is launched via ``RunMatsimPreloaded`` (tools/java/) with
    ``lastIteration=0`` — the Student is NEVER called during the Java run.
  * All decisions are precomputed OFFLINE in ``MATSimAdapter.build_real_scenario``
    (alt_factory -> UniversalTravelerState -> adapter.decide -> legs -> XML),
    then handed to MATSim as population.xml (file-based handoff).
  * This script benchmarks that real offline pipeline with the real S9 checkpoint,
    real E2 state pool (outputs/e2_efficiency/states_100000.jsonl, G2-verified
    identical to frozen Phase C C0), and the real Singapore supply for the E2E leg.

Modes:
  profile      cold start + warm per-state components (decide/encode/forward)
  batch        batch-size sweep 1/16/64/256/1024 on 10k real states
  scale        1k/10k/100k(/1M) pure forward throughput (measured)
  e2e1k        real 1k-state build with per-component timers (accessibility=real)
  ipc          prototype persistent socket worker (per-request latency)
  onnx         ONNX export + parity + latency (requires onnx + onnxruntime)
  correctness  batch vs single decide() + vs frozen C0 manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
import struct
import subprocess
import sys
import threading
import time
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------- utilities
def _pstats(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)
    return {
        "n": n,
        "mean": sum(xs) / n,
        "median": xs[n // 2],
        "p95": xs[min(n - 1, int(n * 0.95))],
    }


def _write_csv(path: Path, row: dict):
    path = Path(path)
    cols = ["method", "batch_size", "num_decisions", "total_time",
            "ms_per_decision", "decisions_per_second", "cold_or_warm"]
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def _timeit(fn, repeats: int, warm: int = 3):
    for _ in range(warm):
        fn()
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return _pstats(xs)


# ---------------------------------------------------------------- state pool
def _load_states(limit: int | None, path: Path | None = None):
    """Reconstruct real UniversalTravelerState objects from the E2 pool."""
    from traveler_distillation.schemas.alternative import TravelAlternative
    from traveler_distillation.schemas.context import DynamicContext, Weather
    from traveler_distillation.schemas.persona import Persona
    from traveler_distillation.schemas.state import UniversalTravelerState
    from traveler_distillation.schemas.trip import Trip

    p = path or ROOT / "outputs" / "e2_efficiency" / "states_100000.jsonl"
    states = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            d = json.loads(line)
            states.append(UniversalTravelerState(
                persona=Persona(**d["persona"]),
                trip=Trip(**d["trip"]),
                context=DynamicContext(
                    **{k: (Weather(**v) if k == "weather" else v) for k, v in d["context"].items()}
                ),
                alternatives=[TravelAlternative(**a) for a in d["alternatives"]],
            ))
    return states


def _load_adapter(threads: int = 1):
    import torch
    from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    adapter = S8MATSimAdapter(ROOT / "releases" / "s9_supply_aware_v2" / "checkpoint" / "model.pt")
    return adapter, time.perf_counter() - t0


def _encode_batch(adapter, states: list):
    from traveler_distillation.student.dataset import collate_batch
    import torch
    feats = [adapter.extractor.encode(s) for s in states]
    batch = collate_batch([{
        "global_cat": torch.tensor(f["global_cat"], dtype=torch.long),
        "global_num": torch.tensor(f["global_num"], dtype=torch.float32),
        "alt_mode_idx": torch.tensor(f["alt_mode_idx"], dtype=torch.long),
        "alt_num": torch.tensor(f["alt_num"], dtype=torch.float32),
        "alt_mask": torch.tensor(f["alt_available"], dtype=torch.float32),
    } for f in feats])
    return feats, batch


def _decode_batch(adapter, states, out):
    probs_np = out["mode_probabilities"].cpu().numpy()
    shift_np = out["departure_time_shift_min"].cpu().numpy()
    decisions = []
    for i, state in enumerate(states):
        probs = {
            alt.mode: float(probs_np[i][j])
            for j, alt in enumerate(state.alternatives)
            if alt.available
        }
        chosen = max(probs, key=probs.get)
        decisions.append({"mode": chosen, "mode_probabilities": probs,
                          "departure_time_shift_min": float(shift_np[i])})
    return decisions


# ---------------------------------------------------------------- modes
def mode_profile(args):
    import torch
    out_rows = []

    # cold: python import + module import + checkpoint load + first decide
    t0 = time.perf_counter()
    code = (
        "import sys, time; sys.path.insert(0, r'%s'); t0=time.perf_counter(); import torch; "
        "import traveler_distillation.matsim.s8_adapter as m; "
        "a=m.S8MATSimAdapter(r'%s'); "
        "print(time.perf_counter()-t0, a._source_checkpoint)"
        % (ROOT / "src",
           ROOT / "releases" / "s9_supply_aware_v2" / "checkpoint" / "model.pt")
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    cold_import_load = float(r.stdout.split()[0])
    print(f"[cold] python start + torch import + module import + checkpoint load: {cold_import_load:.3f} s")

    adapter, load_s = _load_adapter(threads=1)
    states = _load_states(limit=1000)

    def one_decide():
        return adapter.decide(states[0])

    st = _timeit(one_decide, repeats=100, warm=10)
    print(f"[warm] decide() per state (encode+collate+forward+decode, threads=1): "
          f"mean {st['mean']*1e3:.3f} ms, median {st['median']*1e3:.3f} ms, p95 {st['p95']*1e3:.3f} ms")

    def one_encode():
        return adapter.extractor.encode(states[0])

    en = _timeit(one_encode, repeats=100, warm=10)
    print(f"[warm] S8FeatureExtractor.encode() only: mean {en['mean']*1e3:.3f} ms")

    feats, batch = _encode_batch(adapter, states[:1])
    def one_forward():
        with torch.no_grad():
            return adapter.model(batch)
    fw = _timeit(one_forward, repeats=100, warm=10)
    print(f"[warm] model forward B=1 only: mean {fw['mean']*1e3:.3f} ms")

    for label, s in (("decide_per_state_ms", st), ("encode_only_ms", en), ("forward_b1_ms", fw)):
        _write_csv(args.csv, {
            "method": label, "batch_size": 1, "num_decisions": s["n"],
            "total_time": s["mean"] * s["n"], "ms_per_decision": s["mean"] * 1e3,
            "decisions_per_second": 1.0 / s["mean"], "cold_or_warm": "warm"})
    _write_csv(args.csv, {
        "method": "cold_import_plus_checkpoint_load", "batch_size": "",
        "num_decisions": 1, "total_time": cold_import_load,
        "ms_per_decision": cold_import_load * 1e3, "decisions_per_second": 1.0 / cold_import_load,
        "cold_or_warm": "cold"})
    _write_csv(args.csv, {
        "method": "checkpoint_load_only", "batch_size": "",
        "num_decisions": 1, "total_time": load_s, "ms_per_decision": load_s * 1e3,
        "decisions_per_second": "", "cold_or_warm": "cold"})
    print(f"[cold] checkpoint load only (in-process): {load_s:.3f} s")


def mode_batch(args):
    import torch
    adapter, _ = _load_adapter(threads=1)
    states = _load_states(limit=10000)
    print(f"loaded {len(states)} real states")

    # reference: per-state decide() on 500 states (original pipeline)
    ref = [adapter.decide(s) for s in states[:500]]

    for bs in (1, 16, 64, 256, 1024):
        t0 = time.perf_counter()
        for i in range(0, len(states), bs):
            chunk = states[i:i + bs]
            feats, batch = _encode_batch(adapter, chunk)
            with torch.no_grad():
                out = adapter.model(batch)
            _decode_batch(adapter, chunk, out)
        dt = time.perf_counter() - t0
        per = dt / len(states) * 1e3
        dps = len(states) / dt
        print(f"[batch] bs={bs:4d}: {dt:.3f} s total, {per:.3f} ms/state, {dps:,.0f} decisions/s")
        _write_csv(args.csv, {
            "method": "batch_encode_forward_decode", "batch_size": bs,
            "num_decisions": len(states), "total_time": dt,
            "ms_per_decision": per, "decisions_per_second": dps, "cold_or_warm": "warm"})
        if bs == 1024:
            # correctness vs reference decide() on first 500
            feats, batch = _encode_batch(adapter, states[:500])
            with torch.no_grad():
                out = adapter.model(batch)
            dec = _decode_batch(adapter, states[:500], out)
            same_mode = sum(1 for a, b in zip(ref, dec) if a["mode"] == b["mode"])
            max_abs = max(abs(a["departure_time_shift_min"] - b["departure_time_shift_min"])
                          for a, b in zip(ref, dec))
            print(f"[correctness] batch vs single decide(): mode match {same_mode}/500, "
                  f"max |Δshift| {max_abs:.2e}")


def mode_scale(args):
    import math
    import torch
    adapter, _ = _load_adapter(threads=24)
    pool = _load_states(limit=100000)
    target = args.max_n
    n_cycles = math.ceil(target / len(pool))

    # encode timing: cycle through the 100k pool chunk-wise (compute-bound)
    t_enc = 0.0
    encoded_batches = []
    for c in range(n_cycles):
        chunk = pool[: min(len(pool), target - c * len(pool))]
        if not chunk:
            break
        t0 = time.perf_counter()
        feats, batch = _encode_batch(adapter, chunk)
        t_enc += time.perf_counter() - t0
        encoded_batches.append(batch)

    # forward timing: bs=1024 sub-batches over the encoded chunks
    t_fwd = 0.0
    with torch.no_grad():
        for batch in encoded_batches:
            B = batch["global_cat"].shape[0]
            for i in range(0, B, 1024):
                sub = {k: v[i:i + 1024] for k, v in batch.items()}
                t0 = time.perf_counter()
                _ = adapter.model(sub)
                t_fwd += time.perf_counter() - t0

    n = target
    print(f"[scale] N={n:,} (pool cycled {n_cycles}x): encode {t_enc:.2f} s ({t_enc/n*1e3:.3f} ms/state), "
          f"forward bs=1024 {t_fwd:.2f} s ({t_fwd/n*1e3:.3f} ms/state, {n/t_fwd:,.0f} decisions/s) measured")
    _write_csv(args.csv, {
        "method": "encode_only_scale", "batch_size": 1024, "num_decisions": n,
        "total_time": t_enc, "ms_per_decision": t_enc / n * 1e3,
        "decisions_per_second": n / t_enc, "cold_or_warm": "warm"})
    _write_csv(args.csv, {
        "method": "forward_only_scale", "batch_size": 1024, "num_decisions": n,
        "total_time": t_fwd, "ms_per_decision": t_fwd / n * 1e3,
        "decisions_per_second": n / t_fwd, "cold_or_warm": "warm"})


def mode_cached(args):
    """Experiment E: accessibility OD cache (disk-persistent) vs recompute.

    The SAME (OD, departure) accessibility vector is recomputed in every
    scenario/run today (per-factory in-process cache only). This measures the
    optimized path: precompute once -> pickle to disk -> load -> build
    alternatives directly (no plan_accessibility call).
    """
    import pickle
    import torch
    from traveler_distillation.accessibility.gtfs_accessibility import SupplyIndex, plan_accessibility
    from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives
    from traveler_distillation.config import load_yaml
    from traveler_distillation.generators import PersonaGenerator, TripGenerator
    from traveler_distillation.schemas.context import DynamicContext, Weather
    from traveler_distillation.schemas.state import UniversalTravelerState

    SUPPLY = {
        "network": str(ROOT / "data/singapore/transit/network_with_transit.xml"),
        "schedule": str(ROOT / "data/singapore/transit/transitSchedule.xml"),
        "vehicles": str(ROOT / "data/singapore/transit/transitVehicles.xml"),
        "stops": str(ROOT / "data/singapore/transit/prep_stops.jsonl"),
        "snapping": str(ROOT / "data/singapore/transit/stop_snapping_report.json"),
        "trips_by_stop": str(ROOT / "data/singapore/transit/trips_by_stop.json"),
        "activity_nodes": str(ROOT / "data/singapore/transit/activity_nodes.json"),
    }
    idx = SupplyIndex(SUPPLY)
    cfg = load_yaml(ROOT / "configs" / "generation_v0_1.yaml")
    personas = PersonaGenerator(seed=2026, config=cfg).generate(args.max_n)
    trips = TripGenerator(seed=2026, config=cfg).generate(args.max_n)
    activity_nodes = json.loads(Path(SUPPLY["activity_nodes"]).read_text(encoding="utf-8"))
    ctx = DynamicContext(context_id="bench", weather=Weather(condition="clear", intensity=0.0),
                         road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
                         road_disruption=False, fare_multiplier=1.0,
                         parking_cost_multiplier=1.0, congestion_charge=0.0)
    import hashlib
    ods = []
    for persona, trip in zip(personas, trips):
        h = int(hashlib.sha256(persona.persona_id.encode()).hexdigest(), 16)
        home = activity_nodes[h % len(activity_nodes)]["node"]
        dest = activity_nodes[(h + 7919) % len(activity_nodes)]["node"]
        ods.append((home, dest, float(trip.desired_departure_min) * 60.0))

    t0 = time.perf_counter()
    accs = [plan_accessibility(idx, o, d, dep) for (o, d, dep) in ods]
    t_recompute = time.perf_counter() - t0
    print(f"[cached] precompute {len(accs)} OD accessibility vectors: {t_recompute:.1f} s "
          f"({t_recompute/len(accs)*1e3:.1f} ms/state)")

    cache = ROOT / "outputs" / "pipeline_benchmark" / "od_accessibility_cache.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with cache.open("wb") as fh:
        pickle.dump(list(zip(ods, accs)), fh)
    t_dump = time.perf_counter() - t0
    t0 = time.perf_counter()
    with cache.open("rb") as fh:
        loaded = pickle.load(fh)
    t_load = time.perf_counter() - t0
    print(f"[cached] pickle dump {t_dump:.2f} s ({cache.stat().st_size/1e6:.1f} MB), "
          f"load {t_load:.2f} s -> cache-hit path saves {t_recompute - t_load:.0f} s per scenario rerun")

    # optimized path: build alternatives directly from cached acc (no routing),
    # with the SAME mode-travel-time memoization that run_phase_c's
    # make_shared_idx applies in production.
    _orig_mtt = idx.mode_travel_time
    _tt_cache = {}
    _SENTINEL = object()

    def _mtt(mode, src, dst):
        key = (mode, src, dst)
        val = _tt_cache.get(key, _SENTINEL)
        if val is _SENTINEL:
            val = _orig_mtt(mode, src, dst)
            _tt_cache[key] = val
        return val

    idx.mode_travel_time = _mtt
    t0 = time.perf_counter()
    n = 0
    for (persona, trip), (o, d, dep), acc in zip(zip(personas, trips), ods, loaded):
        build_real_alternatives(persona, trip, ctx, idx, o, d, acc[1])
        n += 1
    t_build = time.perf_counter() - t0
    print(f"[cached] optimized feature path (disk acc cache + memoized mode TT): "
          f"{t_build:.2f} s, {t_build/n*1e3:.3f} ms/state "
          f"(vs {t_recompute/n*1e3:.1f} ms/state recompute = {t_recompute/t_build:.0f}x)")

    # second pass over the SAME ODs = all-cache-hit path (second scenario/run)
    t0 = time.perf_counter()
    for (persona, trip), (o, d, dep), acc in zip(zip(personas, trips), ods, loaded):
        build_real_alternatives(persona, trip, ctx, idx, o, d, acc[1])
    t_hit = time.perf_counter() - t0
    print(f"[cached] SECOND pass (all caches warm, same ODs): {t_hit:.2f} s, "
          f"{t_hit/n*1e3:.3f} ms/state "
          f"(vs {t_recompute/n*1e3:.1f} ms/state cold = {t_recompute/t_hit:.0f}x faster)")
    _write_csv(args.csv, {
        "method": "e2e_feature_cold_first_pass", "batch_size": 1, "num_decisions": len(accs),
        "total_time": t_build, "ms_per_decision": t_build / len(accs) * 1e3,
        "decisions_per_second": len(accs) / t_build, "cold_or_warm": "cold"})
    _write_csv(args.csv, {
        "method": "e2e_feature_cache_hit_second_pass", "batch_size": 1, "num_decisions": len(accs),
        "total_time": t_hit + t_load, "ms_per_decision": (t_hit + t_load) / len(accs) * 1e3,
        "decisions_per_second": len(accs) / (t_hit + t_load), "cold_or_warm": "warm"})
    _write_csv(args.csv, {
        "method": "e2e_accessibility_recompute", "batch_size": 1, "num_decisions": len(accs),
        "total_time": t_recompute, "ms_per_decision": t_recompute / len(accs) * 1e3,
        "decisions_per_second": len(accs) / t_recompute, "cold_or_warm": "warm"})
    _write_csv(args.csv, {
        "method": "e2e_accessibility_disk_cache_hit", "batch_size": 1, "num_decisions": len(accs),
        "total_time": t_build + t_load, "ms_per_decision": (t_build + t_load) / len(accs) * 1e3,
        "decisions_per_second": len(accs) / (t_build + t_load), "cold_or_warm": "warm"})


def mode_e2e1k(args):
    import torch
    from traveler_distillation.accessibility.gtfs_accessibility import SupplyIndex, plan_accessibility
    from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives
    from traveler_distillation.config import load_yaml
    from traveler_distillation.generators import PersonaGenerator, TripGenerator
    from traveler_distillation.schemas.state import UniversalTravelerState

    SUPPLY = {
        "network": str(ROOT / "data/singapore/transit/network_with_transit.xml"),
        "schedule": str(ROOT / "data/singapore/transit/transitSchedule.xml"),
        "vehicles": str(ROOT / "data/singapore/transit/transitVehicles.xml"),
        "stops": str(ROOT / "data/singapore/transit/prep_stops.jsonl"),
        "snapping": str(ROOT / "data/singapore/transit/stop_snapping_report.json"),
        "trips_by_stop": str(ROOT / "data/singapore/transit/trips_by_stop.json"),
        "activity_nodes": str(ROOT / "data/singapore/transit/activity_nodes.json"),
    }
    t0 = time.perf_counter()
    idx = SupplyIndex(SUPPLY)
    t_supply = time.perf_counter() - t0
    print(f"[e2e1k] SupplyIndex build (graphs + stops + trips index): {t_supply:.1f} s")

    adapter, _ = _load_adapter(threads=1)
    cfg = load_yaml(ROOT / "configs" / "generation_v0_1.yaml")
    personas = PersonaGenerator(seed=2026, config=cfg).generate(1000)
    trips = TripGenerator(seed=2026, config=cfg).generate(1000)

    activity_nodes = json.loads(Path(SUPPLY["activity_nodes"]).read_text(encoding="utf-8"))
    import hashlib
    t_fac = t_dec = t_legs = t_xml = 0.0
    n_states = 0
    for i, (persona, trip) in enumerate(zip(personas, trips)):
        h = int(hashlib.sha256(persona.persona_id.encode()).hexdigest(), 16)
        home = activity_nodes[h % len(activity_nodes)]["node"]
        dest = activity_nodes[(h + 7919) % len(activity_nodes)]["node"]
        dep_sec = float(trip.desired_departure_min) * 60.0
        t0 = time.perf_counter()
        acc = plan_accessibility(idx, home, dest, dep_sec)
        t_fac += time.perf_counter() - t0
        from traveler_distillation.schemas.context import DynamicContext, Weather
        ctx = DynamicContext(context_id="bench", weather=Weather(condition="clear", intensity=0.0),
                             road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
                             road_disruption=False, fare_multiplier=1.0,
                             parking_cost_multiplier=1.0, congestion_charge=0.0)
        alts = build_real_alternatives(persona, trip, ctx, idx, home, dest, acc)
        state = UniversalTravelerState(persona=persona, trip=trip, context=ctx, alternatives=alts)
        t0 = time.perf_counter()
        adapter.decide(state)
        t_dec += time.perf_counter() - t0
        n_states += 1

    print(f"[e2e1k] N={n_states} real states:")
    print(f"  accessibility (plan_accessibility): {t_fac:.1f} s total, {t_fac/n_states*1e3:.1f} ms/state, share {t_fac/(t_fac+t_dec)*100:.1f}%")
    print(f"  decide (encode+forward):            {t_dec:.2f} s total, {t_dec/n_states*1e3:.3f} ms/state")
    _write_csv(args.csv, {
        "method": "e2e_accessibility_real", "batch_size": 1, "num_decisions": n_states,
        "total_time": t_fac, "ms_per_decision": t_fac / n_states * 1e3,
        "decisions_per_second": n_states / t_fac, "cold_or_warm": "warm"})
    _write_csv(args.csv, {
        "method": "e2e_decide_real", "batch_size": 1, "num_decisions": n_states,
        "total_time": t_dec, "ms_per_decision": t_dec / n_states * 1e3,
        "decisions_per_second": n_states / t_dec, "cold_or_warm": "warm"})
    _write_csv(args.csv, {
        "method": "e2e_supply_index_build", "batch_size": "", "num_decisions": 1,
        "total_time": t_supply, "ms_per_decision": t_supply * 1e3,
        "decisions_per_second": "", "cold_or_warm": "cold"})


def mode_ipc(args):
    """Prototype: persistent Python student worker over localhost TCP.

    Protocol: 4-byte big-endian length prefix + UTF-8 JSON request
    {persona, trip, context, alternatives} -> response {mode, probs, shift}.
    Measures per-request latency of a runtime-decision architecture
    (NOT used by the current pipeline; prototype only).
    """
    import torch
    from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter

    states = _load_states(limit=200)
    payloads = [json.dumps({
        "persona": s.persona.model_dump(), "trip": s.trip.model_dump(),
        "context": s.context.model_dump(), "alternatives": [a.model_dump() for a in s.alternatives],
    }).encode("utf-8") for s in states]
    avg_payload = sum(len(p) for p in payloads) / len(payloads)
    print(f"[ipc] average request payload: {avg_payload:.0f} bytes")

    server_code = f'''
import json, socket, struct, sys
sys.path.insert(0, {str(ROOT / "src")!r})
import torch
from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter
from traveler_distillation.schemas.persona import Persona
from traveler_distillation.schemas.trip import Trip
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.schemas.alternative import TravelAlternative
from traveler_distillation.schemas.state import UniversalTravelerState

adapter = S8MATSimAdapter({str(ROOT / "releases" / "s9_supply_aware_v2" / "checkpoint" / "model.pt")!r})
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", {args.port}))
srv.listen(8)
print("READY", flush=True)
conn, _ = srv.accept()
f = conn.makefile("rb")
while True:
    hdr = f.read(4)
    if len(hdr) < 4:
        break
    (n,) = struct.unpack(">I", hdr)
    d = json.loads(f.read(n).decode("utf-8"))
    state = UniversalTravelerState(
        persona=Persona(**d["persona"]), trip=Trip(**d["trip"]),
        context=DynamicContext(**{{k: (Weather(**v) if k == "weather" else v) for k, v in d["context"].items()}}),
        alternatives=[TravelAlternative(**a) for a in d["alternatives"]],
    )
    dec = adapter.decide(state)
    resp = json.dumps({{"mode": dec["mode"], "probs": dec["mode_probabilities"],
                        "shift": dec["departure_time_shift_min"]}}).encode("utf-8")
    conn.sendall(struct.pack(">I", len(resp)) + resp)
'''
    proc = subprocess.Popen([sys.executable, "-c", server_code], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, cwd=str(ROOT))
    line = proc.stdout.readline().strip()
    assert line == "READY", line
    time.sleep(0.2)

    sock = socket.create_connection(("127.0.0.1", args.port), timeout=10)
    lat = []
    for p in payloads[100:]:  # warm requests excluded from stats
        t0 = time.perf_counter()
        sock.sendall(struct.pack(">I", len(p)) + p)
        hdr = sock.recv(4)
        (n,) = struct.unpack(">I", hdr)
        buf = b""
        while len(buf) < n:
            buf += sock.recv(n - len(buf))
        lat.append(time.perf_counter() - t0)
    sock.close()
    proc.terminate()
    st = _pstats(lat)
    print(f"[ipc] persistent TCP worker (per request, {len(lat)} warm requests): "
          f"mean {st['mean']*1e3:.2f} ms, median {st['median']*1e3:.2f} ms, p95 {st['p95']*1e3:.2f} ms")
    _write_csv(args.csv, {
        "method": "ipc_persistent_worker", "batch_size": 1, "num_decisions": len(lat),
        "total_time": st["mean"] * len(lat), "ms_per_decision": st["mean"] * 1e3,
        "decisions_per_second": 1.0 / st["mean"], "cold_or_warm": "warm"})


def mode_onnx(args):
    import torch
    adapter, _ = _load_adapter(threads=1)
    states = _load_states(limit=1000)
    feats, batch = _encode_batch(adapter, states)

    with torch.no_grad():
        ref = adapter.model(batch)

    import onnxruntime as ort
    f = ROOT / "outputs" / "pipeline_benchmark" / "s9.onnx"
    f.parent.mkdir(parents=True, exist_ok=True)

    class _OnnxWrapper(torch.nn.Module):
        """Export-only adapter: tensor args -> model's dict forward."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, global_cat, global_num, alt_mode_idx, alt_num, alt_mask):
            out = self.m({"global_cat": global_cat, "global_num": global_num,
                          "alt_mode_idx": alt_mode_idx, "alt_num": alt_num,
                          "alt_mask": alt_mask})
            return out["mode_probabilities"], out["departure_time_shift_min"], out["utilities"]

    torch.onnx.export(
        _OnnxWrapper(adapter.model),
        (batch["global_cat"], batch["global_num"], batch["alt_mode_idx"], batch["alt_num"], batch["alt_mask"]),
        str(f),
        input_names=["global_cat", "global_num", "alt_mode_idx", "alt_num", "alt_mask"],
        output_names=["mode_probabilities", "departure_time_shift_min", "utilities"],
        dynamic_axes={"global_cat": {0: "B"}, "global_num": {0: "B"}, "alt_mode_idx": {0: "B"},
                      "alt_num": {0: "B"}, "alt_mask": {0: "B"},
                      "mode_probabilities": {0: "B"}, "departure_time_shift_min": {0: "B"},
                      "utilities": {0: "B"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"[onnx] exported {f} ({f.stat().st_size} bytes)")

    sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
    inputs = {k: v.numpy() for k, v in
              zip(("global_cat", "global_num", "alt_mode_idx", "alt_num", "alt_mask"),
                  (batch["global_cat"], batch["global_num"], batch["alt_mode_idx"], batch["alt_num"], batch["alt_mask"]))}
    out = sess.run(None, inputs)
    p_ort = out[0]
    p_torch = ref["mode_probabilities"].cpu().numpy()
    d_ort = out[1]
    d_torch = ref["departure_time_shift_min"].cpu().numpy()
    max_abs_p = float((abs(p_ort - p_torch)).max())
    max_abs_d = float((abs(d_ort - d_torch)).max())
    same_mode = sum(1 for i in range(len(states))
                    if int(p_ort[i].argmax()) == int(p_torch[i].argmax()))
    print(f"[onnx] parity vs torch on {len(states)} states: max|Δprob| {max_abs_p:.2e}, "
          f"max|Δshift| {max_abs_d:.2e} min, argmax match {same_mode}/{len(states)}")

    def one_onnx():
        return sess.run(None, inputs)
    st = _timeit(one_onnx, repeats=50, warm=5)
    per = st["mean"] / len(states) * 1e3
    print(f"[onnx] ORT CPU batch=1000: mean {st['mean']*1e3:.1f} ms/batch, {per:.4f} ms/state, "
          f"{len(states)/st['mean']:,.0f} decisions/s")
    _write_csv(args.csv, {
        "method": "onnxruntime_cpu", "batch_size": len(states), "num_decisions": len(states),
        "total_time": st["mean"], "ms_per_decision": per,
        "decisions_per_second": len(states) / st["mean"], "cold_or_warm": "warm"})


def mode_correctness(args):
    adapter, _ = _load_adapter(threads=1)
    states = _load_states(limit=10000)
    ref = [adapter.decide(s) for s in states]
    _, batch = _encode_batch(adapter, states)
    import torch
    with torch.no_grad():
        out = adapter.model(batch)
    dec = _decode_batch(adapter, states, out)
    same_mode = sum(1 for a, b in zip(ref, dec) if a["mode"] == b["mode"])
    max_abs = max(abs(a["departure_time_shift_min"] - b["departure_time_shift_min"]) for a, b in zip(ref, dec))
    print(f"[correctness] single decide() vs batched on {len(states)} real states: "
          f"mode match {same_mode}/{len(states)} ({100*same_mode/len(states):.4f}%), "
          f"max |Δshift| {max_abs:.2e} min")

    manifest = json.loads((ROOT / "outputs" / "singapore_phase_c_s9" / "C0_baseline" /
                           "adapter_manifest.json").read_text(encoding="utf-8"))["decisions"]
    match = sum(1 for i in range(min(len(states), len(manifest)))
                if dec[i]["mode"] == manifest[i]["student_mode"])
    print(f"[correctness] batched decisions vs frozen C0 manifest (first {len(states)}): "
          f"mode match {match}/{min(len(states), len(manifest))} ({100*match/min(len(states), len(manifest)):.4f}%)")
    _write_csv(args.csv, {
        "method": "correctness_batch_vs_single", "batch_size": 1024, "num_decisions": len(states),
        "total_time": "", "ms_per_decision": "", "decisions_per_second": "",
        "cold_or_warm": "warm"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["profile", "batch", "scale", "e2e1k", "ipc", "onnx", "correctness", "cached"])
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "pipeline_benchmark.csv"))
    ap.add_argument("--max-n", type=int, default=100000)
    ap.add_argument("--port", type=int, default=38881)
    args = ap.parse_args()
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    {
        "profile": mode_profile,
        "batch": mode_batch,
        "scale": mode_scale,
        "e2e1k": mode_e2e1k,
        "ipc": mode_ipc,
        "onnx": mode_onnx,
        "correctness": mode_correctness,
        "cached": mode_cached,
    }[args.mode](args)


if __name__ == "__main__":
    main()
