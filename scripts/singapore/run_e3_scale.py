#!/usr/bin/env python
"""E3 (TRC_AIT_5_EXPERIMENT_PLAN §E3; design TRC_AIT_5_E3_POPULATION_SCALABILITY_DESIGN.md)
population-scalability runner — frozen S9 + frozen Singapore supply, C0 only.

One invocation = one (N, repeat) run:
  setup  (checkpoint SHA256 + SupplyIndex + shared factory)          [T_setup]
  build_real_scenario with timed alt_factory + timed decision_fn
         -> T_factory / T_decide (per-state arrays), T_build, T_residual
  MATSim via Popen with 1 s peak-working-set sampling, wall clock,
         design timeout (10,800 s for N<=20k; 21,600 s for N>=50k)   [T_matsim]
  event parse (reused _parse_events_full)                            [T_parse]
  gates G1 / G2a / G2b / G3' (10k only) / G4 -> e3_result.json

Reuses run_phase_c.py machinery by import (SUPPLY, make_context,
make_shared_idx, make_shared_factory, _CP, _parse_events_full,
_manifest_stats, load_link_attrs, sha256). run_phase_c.py and
adapter.py default paths are NOT modified; timing wrappers use the
official alt_factory / decision_fn extension points (byte-identical
decisions, verified by G1).

Usage:
    python scripts/singapore/run_e3_scale.py --num-agents 1000 --repeat 0
    python scripts/singapore/run_e3_scale.py --num-agents 10000
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import PersonaGenerator, TripGenerator  # noqa: E402
from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter  # noqa: E402
from traveler_distillation.student.release_guard import assert_not_frozen_output  # noqa: E402

from run_phase_c import (  # noqa: E402
    SUPPLY,
    _CP,
    _manifest_stats,
    _parse_events_full,
    load_link_attrs,
    make_context,
    make_shared_factory,
    make_shared_idx,
    sha256,
)

FROZEN_S9_SHA256 = "6af79b44bc699c00cc8e913da15043a43398829fdc8bbb4dad2ddcb8e11d31e6"
POP_SEED = 2026
GEN_CONFIG = "configs/generation_v0_1.yaml"
FROZEN_C0_MANIFEST = "outputs/singapore_phase_c_s9/C0_baseline/adapter_manifest.json"
FROZEN_C0_RESULT = "outputs/singapore_phase_c_s9/C0_baseline/phase_c_result.json"
MANIFEST_FIELDS = [
    "persona_id", "trip_id", "student_mode", "outbound_mode", "return_mode",
    "departure_shift_min", "departure_min", "home_node", "dest_node",
    "outbound", "return",
]

# ---------------------------------------------------------------- memory (ctypes)
class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_psapi = ctypes.WinDLL("psapi")
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
_k32 = ctypes.WinDLL("kernel32")
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _peak_ws(handle: int) -> int:
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not _psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
        return -1
    return int(pmc.PeakWorkingSetSize)


def _self_peak_ws() -> int:
    return _peak_ws(_k32.GetCurrentProcess())


# ---------------------------------------------------------------- timing wrappers
def _timed_factory(base_factory):
    times: list[float] = []

    def factory(persona, trip, ctx, origin, dest):
        t0 = time.perf_counter()
        alts = base_factory(persona, trip, ctx, origin, dest)
        times.append(time.perf_counter() - t0)
        return alts

    return factory, times


def _timed_decide(adapter: S8MATSimAdapter):
    times: list[float] = []

    def decide(state):
        t0 = time.perf_counter()
        d = adapter.decide(state)
        times.append(time.perf_counter() - t0)
        return d

    return decide, times


def _perf_stats(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "total_s": round(sum(s), 4),
        "mean_ms": round(statistics.mean(s) * 1000, 4),
        "p50_ms": round(s[n // 2] * 1000, 4),
        "p95_ms": round(s[int(n * 0.95)] * 1000, 4),
        "min_ms": round(s[0] * 1000, 4),
        "max_ms": round(s[-1] * 1000, 4),
        "first_ms": round(s[0] * 1000, 4) if n else None,
    }


# ---------------------------------------------------------------- matsim (sampled)
def _run_matsim_sampled(sdir: Path, timeout_s: int) -> dict:
    """Popen-based MATSim run with 1 s peak-working-set sampling."""
    log_path = sdir / "java_run.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", errors="replace") as out_f:
        proc = subprocess.Popen(
            ["java", "-Xmx6g", "-cp", _CP, "RunMatsimPreloaded", "config.xml"],
            cwd=str(sdir), stdout=out_f, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + timeout_s
        peak = 0
        samples = 0
        timed_out = False
        while True:
            code = proc.poll()
            h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, proc.pid)
            if h:
                ws = _peak_ws(h)
                _k32.CloseHandle(h)
                if ws > peak:
                    peak = ws
            samples += 1
            if code is not None:
                break
            if time.monotonic() > deadline:
                proc.kill()
                code = proc.wait()
                timed_out = True
                break
            time.sleep(1.0)
    wall = time.perf_counter() - t0
    tail = ""
    warn_log = sdir / "output" / "logfileWarningsErrors.log"
    if warn_log.exists():
        tail = "\n".join(warn_log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:])
    return {
        "exit_code": code, "wall_s": round(wall, 2),
        "peak_ws_bytes": peak, "samples": samples,
        "sample_interval_s": 1.0, "timed_out": timed_out, "log_tail": tail,
    }


# ---------------------------------------------------------------- gates
def _g1_check(built: list[dict], frozen: list[dict], n: int,
              built_pop_xml: Path | None, frozen_pop_xml: Path | None) -> dict:
    k = min(n, len(frozen))
    mismatches = []
    for i in range(k):
        b, f = built[i], frozen[i]
        for key in MANIFEST_FIELDS:
            if b.get(key) != f.get(key):
                mismatches.append({"index": i, "field": key,
                                   "built": str(b.get(key))[:120],
                                   "frozen": str(f.get(key))[:120]})
                break
            if len(mismatches) >= 10:
                break
        if len(mismatches) >= 10:
            break
    pop_sha_match = None
    if n == 10_000 and built_pop_xml is not None and frozen_pop_xml is not None:
        pop_sha_match = sha256(built_pop_xml) == sha256(frozen_pop_xml)
    return {
        "gate": "G1",
        "built_n": len(built),
        "compared_n": k,
        "mismatches": len(mismatches),
        "first_mismatches": mismatches,
        "population_xml_sha256_match": pop_sha_match,
        "pass": len(mismatches) == 0 and (pop_sha_match is not False),
    }


def _g2a_check(m: dict, run: dict) -> dict:
    checks = {
        "matsim_exit_0": run.get("matsim_exit_code") == 0,
        "not_timed_out": not run.get("timeout_hit", False),
        "four_modes_executed": {"car", "pt", "walk", "bike"} <= set(m.get("leg_departures", {})),
        "pt_alightings_le_boardings": m.get("pt_alightings", 0) <= m.get("pt_boardings", 0)
        or not m.get("pt_boardings"),
        "legs_executed": bool(m.get("leg_departures")),
    }
    return {"gate": "G2a", "checks": checks,
            "pass": all(v for k, v in checks.items())}


def _g2b_check(m: dict, n: int, frozen_metrics: dict | None) -> dict:
    boardings = max(1, m.get("pt_boardings", 0))
    rate = m.get("stuck_persons", 0) / boardings
    rate_ok = rate <= 0.05
    failed_ok = None
    if frozen_metrics and frozen_metrics.get("failed_trips") is not None:
        # PER-AGENT comparison (design §8.1 + S7 criterion 4: "failed trips/agent <= 2x 10k");
        # absolute counts scale with N by construction and must not be gated.
        frozen_rate = frozen_metrics["failed_trips"] / 10_000
        failed_ok = (m.get("failed_trips", 0) / n) <= 2 * frozen_rate
    hard = n <= 20_000
    return {
        "gate": "G2b",
        "stuck_rate_per_pt_boarding": round(rate, 4),
        "stuck_rate_le_5pct": rate_ok,
        "failed_trips_per_agent": round(m.get("failed_trips", 0) / n, 4),
        "failed_per_agent_le_2x_frozen": failed_ok,
        "hard_gate": hard,
        "note": f"rate gate {'hard' if hard else 'REPORT-ONLY'} at N={n} "
                f"(design §8.1: hard for N<=20k); frozen 10k stuck rate 3.8%; "
                f"failed check is per-agent (revision 5)",
        "pass": (not hard) or (rate_ok and failed_ok is not False),
    }


def _g3prime_check(m: dict, frozen_metrics: dict | None) -> dict | None:
    if frozen_metrics is None:
        return None
    mismatches = {k: (m.get(k), v) for k, v in frozen_metrics.items() if m.get(k) != v}
    return {"gate": "G3'", "compared_fields": len(frozen_metrics),
            "mismatches": mismatches, "pass": len(mismatches) == 0}


def _g4_check(decide_s: list[float]) -> dict:
    if len(decide_s) < 2:
        return {"gate": "G4", "pass": False, "note": "insufficient samples"}
    body = decide_s[1:]
    med = statistics.median(body)
    thr = 10 * med
    stalls = [x for x in body if x > thr]
    pass_ = (len(stalls) == 0) or (len(stalls) == 1 and stalls[0] <= 2.0)
    return {
        "gate": "G4",
        "first_state_ms": round(decide_s[0] * 1000, 4),
        "body_median_ms": round(med * 1000, 4),
        "body_max_ms": round(max(body) * 1000, 4),
        "stall_count": len(stalls),
        "stall_durations_ms": [round(x * 1000, 2) for x in stalls],
        "pass": pass_,
        "note": "design v0.1 revision 4: first state = cold-start warmup; exactly ONE stall "
                "<= 2s (OS-level, 500x P99) is documented and does not invalidate the run; "
                ">= 2 stalls or any stall > 2s -> stop and investigate machine contention",
    }


# ---------------------------------------------------------------- main
def run_one(args: argparse.Namespace) -> int:
    n = args.num_agents
    root = ROOT
    ckpt = root / args.checkpoint
    digest = sha256(ckpt)
    if digest != FROZEN_S9_SHA256:
        raise SystemExit(f"checkpoint SHA256 {digest} != frozen S9 {FROZEN_S9_SHA256} ({ckpt})")

    out_root = assert_not_frozen_output(root / args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    dirname = f"N{n:05d}" + (f"_r{args.repeat}" if args.repeat is not None else "")
    out = out_root / dirname
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists and is non-empty; use --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    import platform
    import torch
    import datetime as _dt
    t_start = time.perf_counter()
    gen_cfg = load_yaml(str(root / GEN_CONFIG))
    personas = PersonaGenerator(seed=POP_SEED, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=POP_SEED, config=gen_cfg).generate(n)
    trips_per_persona = [[t] for t in trips]
    adapter = S8MATSimAdapter(ckpt)
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}
    context = make_context("C0_baseline")

    shared_idx = make_shared_idx(supply)
    factory_base = make_shared_factory(shared_idx, context)
    factory, factory_s = _timed_factory(factory_base)
    decide, decide_s = _timed_decide(adapter)
    setup_s = time.perf_counter() - t_start
    p0 = _self_peak_ws()

    t_b0 = time.perf_counter()
    manifest = adapter.build_real_scenario(
        personas, trips, context, out,
        network_path=supply["network"], schedule_path=supply["schedule"],
        vehicles_path=supply["vehicles"], stops_path=supply["stops"],
        snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
        activity_nodes_path=supply["activity_nodes"],
        trips_per_persona=trips_per_persona,
        flow_capacity_factor=0.3, storage_capacity_factor=0.3,
        alt_factory=factory, decision_fn=decide,
    )
    build_s = time.perf_counter() - t_b0
    p1 = _self_peak_ws()
    factory_total = sum(factory_s)
    decide_total = sum(decide_s)
    residual_s = build_s - factory_total - decide_total

    manifest_file = out / "adapter_manifest.json"
    manifest_json = json.loads(manifest_file.read_text(encoding="utf-8"))
    built_decisions = manifest_json["decisions"]
    fallback_counts = manifest_json.get("fallback_counts", {})
    intended_pt = sum(1 for d in built_decisions if d["student_mode"] == "pt")
    pt_validity = (1 - fallback_counts.get("pt_fallback_walk", 0) / max(1, intended_pt))

    frozen_manifest = json.loads(
        (root / FROZEN_C0_MANIFEST).read_text(encoding="utf-8"))
    frozen_decisions = frozen_manifest["decisions"]
    g1 = _g1_check(built_decisions, frozen_decisions, n,
                   out / "population.xml",
                   root / "outputs" / "singapore_phase_c_s9" / "C0_baseline" / "population.xml")
    g4 = _g4_check(decide_s)

    matsim_run = None
    metrics = None
    parse_s = None
    frozen_metrics = None
    timeout_s = 10_800 if n <= 20_000 else 21_600
    if not args.skip_matsim:
        links = load_link_attrs(str(root / supply["network"]))
        matsim_run = _run_matsim_sampled(out, timeout_s)
        t_p0 = time.perf_counter()
        metrics = _parse_events_full(out, links)
        parse_s = time.perf_counter() - t_p0
        metrics["leg_departures"] = dict(metrics.pop("leg_departures"))
        frozen_result = json.loads((root / FROZEN_C0_RESULT).read_text(encoding="utf-8"))
        frozen_metrics = frozen_result.get("metrics")

    g2a = _g2a_check(metrics or {}, {"matsim_exit_code": None if matsim_run is None else matsim_run["exit_code"],
                                     "timeout_hit": False if matsim_run is None else matsim_run["timed_out"]})
    if matsim_run is None:
        g2a = {"gate": "G2a", "pass": False, "note": "skip-matsim (debug only, not evidence)"}
    g2b = _g2b_check(metrics or {}, n, frozen_metrics)
    g3p = _g3prime_check(metrics or {}, frozen_metrics) if (n == 10_000 and metrics is not None) else None
    if n == 10_000 and metrics is None:
        g3p = {"gate": "G3'", "pass": False, "note": "skip-matsim (debug only)"}

    gates = {"g1": g1, "g2a": g2a, "g2b": g2b, "g3prime": g3p, "g4": g4,
             "g5": None}
    overall = g1["pass"] and g2a["pass"] and g2b["pass"] and g4["pass"] and (g3p is None or g3p["pass"])

    events_path = out / "output" / "ITERS" / "it.0" / "0.events.xml.zst"
    pop_xml = out / "population.xml"
    record = {
        "experiment": "E3",
        "design": "TRC_AIT_5_E3_POPULATION_SCALABILITY_DESIGN.md v0.1",
        "scenario": "C0_baseline",
        "n_agents": n,
        "repeat": args.repeat,
        "population_seed": POP_SEED,
        "prefix_rule": "nested prefix of seed-2026 stream (E2 §1 prefix property)",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": digest,
        "gen_config": GEN_CONFIG,
        "gen_config_sha256": sha256(root / GEN_CONFIG),
        "supply_sha256": {k: sha256(root / v) for k, v in SUPPLY.items()},
        "environment": {
            "platform": platform.platform(),
            "cpu_logical": 24,
            "torch_threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "java": "OpenJDK 25.0.4",
            "matsim": "MATSim 2026.0 (matsim_rel/matsim-2026.0.jar)",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "dedicated": "sequential execution policy; no other E3 job concurrent",
            "contention_note": args.contention_note or "none declared",
        },
        "timing": {
            "setup_s": round(setup_s, 2),
            "setup_covers": "persona/trip gen + adapter load + SupplyIndex + shared factory",
            "factory_stats": _perf_stats(factory_s),
            "decide_stats": _perf_stats(decide_s),
            "factory_s": [round(x, 6) for x in factory_s],
            "decide_s": [round(x, 6) for x in decide_s],
            "build_s": round(build_s, 2),
            "residual_s": round(residual_s, 2),
            "residual_note": "T_build - T_factory - T_decide (leg routing + population.xml/manifest write)",
            "matsim_wall_s": None if matsim_run is None else matsim_run["wall_s"],
            "matsim_exit_code": None if matsim_run is None else matsim_run["exit_code"],
            "matsim_timeout_s": None if matsim_run is None else timeout_s,
            "timeout_hit": False if matsim_run is None else matsim_run["timed_out"],
            "parse_s": None if parse_s is None else round(parse_s, 2),
        },
        "memory": {
            "py_peak_ws_bytes": {"p0_after_setup": p0, "p1_after_build": p1, "delta": p1 - p0},
            "py_peak_ws_mb": {"p0_after_setup": round(p0 / 2**20, 1),
                              "p1_after_build": round(p1 / 2**20, 1),
                              "delta": round((p1 - p0) / 2**20, 1)},
            "java_peak_ws_bytes": None if matsim_run is None else matsim_run["peak_ws_bytes"],
            "java_peak_ws_mb": None if matsim_run is None else round(matsim_run["peak_ws_bytes"] / 2**20, 1),
            "java_samples": None if matsim_run is None else matsim_run["samples"],
            "java_sample_interval_s": None if matsim_run is None else matsim_run["sample_interval_s"],
            "java_xmx": "6g",
        },
        "artifacts": {
            "population_xml_bytes": pop_xml.stat().st_size if pop_xml.exists() else None,
            "population_xml_sha256": sha256(pop_xml) if pop_xml.exists() else None,
            "adapter_manifest_bytes": manifest_file.stat().st_size,
            "events_zst_bytes": events_path.stat().st_size if events_path.exists() else None,
        },
        "decisions": _manifest_stats(manifest),
        "pt_routing_validity": {
            "intended_pt": intended_pt,
            "pt_fallback_walk": fallback_counts.get("pt_fallback_walk", 0),
            "validity": round(pt_validity, 4),
            "fallback_counts": fallback_counts,
        },
        "metrics": metrics,
        "gates": gates,
        "overall_pass": overall,
    }
    if matsim_run is not None and matsim_run["log_tail"]:
        record["log_tail"] = matsim_run["log_tail"][-2000:]
    (out / "e3_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "dir": str(out), "n": n, "repeat": args.repeat,
        "setup_s": record["timing"]["setup_s"], "build_s": record["timing"]["build_s"],
        "decide_mean_ms": record["timing"]["decide_stats"].get("mean_ms"),
        "factory_total_s": round(factory_total, 2), "decide_total_s": round(decide_total, 2),
        "matsim_wall_s": record["timing"]["matsim_wall_s"],
        "parse_s": record["timing"]["parse_s"],
        "py_peak_mb": record["memory"]["py_peak_ws_mb"],
        "java_peak_mb": record["memory"]["java_peak_ws_mb"],
        "gates": {k: (v.get("pass") if isinstance(v, dict) else v) for k, v in gates.items()},
        "overall_pass": overall,
    }, ensure_ascii=False))
    return 0 if overall else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-agents", type=int, required=True)
    ap.add_argument("--repeat", type=int, default=None,
                    help="repeat index for the 1k pilot (output dir suffix _rN)")
    ap.add_argument("--checkpoint", default="releases/s9_supply_aware_v2/checkpoint/model.pt")
    ap.add_argument("--output", default="outputs/e3_scale")
    ap.add_argument("--skip-matsim", action="store_true",
                    help="build-only debug (gates G2a/G2b/G3' not applicable; never evidence)")
    ap.add_argument("--contention-note", default=None,
                    help="honest record of any concurrent machine workload during this run")
    ap.add_argument("--force", action="store_true",
                    help="delete an existing non-empty output dir before running")
    args = ap.parse_args()
    if args.force:
        import shutil
        out_root = ROOT / args.output
        dirname = f"N{args.num_agents:05d}" + (f"_r{args.repeat}" if args.repeat is not None else "")
        target = out_root / dirname
        if target.exists():
            shutil.rmtree(target)
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
