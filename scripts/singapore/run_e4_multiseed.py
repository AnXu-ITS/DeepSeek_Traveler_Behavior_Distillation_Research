#!/usr/bin/env python
"""E4 (TRC_AIT_5_EXPERIMENT_PLAN §E4; design TRC_AIT_5_E4_MULTISEED_ROBUSTNESS_DESIGN.md)
multi-seed robustness runner — frozen S9 + frozen Singapore supply, seeds 42 / 7.

One invocation = one seed, all requested scenarios in one process (shared
SupplyIndex + per-scenario factories, same structure as run_phase_c.main()).
Seed 2026 is NEVER run here — its column comes from the frozen Phase C files
(outputs/singapore_phase_c_s9/<scenario>/phase_c_result.json, read only by
make_e4_report.py).

Reuses run_phase_c.py machinery by import (SUPPLY, SCENARIOS, make_context,
make_shared_idx, make_shared_factory, _run_matsim, _parse_events_full,
_manifest_stats, _evaluate_gate, load_link_attrs, sha256). run_phase_c.py and
adapter.py default paths are NOT modified; decisions use decision_fn=None
(official default, identical to Phase C). The per-scenario record writes the
TRUE population seed and TRUE checkpoint identity (the frozen Phase C records
carry the stale S8-path metadata bug documented in E2 v0.2 revision 1).

G1 determinism gate: run `--scenarios C0_baseline --gate-repeat` after the
seed-42 evidence run; it writes to outputs/e4_multiseed/_gate/seed42_C0_r1/ and
asserts byte-identical population.xml / adapter_manifest.json plus identical
decision stats and metrics vs the evidence run (seed42/C0_baseline/e4_result.json).
The r1 run is a gate artifact, NOT evidence.

Usage:
    python scripts/singapore/run_e4_multiseed.py --seed 42 --contention-note "E5 Helsinki 并发"
    python scripts/singapore/run_e4_multiseed.py --seed 42 --scenarios C0_baseline --gate-repeat
    python scripts/singapore/run_e4_multiseed.py --seed 7 --contention-note "E5 Helsinki 并发"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import PersonaGenerator, TripGenerator  # noqa: E402
from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter  # noqa: E402
from traveler_distillation.student.release_guard import assert_not_frozen_output  # noqa: E402

from run_phase_c import (  # noqa: E402
    SCENARIOS,
    SUPPLY,
    _evaluate_gate,
    _manifest_stats,
    _parse_events_full,
    _run_matsim,
    load_link_attrs,
    make_context,
    make_shared_factory,
    make_shared_idx,
    sha256,
)

FROZEN_S9_SHA256 = "6af79b44bc699c00cc8e913da15043a43398829fdc8bbb4dad2ddcb8e11d31e6"
GEN_CONFIG = "configs/generation_v0_1.yaml"
DESIGN = "TRC_AIT_5_E4_MULTISEED_ROBUSTNESS_DESIGN.md v0.1"
ALL_SCENARIOS = [
    "C0_baseline", "C1_heavy_rain", "C2_fare_increase",
    "C3_transit_delay", "C4_road_disruption", "C5_joint_rain_delay",
]
MATSIM_TIMEOUT_S = 10_800


def run_scenario_e4(name: str, n: int, seed: int, personas, trips, trips_per_persona,
                    adapter: S8MATSimAdapter, supply: dict, root: Path, out_root: Path,
                    skip_matsim: bool, factory, checkpoint_path: Path,
                    checkpoint_digest: str, contention_note: str | None,
                    torch) -> tuple[dict, bool]:
    """Thin seed-aware mirror of run_phase_c.run_scenario (same code paths).

    Differences vs run_scenario (by design §3.2): real population_seed, real
    checkpoint path/SHA256, artifact fingerprints, parse timing. Everything
    else (context, factory, build, MATSim, event parse, gate) is identical.
    """
    s = SCENARIOS[name]
    context = make_context(name)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    manifest = adapter.build_real_scenario(
        personas, trips, context, out,
        network_path=supply["network"], schedule_path=supply["schedule"],
        vehicles_path=supply["vehicles"], stops_path=supply["stops"],
        snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
        activity_nodes_path=supply["activity_nodes"],
        trips_per_persona=trips_per_persona,
        flow_capacity_factor=0.3, storage_capacity_factor=0.3,
        alt_factory=factory,
    )
    build_s = time.perf_counter() - t0
    stats = _manifest_stats(manifest)

    manifest_file = out / "adapter_manifest.json"
    pop_xml = out / "population.xml"
    record = {
        "experiment": "E4",
        "design": DESIGN,
        "scenario": name,
        "label": s["label"],
        "context": {k: (v if not isinstance(v, tuple) else {"condition": v[0], "intensity": v[1]})
                    for k, v in s.items()},
        "n_agents": n,
        "population_seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "gen_config": GEN_CONFIG,
        "gen_config_sha256": sha256(root / GEN_CONFIG),
        "frozen_settings": {
            "flow_capacity_factor": 0.3, "storage_capacity_factor": 0.3,
            "last_iteration": 0, "network": supply["network"], "schedule": supply["schedule"],
            "matsim_random_seed": 4711,
        },
        "environment": {
            "platform": platform.platform(),
            "cpu_logical": 24,
            "torch_threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "java": "OpenJDK 25.0.4",
            "matsim": "MATSim 2026.0 (matsim_rel/matsim-2026.0.jar)",
            "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "contention_note": contention_note or "none declared",
        },
        "build_seconds": round(build_s, 1),
        "decisions": stats,
        "artifacts": {
            "population_xml_sha256": sha256(pop_xml),
            "adapter_manifest_sha256": sha256(manifest_file),
            "population_xml_bytes": pop_xml.stat().st_size,
            "adapter_manifest_bytes": manifest_file.stat().st_size,
        },
        "matsim_exit_code": None,
        "runtime_seconds": None,
        "parse_seconds": None,
        "metrics": None,
        "log_tail": "",
    }

    links = load_link_attrs(str(root / supply["network"]))
    if not skip_matsim:
        t1 = time.perf_counter()
        code, tail = _run_matsim(out, timeout_s=MATSIM_TIMEOUT_S)
        record["runtime_seconds"] = round(time.perf_counter() - t1, 1)
        record["matsim_exit_code"] = code
        record["log_tail"] = tail[-2000:] if tail else ""
        t2 = time.perf_counter()
        metrics = _parse_events_full(out, links)
        record["parse_seconds"] = round(time.perf_counter() - t2, 1)
        metrics["leg_departures"] = dict(metrics.pop("leg_departures"))
        record["metrics"] = metrics
        if code != 0:
            print(f"[{name}] MATSim exit={code}\nlog tail:\n{tail}")

    gate = _evaluate_gate(name, record)
    record["gate"] = gate
    (out / "e4_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record, gate["pass"]


def _check_g1(r1_dir: Path, r0_dir: Path) -> dict:
    """G1 determinism: gate repeat (r1) vs evidence run (r0), design §8.1."""
    r0 = json.loads((r0_dir / "C0_baseline" / "e4_result.json").read_text(encoding="utf-8"))
    r1 = json.loads((r1_dir / "C0_baseline" / "e4_result.json").read_text(encoding="utf-8"))
    checks = {
        "population_xml_sha256_equal": r0["artifacts"]["population_xml_sha256"] == r1["artifacts"]["population_xml_sha256"],
        "adapter_manifest_sha256_equal": r0["artifacts"]["adapter_manifest_sha256"] == r1["artifacts"]["adapter_manifest_sha256"],
        "decisions_equal": r0["decisions"] == r1["decisions"],
        "metrics_equal": r0["metrics"] == r1["metrics"],
    }
    return {
        "gate": "G1", "r0_dir": str(r0_dir), "r1_dir": str(r1_dir),
        "checks": checks, "pass": all(checks.values()),
    }


def run_one_seed(seed: int, scenarios: list[str], args: argparse.Namespace) -> int:
    import torch  # local import (E3 pattern)

    if seed == 2026:
        raise SystemExit("seed 2026 is the frozen Phase C column — E4 never re-runs it (design §0/§7)")
    root = ROOT
    ckpt = root / args.checkpoint
    digest = sha256(ckpt)
    if digest != FROZEN_S9_SHA256:
        raise SystemExit(f"checkpoint SHA256 {digest} != frozen S9 {FROZEN_S9_SHA256} ({ckpt})")

    out_root = assert_not_frozen_output(root / args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.gate_repeat:
        out = out_root / "_gate" / f"seed{seed}_C0_r1"
    else:
        out = out_root / f"seed{seed}"
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} already exists and is non-empty; use --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    gen_cfg = load_yaml(str(root / GEN_CONFIG))
    print(f"generating population (N={args.num_agents}, seed={seed})...", flush=True)
    personas = PersonaGenerator(seed=seed, config=gen_cfg).generate(args.num_agents)
    trips = TripGenerator(seed=seed, config=gen_cfg).generate(args.num_agents)
    trips_per_persona = [[t] for t in trips]
    adapter = S8MATSimAdapter(ckpt)
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}
    shared_idx = make_shared_idx(supply)
    factories = {name: make_shared_factory(shared_idx, make_context(name)) for name in scenarios}

    records = []
    ok = True
    for name in scenarios:
        print(f"=== seed {seed} scenario {name} ({SCENARIOS[name]['label']}) ===", flush=True)
        rec, passed = run_scenario_e4(
            name, args.num_agents, seed, personas, trips, trips_per_persona,
            adapter, supply, root, out, args.skip_matsim, factories[name],
            checkpoint_path=ckpt, checkpoint_digest=digest,
            contention_note=args.contention_note, torch=torch,
        )
        records.append(rec)
        ok = ok and passed
        print(f"[seed{seed}] {name} done; gate={'PASS' if passed else 'FAIL'} "
              f"(build {rec['build_seconds']}s, matsim {rec['runtime_seconds']}s)", flush=True)

    (out / "seed_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    g1 = None
    if args.gate_repeat:
        g1 = _check_g1(out, out_root / f"seed{seed}")
        (out / "g1_check.json").write_text(json.dumps(g1, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = ok and g1["pass"]
        print(json.dumps({"g1": g1}, ensure_ascii=False))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scenarios", default=",".join(ALL_SCENARIOS))
    ap.add_argument("--num-agents", type=int, default=10000)
    ap.add_argument("--checkpoint", default="releases/s9_supply_aware_v2/checkpoint/model.pt")
    ap.add_argument("--output", default="outputs/e4_multiseed")
    ap.add_argument("--gate-repeat", action="store_true",
                    help="C0-only validation repeat for G1 (writes _gate/seed<seed>_C0_r1; not evidence)")
    ap.add_argument("--skip-matsim", action="store_true",
                    help="build-only debug (never evidence)")
    ap.add_argument("--contention-note", default=None,
                    help="honest record of concurrent machine workload (e.g. E5 Helsinki)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    if args.gate_repeat and names != ["C0_baseline"]:
        raise SystemExit("--gate-repeat requires --scenarios C0_baseline")
    return run_one_seed(args.seed, names, args)


if __name__ == "__main__":
    raise SystemExit(main())
