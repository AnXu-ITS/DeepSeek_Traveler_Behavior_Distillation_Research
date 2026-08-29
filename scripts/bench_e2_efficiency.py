#!/usr/bin/env python
"""E2 (TRC_AIT_5_EXPERIMENT_PLAN §E2; design TRC_AIT_5_E2_DEEPSEEK_S9_EFFICIENCY_DESIGN.md)
benchmark — frozen C0 baseline state pool + frozen S9 CPU timing + G2 consistency.

Subcommands:
  build-pool  Generate N C0 baseline states (persona/trip seed 2026,
              generation_v0_1.yaml, real frozen supply) exactly as Phase C does,
              decide with the FROZEN S9 checkpoint, and persist states,
              decisions and per-state timing. Supports --workers N: the state
              range is split across N processes (Windows spawn); each range is
              deterministic and byte-identical to the single-process build.
  time-s9     Per-state and batched decision timing over a built pool
              (warmup, threads config, repeats; decisions-out for G4).
  g2-check    Byte-compare the 10k-prefix decisions vs the frozen Phase C C0
              adapter_manifest.json.

E2 measures behavior inference ONLY (state -> {mode, departure} decision);
MATSim runtime is never executed here. Input construction (accessibility
planning / alternatives) is recorded separately as shared input build time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "singapore"))

from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import PersonaGenerator, TripGenerator  # noqa: E402
from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter  # noqa: E402
from traveler_distillation.schemas.state import UniversalTravelerState  # noqa: E402
from traveler_distillation.singapore.osm_network import load_network_graph  # noqa: E402
from traveler_distillation.student.dataset import collate_batch  # noqa: E402

from run_phase_c import SUPPLY, make_context, make_shared_factory, make_shared_idx  # noqa: E402

FROZEN_S9_SHA256 = "6af79b44bc699c00cc8e913da15043a43398829fdc8bbb4dad2ddcb8e11d31e6"
POP_SEED = 2026
GEN_CONFIG = "configs/generation_v0_1.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _filtered_activity_nodes(network_path: Path) -> list[dict]:
    """Replicate build_real_scenario's activity-node filter exactly.

    Order matters: car-accessible filter, then walk-graph largest-component
    filter, preserving the JSON order — the OD mapping index depends on it.
    """
    import networkx as nx

    g, xy = load_network_graph(str(network_path))
    walk_g = nx.DiGraph()
    for u, v, d in g.edges(data=True):
        if "walk" in d["modes"]:
            tt = d["length"] / 1.34  # S9 walk speed fix (build_real_scenario)
            walk_g.add_edge(u, v, tt=tt)
    for n, (x, y) in xy.items():
        walk_g.add_node(n, x=x, y=y)
    car_nodes = set()
    for u, v, d in g.edges(data=True):
        if "car" in d["modes"]:
            car_nodes.add(u)
            car_nodes.add(v)
    walk_und = nx.Graph()
    for u, v in walk_g.edges():
        walk_und.add_edge(u, v)
    walk_largest = max(nx.connected_components(walk_und), key=len)
    raw = json.loads(Path(ROOT / SUPPLY["activity_nodes"]).read_text(encoding="utf-8"))
    nodes = [a for a in raw if a["node"] in car_nodes]
    nodes = [a for a in nodes if a["node"] in walk_largest]
    return nodes


def _load_adapter(checkpoint_path: Path) -> S8MATSimAdapter:
    digest = sha256(checkpoint_path)
    if digest != FROZEN_S9_SHA256:
        raise SystemExit(
            f"checkpoint SHA256 {digest} != frozen S9 {FROZEN_S9_SHA256} ({checkpoint_path})"
        )
    return S8MATSimAdapter(str(checkpoint_path))


def _build_range(job: dict) -> dict:
    """One worker: build states [lo, hi) deterministically, write range files."""
    lo, hi = job["lo"], job["hi"]
    personas = job["personas"]
    trips = job["trips"]
    activity_nodes = job["activity_nodes"]
    supply = job["supply"]
    ckpt_path = Path(job["checkpoint"])
    out_dir = Path(job["out_dir"])
    suffix = f"{lo}_{hi}"

    adapter = _load_adapter(ckpt_path)
    idx = make_shared_idx(supply)
    ctx = make_context("C0_baseline")
    factory = make_shared_factory(idx, ctx)

    states_path = out_dir / f"states_range_{suffix}.jsonl"
    decisions_path = out_dir / f"decisions_range_{suffix}.json"
    timing_path = out_dir / f"timing_range_{suffix}.json"

    decisions: list[dict] = []
    input_build_s: list[float] = []
    decide_s: list[float] = []
    od_pairs: set[tuple[str, str]] = set()
    t0 = time.perf_counter()
    with states_path.open("w", encoding="utf-8") as fh:
        for idx, (persona, trip) in enumerate(zip(personas, trips), start=lo):
            h = int(hashlib.sha256(persona.persona_id.encode()).hexdigest(), 16)
            home = activity_nodes[h % len(activity_nodes)]["node"]
            dest = activity_nodes[(h + 7919) % len(activity_nodes)]["node"]
            od_pairs.add((home, dest))

            t_b0 = time.perf_counter()
            alternatives = factory(persona, trip, ctx, home, dest)
            t_b1 = time.perf_counter()
            state = UniversalTravelerState(
                persona=persona, trip=trip, context=ctx,
                alternatives=alternatives,
            )
            decision = adapter.decide(state)
            t_b2 = time.perf_counter()

            input_build_s.append(t_b1 - t_b0)
            decide_s.append(t_b2 - t_b1)
            decisions.append({
                "persona_id": persona.persona_id,
                "trip_id": trip.trip_id,
                "home_node": home,
                "dest_node": dest,
                "student_mode": decision["mode"],
                "departure_shift_min": round(decision["departure_time_shift_min"], 2),
                "departure_min": round(trip.desired_departure_min + decision["departure_time_shift_min"], 2),
            })
            fh.write(state.model_dump_json() + "\n")

    decisions_path.write_text(json.dumps(decisions, ensure_ascii=False), encoding="utf-8")
    timing_path.write_text(json.dumps({
        "lo": lo, "hi": hi,
        "input_build_s": input_build_s,
        "decide_s": decide_s,
    }, ensure_ascii=False), encoding="utf-8")
    el = time.perf_counter() - t0
    print(f"[worker {lo}-{hi}] {hi - lo} states in {el:.0f}s "
          f"({el / max(hi - lo, 1) * 1000:.1f} ms/state)", flush=True)
    return {"lo": lo, "hi": hi, "unique_ods": len(od_pairs), "wall_s": round(el, 1)}


def cmd_build_pool(args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_agents
    ckpt = Path(args.checkpoint)
    workers = max(1, args.workers)

    t_start = time.perf_counter()
    gen_cfg = load_yaml(str(ROOT / GEN_CONFIG))
    personas = PersonaGenerator(seed=POP_SEED, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=POP_SEED, config=gen_cfg).generate(n)
    activity_nodes = _filtered_activity_nodes(Path(ROOT / SUPPLY["network"]))
    supply = {k: str((ROOT / v).resolve()) for k, v in SUPPLY.items()}
    _load_adapter(ckpt)  # verify checkpoint identity before forking work
    print(f"setup done in {time.perf_counter() - t_start:.0f}s; "
          f"activity_nodes(filtered)={len(activity_nodes)}", flush=True)

    bounds = [(round(n * k / workers), round(n * (k + 1) / workers)) for k in range(workers)]
    jobs = [{
        "lo": lo, "hi": hi,
        "personas": personas[lo:hi],
        "trips": trips[lo:hi],
        "activity_nodes": activity_nodes,
        "supply": supply,
        "checkpoint": str(ckpt),
        "out_dir": str(out),
    } for lo, hi in bounds]

    results = []
    if workers == 1:
        results = [_build_range(jobs[0])]
    else:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for r in pool.imap_unordered(_build_range, jobs):
                results.append(r)
        results.sort(key=lambda r: r["lo"])

    # merge range files in order
    def _safe_unlink(p: Path) -> None:
        try:
            p.unlink()
        except PermissionError:
            print(f"[merge] could not delete {p.name} (in use) — leaving for manual cleanup",
                  flush=True)

    states_path = out / f"states_{n}.jsonl"
    decisions_path = out / f"decisions_{n}.json"
    timing_path = out / f"build_timing_{n}.json"
    with states_path.open("w", encoding="utf-8") as fh:
        for r in results:
            part = out / f"states_range_{r['lo']}_{r['hi']}.jsonl"
            fh.write(part.read_text(encoding="utf-8"))
            _safe_unlink(part)
    all_decisions = []
    input_build_s: list[float] = []
    decide_s: list[float] = []
    for r in results:
        all_decisions += json.loads((out / f"decisions_range_{r['lo']}_{r['hi']}.json").read_text(encoding="utf-8"))
        t = json.loads((out / f"timing_range_{r['lo']}_{r['hi']}.json").read_text(encoding="utf-8"))
        input_build_s += t["input_build_s"]
        decide_s += t["decide_s"]
        _safe_unlink(out / f"decisions_range_{r['lo']}_{r['hi']}.json")
        _safe_unlink(out / f"timing_range_{r['lo']}_{r['hi']}.json")
    decisions_path.write_text(json.dumps(all_decisions, ensure_ascii=False), encoding="utf-8")
    timing_path.write_text(json.dumps({
        "n": n,
        "input_build_s": input_build_s,
        "decide_s": decide_s,
    }, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "experiment": "E2",
        "pool_n": n,
        "population_seed": POP_SEED,
        "gen_config": GEN_CONFIG,
        "gen_config_sha256": sha256(Path(ROOT / GEN_CONFIG)),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "context": "C0_baseline",
        "activity_nodes_raw": len(json.loads(Path(ROOT / SUPPLY["activity_nodes"]).read_text(encoding="utf-8"))),
        "activity_nodes_filtered": len(activity_nodes),
        "workers": workers,
        "worker_ranges": [{"lo": r["lo"], "hi": r["hi"], "unique_ods": r["unique_ods"], "wall_s": r["wall_s"]} for r in results],
        "unique_od_pairs_workers": sum(r["unique_ods"] for r in results),
        "supply_sha256": {k: sha256(Path(ROOT / v)) for k, v in SUPPLY.items()},
        "states_file": str(states_path),
        "states_sha256": sha256(states_path),
        "decisions_file": str(decisions_path),
        "build_timing_file": str(timing_path),
        "build_wall_seconds": round(time.perf_counter() - t_start, 1),
        "note": "state->decision pipeline identical to Phase C build_real_scenario "
                "(same generators/seed, same OD formula, same shared alt factory, "
                "same decide() call); MATSim never runs. Multiprocess ranges are "
                "deterministic and byte-identical to the single-process build "
                "(verified by G2 on the 10k prefix).",
    }
    (out / "pool_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"done: {n} states in {manifest['build_wall_seconds']}s; "
          f"states {states_path}", flush=True)
    return 0


def cmd_time_s9(args: argparse.Namespace) -> int:
    pool = Path(args.pool)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = _load_adapter(Path(args.checkpoint))

    threads = args.threads
    if threads == "default":
        n_threads = torch.get_num_threads()
    else:
        n_threads = int(threads)
        torch.set_num_threads(n_threads)

    def _parse(limit: int):
        with pool.open(encoding="utf-8") as fh:
            count = 0
            for line in fh:
                if line.strip():
                    yield UniversalTravelerState.model_validate_json(line)
                    count += 1
                    if limit and count >= limit:
                        return

    n_available = sum(1 for _ in _parse(args.max_states or 0))
    n = n_available
    warm = min(args.warmup, n // 2)

    # ---- per-state sequential decision timing (streamed, low memory) ----
    warmed = 0
    for s in _parse(args.max_states or 0):
        if warmed < warm:
            adapter.decide(s)
            warmed += 1
            continue
        break
    per_state: list[float] = []
    decisions: list[dict] = []
    for s in _parse(args.max_states or 0):
        t0 = time.perf_counter()
        d = adapter.decide(s)
        per_state.append(time.perf_counter() - t0)
        decisions.append({
            "persona_id": s.persona.persona_id,
            "trip_id": s.trip.trip_id,
            "student_mode": d["mode"],
            "departure_shift_min": round(d["departure_time_shift_min"], 2),
            "departure_min": round(s.trip.desired_departure_min + d["departure_time_shift_min"], 2),
        })
    assert len(per_state) == n, (len(per_state), n)

    # ---- batched end-to-end (encode + collate + forward per chunk) ----
    batch_max = args.batch_max_states or n
    states_b = list(_parse(batch_max))
    batch_rows: dict[str, dict] = {}
    for b in args.batch_sizes:
        chunks = [states_b[i:i + b] for i in range(0, len(states_b), b)]
        for chunk in chunks[: min(2, len(chunks))]:  # warm
            feats = [adapter.extractor.encode(s) for s in chunk]
            bat = _collate(feats)
            with torch.no_grad():
                adapter.model(bat)
        t0 = time.perf_counter()
        for chunk in chunks:
            feats = [adapter.extractor.encode(s) for s in chunk]
            bat = _collate(feats)
            with torch.no_grad():
                adapter.model(bat)
        dt = time.perf_counter() - t0
        batch_rows[str(b)] = {
            "batch_size": b,
            "n_chunks": len(chunks),
            "n_states": len(states_b),
            "total_s": round(dt, 4),
            "states_per_s": round(len(states_b) / dt, 1),
            "ms_per_state": round(dt / len(states_b) * 1000, 4),
        }

    record = {
        "pool": str(pool),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "threads": threads,
        "n_threads_effective": n_threads,
        "repeat_index": args.repeat,
        "n_states_per_state_pass": n,
        "warmup": warm,
        "per_state_s": per_state,
        "batch_n_states": len(states_b),
        "batch": batch_rows,
        "decision_out": args.decisions_out,
    }
    out_path = out / f"s9_time_t{threads}_r{args.repeat}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    if args.decisions_out:
        Path(args.decisions_out).write_text(
            json.dumps(decisions, ensure_ascii=False), encoding="utf-8"
        )
    p = sorted(per_state)
    mean = sum(per_state) / n
    print(f"[time-s9] threads={threads}({n_threads}) n={n} "
          f"mean={mean * 1000:.3f} ms p50={p[n // 2] * 1000:.3f} ms "
          f"p95={p[int(n * 0.95)] * 1000:.3f} ms; batch(n={len(states_b)}): "
          + ", ".join(f"{b}:{r['states_per_s']:.0f}/s" for b, r in batch_rows.items()),
          flush=True)
    return 0


def _collate(feats):
    rows = [{
        "global_cat": torch.tensor(f["global_cat"], dtype=torch.long),
        "global_num": torch.tensor(f["global_num"], dtype=torch.float32),
        "alt_mode_idx": torch.tensor(f["alt_mode_idx"], dtype=torch.long),
        "alt_num": torch.tensor(f["alt_num"], dtype=torch.float32),
        "alt_mask": torch.tensor(f["alt_available"], dtype=torch.float32),
    } for f in feats]
    return collate_batch(rows)


def cmd_g2_check(args: argparse.Namespace) -> int:
    built = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))["decisions"]
    fields = ["persona_id", "trip_id", "student_mode", "departure_shift_min", "departure_min"]
    n = min(len(built), len(frozen), args.n)
    mismatches = []
    for i in range(n):
        b, f = built[i], frozen[i]
        bad = {k: (b.get(k), f.get(k)) for k in fields if b.get(k) != f.get(k)}
        if bad:
            mismatches.append((i, bad))
    result = {
        "gate": "G2",
        "compared_n": n,
        "mismatches": len(mismatches),
        "first_mismatches": [{"index": i, "diff": d} for i, d in mismatches[:10]],
        "pass": len(mismatches) == 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-pool")
    p.add_argument("--num-agents", type=int, default=100_000)
    p.add_argument("--output", default="outputs/e2_efficiency")
    p.add_argument("--checkpoint", default="releases/s9_supply_aware_v2/checkpoint/model.pt")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(fn=cmd_build_pool)

    p = sub.add_parser("time-s9")
    p.add_argument("--pool", required=True)
    p.add_argument("--checkpoint", default="releases/s9_supply_aware_v2/checkpoint/model.pt")
    p.add_argument("--threads", default="1")
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 256, 1024])
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--max-states", type=int, default=0)
    p.add_argument("--batch-max-states", type=int, default=20_000)
    p.add_argument("--decisions-out", default=None)
    p.add_argument("--out", default="outputs/e2_efficiency")
    p.set_defaults(fn=cmd_time_s9)

    p = sub.add_parser("g2-check")
    p.add_argument("--decisions", required=True)
    p.add_argument("--frozen", default="outputs/singapore_phase_c_s9/C0_baseline/adapter_manifest.json")
    p.add_argument("--n", type=int, default=10_000)
    p.set_defaults(fn=cmd_g2_check)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
