#!/usr/bin/env python
"""Phase B.5A — PT Routing Validity Gate.

Builds the 1000-agent scenario with the transfer-aware PT planner and the
two-window supply, runs MATSim, and reports the PT itinerary-construction
validity:

    intended PT -> routed PT (direct / transfer) / fallback (by reason)

Success target: >= 90% of Student-intended PT trips get a built PT itinerary.
The MATSim run additionally verifies that transfer passengers actually board,
alight and leave no one stuck (the chainedRoute crash class).

Usage:
    python scripts/singapore/run_phase_b5a_pt_validity.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traveler_distillation.config import load_yaml
from traveler_distillation.generators import PersonaGenerator, TripGenerator
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.matsim import MATSimAdapter

_TMP = "C:/Users/xuan1/AppData/Local/Temp"
_CP = f"{_TMP}/matsim_run;{_TMP}/matsim_rel/matsim-2026.0.jar;{_TMP}/matsim_rel/libs/*"

SUPPLY = {
    "network": "data/singapore/transit/network_with_transit.xml",
    "schedule": "data/singapore/transit/transitSchedule.xml",
    "vehicles": "data/singapore/transit/transitVehicles.xml",
    "stops": "data/singapore/transit/prep_stops.jsonl",
    "snapping": "data/singapore/transit/stop_snapping_report.json",
    "trips_by_stop": "data/singapore/transit/trips_by_stop.json",
    "activity_nodes": "data/singapore/transit/activity_nodes.json",
}


def _make_context() -> DynamicContext:
    return DynamicContext(
        context_id="C_SG_B5A_BASELINE",
        weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )


def _events_check(sdir: Path) -> dict:
    import zstandard
    ev = sdir / "output" / "ITERS" / "it.0" / "0.events.xml.zst"
    if not ev.exists():
        return {"events_missing": True}
    dctx = zstandard.ZstdDecompressor()
    enters = Counter()
    leaves = Counter()
    stuck = 0
    with dctx.stream_reader(open(ev, "rb")) as f:
        for event, el in ET.iterparse(f, events=("end",)):
            if el.tag != "event":
                continue
            t = el.get("type")
            if t == "PersonEntersPtVehicle":
                enters[str(el.get("vehicle"))] += 1
            elif t == "PersonLeavesPtVehicle":
                leaves[str(el.get("vehicle"))] += 1
            elif t == "stuckAndAbort":
                stuck += 1
            el.clear()
    return {"pt_boardings": sum(enters.values()), "pt_alightings": sum(leaves.values()),
            "stuck_and_abort": stuck,
            "vehicles_with_unbalanced_enter_leave": sum(1 for v in enters if enters[v] != leaves[v])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/student_s7_w3/checkpoints/best.pt")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--n-agents", type=int, default=1000)
    ap.add_argument("--output", default="outputs/singapore_phase_b5/pt_validity")
    ap.add_argument("--skip-matsim", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}
    gen_cfg = load_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    adapter = MATSimAdapter(args.checkpoint)
    context = _make_context()
    n = args.n_agents

    personas = PersonaGenerator(seed=3000, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=3000, config=gen_cfg).generate(n)
    trips_per_persona = [[t] for t in trips]

    t0 = time.time()
    manifest = adapter.build_real_scenario(
        personas, trips, context, out,
        network_path=supply["network"], schedule_path=supply["schedule"],
        vehicles_path=supply["vehicles"], stops_path=supply["stops"],
        snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
        activity_nodes_path=supply["activity_nodes"],
        trips_per_persona=trips_per_persona,
    )
    build_s = time.time() - t0

    # ---- PT validity from the manifest (outbound + return separately) ----
    intended_out = sum(1 for d in manifest if d["student_mode"] == "pt")
    intended_ret = sum(1 for d in manifest if d["outbound_mode"] == "pt")

    def _classify(info):
        if info.get("mode") != "pt":
            return "fallback", info.get("pt_reason", "unknown")
        if info.get("transfers", 0) == 0:
            return "direct", None
        return "transfer", None

    def _table(rows):
        routed = sum(1 for _, c, _ in rows if c in ("direct", "transfer"))
        direct = sum(1 for _, c, _ in rows if c == "direct")
        transfer = sum(1 for _, c, _ in rows if c == "transfer")
        reasons = Counter(r for _, c, r in rows if c == "fallback")
        return {
            "intended": len(rows), "routed": routed, "direct": direct,
            "transfer": transfer, "fallback": len(rows) - routed,
            "success_rate": round(routed / max(1, len(rows)), 4),
            "fallback_reasons": dict(reasons),
        }

    out_rows = [(_classify(d["outbound"])[0], _classify(d["outbound"])[0], _classify(d["outbound"])[1])
                for d in manifest if d["student_mode"] == "pt"]
    ret_rows = [(_classify(d["return"])[0], _classify(d["return"])[0], _classify(d["return"])[1])
                for d in manifest if d["outbound_mode"] == "pt"]
    table = {"outbound": _table(out_rows), "return": _table(ret_rows)}
    combined = {
        "intended": table["outbound"]["intended"] + table["return"]["intended"],
        "routed": table["outbound"]["routed"] + table["return"]["routed"],
        "direct": table["outbound"]["direct"] + table["return"]["direct"],
        "transfer": table["outbound"]["transfer"] + table["return"]["transfer"],
        "fallback": table["outbound"]["fallback"] + table["return"]["fallback"],
    }
    combined["success_rate"] = round(combined["routed"] / max(1, combined["intended"]), 4)

    record = {
        "n_agents": n,
        "build_seconds": round(build_s, 1),
        "outbound": table["outbound"],
        "return": table["return"],
        "combined": combined,
        "planning_fallbacks": json.loads((out / "adapter_manifest.json").read_text(encoding="utf-8"))["fallback_counts"],
    }
    if not args.skip_matsim:
        with open(out / "java_run.log", "w", encoding="utf-8", errors="replace") as f:
            t0 = time.time()
            proc = subprocess.run(
                ["java", "-Xmx8g", "-cp", _CP, "RunMatsimPreloaded", "config.xml"],
                cwd=str(out), stdout=f, stderr=subprocess.STDOUT, timeout=10800,
            )
            record["matsim_exit_code"] = proc.returncode
            record["matsim_runtime_seconds"] = round(time.time() - t0, 1)
            record["execution_check"] = _events_check(out)

    (out / "pt_validity.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Phase B.5A — PT Routing Validity Gate\n",
        f"- supply：两时窗（06:00–10:30 + 15:00–20:00），11,796 trips / 856 站；\n"
        f"- 规划器：直连 + 1 次换乘（独立 pt leg + 换乘步行，无 chainedRoute）；\n"
        f"- 样本：{n} agents（seed=3000），Student 决策由冻结 S7-W3 给出。\n",
        "## 结果表\n",
        "| 方向 | intended PT | routed PT | direct PT | transfer PT | fallback | success rate |\n|---|---|---|---|---|---|---|",
        f"| outbound | {table['outbound']['intended']} | {table['outbound']['routed']} | {table['outbound']['direct']} | "
        f"{table['outbound']['transfer']} | {table['outbound']['fallback']} | {table['outbound']['success_rate']} |",
        f"| return | {table['return']['intended']} | {table['return']['routed']} | {table['return']['direct']} | "
        f"{table['return']['transfer']} | {table['return']['fallback']} | {table['return']['success_rate']} |",
        f"| **合计** | {combined['intended']} | {combined['routed']} | {combined['direct']} | {combined['transfer']} | "
        f"{combined['fallback']} | **{combined['success_rate']}** |",
        "",
        "## Fallback 原因分类（合计）\n",
    ]
    reasons = Counter()
    for d in table.values():
        reasons.update(d["fallback_reasons"])
    for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        md.append(f"- `{r}`: {c}\n")
    md.append("\n## 判定\n")
    ok = combined["success_rate"] >= 0.90
    md.append(f"- 目标 ≥90%：**{'达成 ✅' if ok else '未达成 ❌'}**（{combined['success_rate']:.1%}）\n")
    if "execution_check" in record:
        ec = record["execution_check"]
        md.append(f"- MATSim 执行：exit={record['matsim_exit_code']}；pt 上车 {ec.get('pt_boardings')} / "
                  f"下车 {ec.get('pt_alightings')}；stuckAndAbort {ec.get('stuck_and_abort')}；"
                  f"上下车不平衡车辆 {ec.get('vehicles_with_unbalanced_enter_leave')}\n")
    (out / "pt_validity.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"wrote {out / 'pt_validity.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
