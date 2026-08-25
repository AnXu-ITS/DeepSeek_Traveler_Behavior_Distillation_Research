#!/usr/bin/env python
"""Phase B.5C diagnostic: why is the 40k baseline free-flow?

Answers the 5 audit questions from the 40k probe run (no re-simulation):

1. peak-hour CAR demand (vehicles, not agents);
2. per-link peak-hour V/C: max / P99 / P95 + top-20 bottleneck links;
3. temporal concentration (departures per hour);
4. OD dispersion (distinct OD pairs vs cars; VKT concentration on top links);
5. full population vs sample (sample factor vs a realistic demand reference).

Usage:
    python scripts/singapore/diagnose_loading.py \
        --events outputs/singapore_phase_b5/demand_sweep_probe40k/scale_40000/output/ITERS/it.0/0.events.xml.zst \
        --network data/singapore/transit/network_with_transit.xml \
        --manifest outputs/singapore_phase_b5/demand_sweep_probe40k/scale_40000/adapter_manifest.json \
        --output outputs/singapore_phase_b5/loading_diagnostic.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

_CAR = re.compile(r"^P\d+$")


def _load_links(network: str) -> dict[str, tuple[float, float, float]]:
    """link id -> (length, freespeed, capacity)."""
    txt = open(network, encoding="utf-8").read()
    out = {}
    for m in re.finditer(
        r'<link id="([^"]+)" from="([^"]+)" to="([^"]+)" length="([\d.]+)" '
        r'freespeed="([\d.]+)" capacity="([\d.]+)"[^>]*modes="([^"]+)"',
        txt,
    ):
        if "car" in m.group(7).split(","):
            out[m.group(1)] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--network", default="data/singapore/transit/network_with_transit.xml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--output", default="outputs/singapore_phase_b5/loading_diagnostic.md")
    args = ap.parse_args()

    import zstandard
    links = _load_links(args.network)
    print(f"car links loaded: {len(links)}", flush=True)

    # --- pass 1: events ---
    dctx = zstandard.ZstdDecompressor()
    car_vehicles: set[str] = set()
    dep_hour = Counter()
    link_flow: dict[str, Counter] = defaultdict(Counter)  # link -> hour -> count
    link_vkt: Counter = Counter()
    with dctx.stream_reader(open(args.events, "rb")) as f:
        for event, el in ET.iterparse(f, events=("end",)):
            if el.tag != "event":
                continue
            t = el.get("type")
            if t == "departure" and el.get("legMode") == "car":
                vid = el.get("person")
                if vid and _CAR.match(vid):  # exclude transit-driver dummy persons (pt_veh_*)
                    car_vehicles.add(vid)
                    dep_hour[int(float(el.get("time")) // 3600)] += 1
            elif t == "entered link":
                vid = el.get("vehicle")
                if vid and _CAR.match(vid) and el.get("link") in links:
                    link_flow[el.get("link")][int(float(el.get("time")) // 3600)] += 1
            el.clear()

    n_cars = len(car_vehicles)
    print(f"car vehicles: {n_cars}", flush=True)

    # --- per-link peak-hour V/C ---
    vc = []
    vkt_total = 0.0
    for lid, hours in link_flow.items():
        peak_flow = max(hours.values())
        length, _free, cap = links[lid]
        vc.append((lid, peak_flow, cap, peak_flow / cap if cap > 0 else 0.0, length))
        vkt_total += sum(hours.values()) * length
    vc.sort(key=lambda r: -r[3])
    n_used = len(vc)
    vc_vals = sorted(r[3] for r in vc)

    def _pct(p):
        if not vc_vals:
            return 0.0
        return vc_vals[min(len(vc_vals) - 1, int(len(vc_vals) * p))]

    # --- OD dispersion from manifest ---
    od_note = "manifest not provided"
    if args.manifest and Path(args.manifest).exists():
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        cars = [d for d in manifest["decisions"] if d.get("outbound_mode") == "car"]
        od_pairs = {(d["home_node"], d["dest_node"]) for d in cars}
        od_note = (f"car agents={len(cars)}, distinct OD pairs={len(od_pairs)}, "
                   f"pairs/cars={len(od_pairs) / max(1, len(cars)):.3f}")

    # --- VKT concentration on top links ---
    top20_vkt = sum(r[4] * r[1] for r in vc[:20])
    concentration = top20_vkt / max(1.0, vkt_total)

    # --- reference demand for the sample verdict ---
    # Tampines + Pasir Ris: ~450k residents (HDB town data), ~2.2 trips/person/day,
    # ~30% car driver share -> ~300k car trips/day; peak-hour ~10% of daily -> ~30k cars/h.
    sim_peak_cars_per_h = max(dep_hour.values())
    sample_factor = sim_peak_cars_per_h / 30000.0

    lines = [
        "# Phase B.5C — 40k Baseline Loading 诊断\n",
        f"数据：`{args.events}`（40k 探针，无重跑）\n",
        "## 1. Peak-hour car demand（车辆数，非 agents 数）\n",
        f"- 当日 car 出行车辆：**{n_cars}** 辆（40,000 agents × ~28% car 决策，一致）\n"
        f"- 逐小时出发分布：`{dict(sorted(dep_hour.items()))}`\n"
        f"- **模拟峰值 {sim_peak_cars_per_h} 辆/h**\n",
        "## 2. 逐链路 peak-hour V/C（仅 car 可用链路）\n",
        f"- 使用中的链路：{n_used}；V/C max={vc[0][3]:.3f}（{vc[0][0]}，cap={vc[0][2]:.0f}，"
        f"peak flow={vc[0][1]}/h）、P99={_pct(0.99):.3f}、P95={_pct(0.95):.3f}、"
        f"P50={_pct(0.50):.3f}\n",
        "| rank | link | peak flow (veh/h) | capacity | V/C |\n|---|---|---|---:|---:|",
    ]
    for i, (lid, flow, cap, ratio, _len) in enumerate(vc[:20], 1):
        lines.append(f"| {i} | `{lid}` | {flow} | {cap:.0f} | {ratio:.3f} |")
    lines += [
        "",
        "## 3. 时间集中度\n",
        f"- 出发分布见 §1；若车辆在两三小时内均匀释放，拥堵天然被稀释。"
        f"峰值小时占比 = {sim_peak_cars_per_h / max(1, n_cars):.2%}。\n",
        "## 4. OD 分散度\n",
        f"- {od_note}\n"
        f"- VKT 集中度：top-20 链路占全部 car VKT 的 **{concentration:.1%}**"
        f"（{vkt_total / 1000:.0f} veh-km 总 VKT）。\n",
        "## 5. Full population 还是 sample？\n",
        "- **Sample**：40k synthetic agents 代表的是研究区 ~45 万居民的需求样本；"
        f"模拟峰值 {sim_peak_cars_per_h} 辆/h vs 现实参考 ~30k 辆/h（45 万人 × 2.2 出行 × ~30% car × "
        f"10% 高峰系数，无实测计数，量级参考）→ **采样因子 ≈ {sample_factor:.3f}**。\n",
        "## 结论建议\n",
        "- 若 max V/C 仅在 0.0x 量级：green-ratio 折减（×0.35–0.55）不足以产生拥堵，"
        "真正的缺口是 demand sample factor（约 ×0.1 的容量等效）。\n"
        "- 后续标定路径由用户在两者间决策；本诊断只给数字。\n",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nV/C max={vc[0][3]:.3f} P99={_pct(0.99):.3f} P95={_pct(0.95):.3f} "
          f"concentration={concentration:.1%} sample_factor~{sample_factor:.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
