#!/usr/bin/env python
"""Assemble the Phase 9 population-scale report from population_results.json.

Usage:
    python scripts/make_phase9_report.py \
        --results data/population_experiment/population_results.json \
        --report outputs/PHASE9_population_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODES = ["bike", "car", "pt", "walk"]


def _f(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--report", default="outputs/PHASE9_population_report.md")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    order = ["baseline", "rain", "fare_surge", "combined"]

    n_personas = None
    for name in order:
        if name in results:
            n_personas = results[name].get("n_personas")
            break

    lines = []
    a = lines.append
    a("# Phase 9 报告：人口级仿真（蒸馏 Student × MATSim）\n")
    a(f"> 蒸馏出行意图 · 个体行为蒸馏研究 · {n_personas or '?'} personas × 每人 2 次出行\n")

    a("## 1. 情景\n")
    a("| 情景 | 天气 | 票价 |")
    a("|---|---|---|")
    for name in order:
        if name not in results:
            continue
        ctx = results[name]["context"]
        a(f"| {name} | intensity={ctx['weather_intensity']} | x{ctx['fare_multiplier']} |")
    a("")

    a("## 2. 人口级 mode share（Student 决策 = MATSim 执行，lastIteration=0）\n")
    a("| 情景 | bike | car | pt | walk |")
    a("|---|---|---|---|---|")
    for name in order:
        if name not in results:
            continue
        sm = results[name]["student_mode_share"]
        mm = results[name]["matsim_mode_share"] or {}
        a(f"| {name} (student) | {_f(sm.get('bike', 0))} | {_f(sm.get('car', 0))} | "
          f"{_f(sm.get('pt', 0))} | {_f(sm.get('walk', 0))} |")
        a(f"| {name} (executed) | {_f(mm.get('bike', 0))} | {_f(mm.get('car', 0))} | "
          f"{_f(mm.get('pt', 0))} | {_f(mm.get('walk', 0))} |")
    a("")

    a("## 3. 需求转移（vs baseline，百分点）\n")
    base = results.get("baseline", {}).get("student_mode_share", {})
    a("| 情景 | bike | car | pt | walk |")
    a("|---|---|---|---|---|")
    for name in order:
        if name == "baseline" or name not in results:
            continue
        sm = results[name]["student_mode_share"]
        a(f"| {name} | {100*(sm.get('bike',0)-base.get('bike',0)):+.1f}pp | "
          f"{100*(sm.get('car',0)-base.get('car',0)):+.1f}pp | "
          f"{100*(sm.get('pt',0)-base.get('pt',0)):+.1f}pp | "
          f"{100*(sm.get('walk',0)-base.get('walk',0)):+.1f}pp |")
    a("")

    a("## 4. 出行距离（米）\n")
    a("| 情景 | avg leg | avg trip |")
    a("|---|---|---|")
    for name in order:
        if name not in results:
            continue
        r = results[name]
        a(f"| {name} | {_f(r.get('avg_leg_distance_m'), 1)} | {_f(r.get('avg_trip_distance_m'), 1)} |")
    a("")

    a("## 5. 结论\n")
    a("（数据见上表；由实验者核对后填写。）\n")
    a("## 6. 复现命令\n")
    a("```powershell")
    a(".venv\\Scripts\\python.exe scripts\\run_population_experiment.py "
      "--checkpoint outputs/student_v0_3_c/checkpoints/best.pt "
      "--num-personas 1000 --trips-per-persona 2 "
      "--scenarios baseline,rain,fare_surge,combined "
      "--output data/population_experiment")
    a("```\n")
    a("## 7. 已知边界\n")
    a("- 合成网格网络（20×20），pt/walk/bike 为 teleported 模式（无公交时刻表）。")
    a("- lastIteration=0：MATSim 不做 replanning，测量的是 student 决策的一阶需求转移；")
    a("  网络拥堵反馈闭环（Phase 10）尚未接入。")
    a("- 1000 personas 为 development-scale 人口（真实城市场景为数十万级）。")
    a("- 过程注记：初版脚本曾把全部 trip 列表复用于每个 persona（人口文件放大 ~1000 倍，")
    a("  1000 人时 MATSim 报 runners-null NPE）；修复为每人独立 2 次出行后，")
    a("  1000 人 × 4 情景全部正常运行。")

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
