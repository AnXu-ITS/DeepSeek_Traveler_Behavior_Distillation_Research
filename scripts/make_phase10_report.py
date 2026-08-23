#!/usr/bin/env python
"""Assemble the Phase 10 feedback-loop report from phase10_results.json.

Usage:
    python scripts/make_phase10_report.py \
        --results data/phase10_loop/phase10_results.json \
        --report outputs/PHASE10_feedback_loop_report.md
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
    ap.add_argument("--report", default="outputs/PHASE10_feedback_loop_report.md")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    order = ["baseline", "rain", "fare_surge"]

    lines = []
    a = lines.append
    a("# Phase 10 报告：网络反馈闭环（Student ↔ MATSim）\n")
    a("> 蒸馏出行意图 · 个体行为蒸馏研究 · 800 personas × 1 次出行 × "
      "10×10 网格（per-link capacity 5 veh/h，为在开发规模人口下呈现拥堵而校准）\n")

    a("## 1. 闭环协议\n")
    a("每轮迭代：student 在 context（含 road_congestion 估计 c_ctx）下决策 → MATSim "
      "执行计划 → 从 linkstats 提取观测拥堵 c_obs（capacity 加权平均延误比 / 0.5，"
      "截断到 [0,1]）→ c_ctx 以平滑系数 0.5 更新 → 下一轮。收敛条件 "
      "|c_obs − c_ctx| < 0.02。\n")

    a("## 2. 收敛轨迹\n")
    for name in order:
        if name not in results:
            continue
        r = results[name]
        a(f"### {name}（converged={r['converged']}）\n")
        a("| iter | c_ctx | bike | car | pt | walk | c_obs | mean_delay_ratio |")
        a("|---|---|---|---|---|---|---|---|")
        for t in r["trace"]:
            sm = t["student_mode_share"]
            a(f"| {t['iteration']} | {_f(t['context_congestion'])} | "
              f"{_f(sm.get('bike', 0))} | {_f(sm.get('car', 0))} | "
              f"{_f(sm.get('pt', 0))} | {_f(sm.get('walk', 0))} | "
              f"{_f(t['observed_congestion'])} | {_f(t['mean_delay_ratio'])} |")
        a("")

    a("## 3. 平衡点对比\n")
    a("| 情景 | 初始 c | 收敛 c_ctx | 平衡 c_obs | car share 变化 |")
    a("|---|---|---|---|---|")
    for name in order:
        if name not in results:
            continue
        trace = results[name]["trace"]
        if not trace:
            continue
        first = trace[0]
        last = trace[-1]
        first_sm = first["student_mode_share"]
        last_sm = last["student_mode_share"]
        a(f"| {name} | {_f(first['context_congestion'])} | "
          f"{_f(last['context_congestion'])} | {_f(last['observed_congestion'])} | "
          f"{100*(last_sm.get('car', 0)-first_sm.get('car', 0)):+.2f}pp |")
    a("")

    a("## 4. 结论\n")
    a("（数据见上表；由实验者核对后填写。）\n")
    a("## 5. 复现命令\n")
    a("```powershell")
    a(".venv\\Scripts\\python.exe scripts\\run_phase10_loop.py "
      "--checkpoint outputs/student_v0_3_c/checkpoints/best.pt "
      "--num-personas 800 --trips-per-persona 1 "
      "--scenarios baseline,rain,fare_surge --max-iterations 6 --eps 0.02 "
      "--grid-n 10 --link-capacity 5 --output data/phase10_loop")
    a("```\n")
    a("## 6. 已知边界\n")
    a("- 合成网格 + 校准容量（cap=5 veh/h）：为在 development-scale 人口下")
    a("  产生可测拥堵而设定，非真实路网标定。")
    a("- 反馈只作用于 context.road_congestion（天气/票价固定）；student 的拥堵弹性")
    a("  较弱（v0.2 教师审计中拥堵响应亦为噪声最大轴）。")
    a("- 单次决策（无 within-day re-planning）；每轮迭代为独立一天。")

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
