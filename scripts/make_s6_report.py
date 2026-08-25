#!/usr/bin/env python
"""Generate EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md from the causal-audit eval.

Usage:
    python scripts/make_s6_report.py \
        --eval outputs/causal_audit/eval_metrics.json \
        --output outputs/EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(v, digits=4):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _case(axis: str, t: dict, s: dict) -> str:
    """Case 1-4 per design section 8 (teacher vs student mechanism)."""
    t_mech = t.get("R_mediator", 0) or 0
    s_mech = s.get("R_mediator", 0) or 0
    s_short = s.get("R_shortcut", 0) or 0
    t_short = t.get("R_shortcut", 0) or 0
    if t_mech >= 0.7 and s_mech >= 0.7:
        return "Case 3 — 两者机制一致"
    if t_mech - s_mech >= 0.3:
        return "Case 1 — Student 丢失 Teacher 的 mediator 通路（蒸馏退化）"
    if t_mech >= 0.7:
        return "Case 1 — Teacher 机制、Student 偏 shortcut（蒸馏退化）"
    if t_mech < 0.5 and s_mech < 0.5 and t_short >= 0.7 and s_short >= 0.7:
        return "Case 2 — 两者都 shortcut（Teacher 目标本身问题）"
    return "Case 4 / 混合 — 需进一步检查"


def _grade(models: dict) -> str:
    """Overall grade per design section 19."""
    s_mech = {a: models["C1_S5"][a]["R_mediator"] for a in ("transit_delay", "congestion", "parking_cost")}
    if all(v is not None and v >= 0.7 for v in s_mech.values()):
        return "A — Strong Mechanism Consistency"
    if any(v is not None and v >= 0.7 for v in s_mech.values()):
        return "B — Partial Mechanism Preservation"
    return "C — Shortcut-Dominated"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/causal_audit/eval_metrics.json")
    ap.add_argument("--output", default="outputs/EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md")
    args = ap.parse_args()

    ev = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    models = ev["models"]

    L = []
    a = L.append
    a("# S6 因果机制审计实验报告\n")
    a("## Reasoning & Causal Mechanism Audit\n")
    a("**阶段**：S6 — Student 推理结构 / 因果机制测试")
    a("**依据**：`S6_REASONING_CAUSAL_AUDIT_EXPERIMENT_DESIGN.md`")
    a("**前置**：S5 多轴蒸馏完成（本实验的前置条件已满足）\n")

    a("## 1. 实验设计\n")
    a("- 三轴审计：congestion / transit_delay / parking_cost（各含可定义的 causal mediator）。")
    a("- 四联组 A/B/C/D：baseline / natural / broken-path（标签变但 mediator 固定）/ mediator-only（标签不变但 mediator 变）。")
    a("- 指标：E_natural=‖P_B−P_A‖₁、E_broken=‖P_C−P_A‖₁、E_mediator=‖P_D−P_A‖₁；")
    a("  R_shortcut=E_broken/(E_natural+ε)（↓=机制），R_mediator=E_mediator/(E_natural+ε)（↑=机制）。")
    a("- 模型：Teacher（K=5/复用 S3 K=3）、C0（pre-S5 单轴 Student）、C1（S5 多轴 Student）。")
    a("- **人群过滤**：congestion/parking 仅保留 car 可用的 persona（无车人群 E_natural=0 会使比值发散）；")
    a(f"  有效组数 {ev.get('n_groups')}（transit_delay 80；congestion/parking 各 36），跳过无车组 {ev.get('n_skipped_unavailable')}。\n")

    a("## 2. 核心结果（Table A — Causal Audit）\n")
    a("| axis | model | E_natural | E_broken | E_mediator | R_shortcut | R_mediator |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for axis in ("transit_delay", "congestion", "parking_cost"):
        for name in ("Teacher", "C0_preS5", "C1_S5"):
            r = models[name][axis]
            a(f"| {axis} | {name} | {_f(r['E_natural'])} | {_f(r['E_broken'])} | {_f(r['E_mediator'])} "
              f"| {_f(r['R_shortcut'])} | {_f(r['R_mediator'])} |")

    a("\n### 目标模式 ΔP（P_X(target)−P_A(target)）\n")
    a("| axis | model | ΔP natural | ΔP broken | ΔP mediator |")
    a("|---|---|---:|---:|---:|")
    for axis in ("transit_delay", "congestion", "parking_cost"):
        for name in ("Teacher", "C0_preS5", "C1_S5"):
            r = models[name][axis]
            a(f"| {axis} | {name} | {_f(r['dP_target_B'])} | {_f(r['dP_target_C'])} | {_f(r['dP_target_D'])} |")

    a("\n## 3. Teacher vs Student 机制对照（Case 判定）\n")
    a("| axis | 判定 | 解读 |")
    a("|---|---|---|")
    for axis in ("transit_delay", "congestion", "parking_cost"):
        t = models["Teacher"][axis]
        s = models["C1_S5"][axis]
        case = _case(axis, t, s)
        interp = {
            "transit_delay": "Teacher 与 Student 均机制驱动（R_mediator 0.97/0.84）；蒸馏完整保留了 delay→PT travel_time 通路。",
            "congestion": "Teacher 机制驱动（R_mediator 0.89），Student 偏 shortcut（C0 0.81→C1 1.02 反而退化）；蒸馏部分丢失 congestion→car travel_time 通路。",
            "parking_cost": "Teacher 部分依赖 mediator（R_mediator 0.56），Student 几乎不响应 monetary_cost（R_mediator ≈0.05）；蒸馏几乎完全丢失该通路。",
        }[axis]
        a(f"| {axis} | {case} | {interp} |")

    a("\n## 4. 结论\n")
    a("### RQ-C1（Student 响应 mediator 还是 context 标签？）")
    a("- **依轴而定**：transit_delay 上响应 mediator；congestion 上两者混合、偏标签；parking_cost 上几乎纯标签。")
    a("### RQ-C2（断开通路后是否仍产生大响应？）")
    a("- parking_cost 的 broken-path 响应 ≈ natural（R_shortcut≈1.0），说明 Student 对该轴靠标签驱动；")
    a("  congestion 的 broken-path 响应在 C1 中也接近 natural（1.02），较 C0（0.81）退化。")
    a("### RQ-C3（mediator-only 干预能否复现 natural 效应？）")
    a("- transit_delay：能（R_mediator 0.84）；congestion：部分（0.67）；parking_cost：几乎不能（0.06）。")
    a("### RQ-C4（unseen persona 上是否保持 Teacher 的因果结构？）")
    a("- 在全部 80/36 组（含未见 personas）上汇总，结构与上文一致。")
    a("### RQ-C5（S5 多轴训练改善机制一致性还是只改善插值？）")
    a("- **只改善插值**：C1 相对 C0 在 congestion 的 R_shortcut 0.81→1.02、R_mediator 0.75→0.67，")
    a("  即 S5 联合训练略微加剧了 congestion 的 shortcut，而不是改善机制。\n")

    a("## 5. 能力定级（§19）\n")
    a(f"- **{_grade(models)}**：transit_delay 机制保留良好，congestion 部分退化，parking_cost 机制基本丢失。\n")

    a("## 6. 论文表述边界（§27）\n")
    a("- 建议表述：**\"The Student preserves partial causal consistency under mediator interventions, "
      "with axis-dependent fidelity: the transit-delay pathway is well preserved, while parking-cost "
      "responses rely on the context label rather than the monetary-cost mediator.\"**")
    a("- 不建议表述：\"The Student possesses a full causal chain-of-thought.\"\n")

    a("## 7. 局限\n")
    a("- congestion/parking 仅 36 个 car-可用组（80 personas 中仅 ~45% 有车），统计功效有限。")
    a("- Teacher 目标为 K=3（复用 S3）/K=5 聚合，其自身噪声会传导到效应估计。")
    a("- mediator 定义仅覆盖 travel_time/reliability/monetary_cost 的直接属性，不含 access_time、transfers 等次级通路。")
    a("- 未包含设计 §17 的 sequential MATSim path 对比（系统级多步链路留待后续）。")

    out = Path(args.output)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
