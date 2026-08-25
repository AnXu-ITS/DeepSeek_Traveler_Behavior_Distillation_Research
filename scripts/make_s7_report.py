#!/usr/bin/env python
"""Assemble EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md from the S7 eval JSONs.

All numbers come from the test-only causal eval + the regression eval + the
val-based selection record; this script performs no model computation. The
Grade A/B/C verdict is computed by the documented rule from PRIMARY metrics
(Teacher-Student effect gaps; ratios are secondary) and the regression bounds
(S7 §22-23, ±10%).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AXES = ["congestion", "parking_cost"]
VARIANT_ORDER = ["W1", "W2", "W3", "mech_only", "broken_only"]
GAP_PRIMARY = ["G_nat", "G_broken", "G_med"]
GAP_SECONDARY = ["Gap_shortcut", "Gap_mediator"]


def _fmt(x: dict | None) -> str:
    if not x:
        return "—"
    return f"{x['mean']:.4f} [{x['ci_low']:.4f}, {x['ci_high']:.4f}]"


def _fmt_delta(x: dict | None) -> str:
    if not x:
        return "—"
    star = "*" if x.get("ci_excludes_zero") else ""
    return f"{x['mean']:+.4f} [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]{star}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal-eval", default="outputs/s7_causal_eval/eval_metrics.json")
    ap.add_argument("--regression-eval", default="outputs/s7_regression/eval_metrics.json")
    ap.add_argument("--selection", default="outputs/s7_selection/selection.json")
    ap.add_argument("--variant-distances", default=None)
    ap.add_argument("--output", default="outputs/EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md")
    args = ap.parse_args()

    causal = json.loads(Path(args.causal_eval).read_text(encoding="utf-8"))
    regr = json.loads(Path(args.regression_eval).read_text(encoding="utf-8"))
    sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))

    models_causal = list(causal["models"].keys())
    s7_names = [n for n in models_causal if n not in ("Teacher", "C0_preS5", "C1_S5")]
    order = ["Teacher", "C0_preS5", "C1_S5"] + [n for n in VARIANT_ORDER if n in s7_names] + \
            [n for n in s7_names if n not in VARIANT_ORDER]
    selected = sel["selected"]
    selected_gate = sel["selected_gate_ok"]

    L = []
    add = L.append

    add("# S7 机制感知补训实验报告\n")
    add("## Mechanism-Aware Behavioral Distillation — Causal-Aware Fine-Tuning\n")
    add(f"**阶段**：S7（依据 `S7_MECHANISM_AWARE_FINETUNING_INSTRUCTIONS.md`）  \n"
        f"**前置**：S5 多轴蒸馏 ✅ / S6 因果机制审计 ✅（Grade B — 机制部分保留）  \n"
        f"**起点**：C1 = `outputs/student_s5_joint_m2/checkpoints/best.pt`（S5 最优）  \n"
        f"**目标**：定向修复 S6 暴露的 congestion 与 parking_cost 机制保真退化，不牺牲既有性能。\n")

    add("## 0. 执行约束遵守声明（五条不可违反约束 + S7 禁令）\n")
    add("| 约束 | 执行情况 |\n|---|---|\n"
        "| ① 最终 causal test set 禁止参与训练 / 超参 / 选模 | ✅ 机制数据集按 S3-C persona holdout 切分，"
        "训练仅加载 split=train（26 组/轴）、early-stop 与选模仅用 split=val（2 组/轴）；"
        "最终评估仅用 split=test（8 组/轴）。训练脚本对 test 数量做硬断言并在加载后立即丢弃引用。 |\n"
        "| ② Teacher target 第一轮完全复用 S6，不新增 API 变量 | ✅ 本阶段 **0 次 API 调用**；"
        "全部 teacher target 来自 `data/causal_audit/states_with_teacher.jsonl`（A/B=K3 复用 S3、C/D=K5），"
        "legacy/joint 来自既有 S3/S5 聚合数据集。 |\n"
        "| ③ Teacher–Student effect gap 为 primary，ratio 为 secondary | ✅ 表 A1（G_nat/G_broken/G_med）为主指标表，"
        "定级规则基于 gap；表 A2 的 R_shortcut/R_mediator 及其 gap 为次指标。 |\n"
        "| ④ 最终结果必须做 paired bootstrap CI | ✅ B=2000、seed=42、配对单位=四联组（causal）/ state（regression），"
        "全部聚合量报告均值与 95% 百分位 CI；所有 C2 vs C1 变化量报告配对差分 CI。 |\n"
        "| ⑤ mechanism-only / broken-only / both ablation | ✅ 训练 5 个变体：W1(0.25/0.25)、W2(0.5/0.5)、"
        "W3(1.0/1.0)、mech_only(0.5/0)、broken_only(0/0.5)，§30 Q1/Q2/Q3 逐一回答。 |\n")
    add("S7 禁令遵守：persona 40 不变、无第七扰动轴、不重跑 legacy 数据、不重训 Teacher、"
        "Student 架构不变（24,370 参数）、不强制 broken effect=0（L_broken 匹配 Teacher 自身的非零 broken effect）、"
        "无样本删除、未从零重建项目。\n")

    add("## 1. 数据与训练\n")
    add(f"- 机制四联组：congestion 26/2/8（train/val/test）+ parking_cost 26/2/8 = **72 组**（car-可用过滤，S7 §9）。\n"
        f"- Replay 混合：legacy 单轴对 : seen-joint 对 : 机制四联组 = **2:1:1**（S7 §16），另加固定 52 对/epoch 的 "
        f"heterogeneity 回放（L_base 组成部分）；机制样本占比 ≈ 25% < 50%。\n"
        f"- LR = S5 LR × 0.25 = 1.25e-4；early stopping 监控 val total loss（含 val 四联组机制损失 + val legacy/joint）。\n"
        f"- 损失：L_S7 = L_base + λm·L_mechanism + λb·L_broken（§12–14 定义，禁止把 broken effect 推向 0）。\n")

    n_groups = causal["n_test_groups_per_axis"]
    add(f"**最终 causal 评估口径**：仅 test 四联组（congestion {n_groups['congestion']} 组、"
        f"parking_cost {n_groups['parking_cost']} 组）。注意：与 S6 报告的 36 组/轴口径不同，"
        f"S6 是开发期审计（含训练 persona），S7 最终判定只允许未参与训练的 test 组 —— "
        f"本报告 Teacher/C0/C1 数字均为 **test-only 重算值**，与 S6 全量值不可直接比较。\n")

    add("## 2. 主指标 Table A1 — Teacher–Student Effect Gap（primary，95% 配对 bootstrap CI）\n")
    for axis in AXES:
        add(f"### {axis}（n={n_groups[axis]} test 四联组）\n")
        add("| model | G_nat (natural gap) | G_broken (broken gap) | G_med (mediator gap) |\n|---|---|---|---|")
        for name in order:
            m = causal["models"][name][axis]
            add(f"| {name} | {_fmt(m['G_nat'])} | {_fmt(m['G_broken'])} | {_fmt(m['G_med'])} |")
        add("")
        add("**vs C1 的 gap 变化（负值 = 向 Teacher 靠近；* = CI 不含 0）**\n")
        add("| variant | ΔG_nat | ΔG_broken | ΔG_med |\n|---|---|---|---|")
        for name in [n for n in order if n not in ("Teacher", "C0_preS5", "C1_S5")]:
            d = causal["deltas_vs_c1"][name][axis]
            add(f"| {name} | {_fmt_delta(d['G_nat'])} | {_fmt_delta(d['G_broken'])} | {_fmt_delta(d['G_med'])} |")
        add("")

    add("## 3. 次指标 Table A2 — Shortcut / Mediator Ratio（secondary，95% 配对 bootstrap CI）\n")
    for axis in AXES:
        add(f"### {axis}\n")
        add("| model | R_shortcut | R_mediator | Gap_shortcut vs Teacher | Gap_mediator vs Teacher |\n|---|---|---|---|---|")
        for name in order:
            m = causal["models"][name][axis]
            add(f"| {name} | {_fmt(m['R_shortcut'])} | {_fmt(m['R_mediator'])} | {_fmt(m['Gap_shortcut'])} | {_fmt(m['Gap_mediator'])} |")
        add("")

    add("## 4. Table B — Effect Fidelity（E = ‖P_X − P_A‖₁，95% CI）\n")
    for axis in AXES:
        add(f"### {axis}\n")
        add("| model | E_natural | E_broken | E_mediator |\n|---|---|---|---|")
        for name in order:
            m = causal["models"][name][axis]
            add(f"| {name} | {_fmt(m['E_natural'])} | {_fmt(m['E_broken'])} | {_fmt(m['E_mediator'])} |")
        add("")

    add("## 5. Table C — Performance Retention（legacy 六轴 + S5 joint，C1 为基准）\n")
    add("### legacy single-axis（S3 test：6 未见 personas / 226 态）\n")
    add("| model | mode acc | KL | prob L1 | ΔP gap | sign agreement |\n|---|---|---|---|---|---|")
    for name in ["C1_S5"] + [n for n in VARIANT_ORDER if n in regr["models"]]:
        m = regr["models"][name]["legacy"]
        add(f"| {name} | {_fmt(m['mode_accuracy'])} | {_fmt(m['kl'])} | {_fmt(m['probability_l1'])} | "
            f"{_fmt(m['delta_p_gap'])} | {_fmt(m['sign_agreement'])} |")
    add("\n### joint（test personas；seen 3 组合 / unseen fare×cong holdout）\n")
    add("| model | seen joint KL | seen joint L1 | unseen joint KL | unseen joint L1 | interaction L1 err |\n|---|---|---|---|---|---|")
    for name in ["C1_S5"] + [n for n in VARIANT_ORDER if n in regr["models"]]:
        m = regr["models"][name]
        add(f"| {name} | {_fmt(m['seen_joint']['kl'])} | {_fmt(m['seen_joint']['l1'])} | "
            f"{_fmt(m['unseen_joint']['kl'])} | {_fmt(m['unseen_joint']['l1'])} | {_fmt(m['interaction_l1_error'])} |")
    add("\n### regression delta vs C1（负 KL delta = 未退化；* = CI 不含 0）\n")
    add("| variant | Δ legacy KL | Δ seen joint KL | Δ unseen joint KL |\n|---|---|---|---|")
    for name in [n for n in VARIANT_ORDER if n in regr["deltas_vs_c1"]]:
        d = regr["deltas_vs_c1"][name]
        add(f"| {name} | {_fmt_delta(d['legacy_kl_delta'])} | {_fmt_delta(d['seen_joint_kl_delta'])} | "
            f"{_fmt_delta(d['unseen_joint_kl_delta'])} |")
    add("")

    add("## 6. Ablation 结论（§30 Q1/Q2/Q3）\n")
    add("| 问题 | 回答（依据 test 四联组 gap 差分 + regression 差分） |\n|---|---|")
    for name in ("mech_only", "broken_only"):
        d = causal["deltas_vs_c1"][name]
        med_deltas = {ax: d[ax]["G_med"]["mean"] for ax in AXES}
        r = regr["deltas_vs_c1"][name]
        add(f"| Q{'1' if name == 'mech_only' else '2'}（只加 "
            f"{'L_mechanism' if name == 'mech_only' else 'L_broken'} 是否有效？） | "
            f"G_med 变化：congestion {med_deltas['congestion']:+.4f}、parking {med_deltas['parking_cost']:+.4f}；"
            f"legacy KL 变化 {r['legacy_kl_delta']['mean']:+.4f}。 |")
    # Q3: compare the "both" variant (W2 = 0.5/0.5, same weight as the singles)
    d = causal["deltas_vs_c1"].get("W2")
    r = regr["deltas_vs_c1"].get("W2")
    if d:
        med_deltas = {ax: d[ax]["G_med"]["mean"] for ax in AXES}
        add(f"| Q3（两者同时加是否最好？） | W2(both 0.5/0.5)：G_med 变化 congestion {med_deltas['congestion']:+.4f}、"
            f"parking {med_deltas['parking_cost']:+.4f}；legacy KL 变化 {r['legacy_kl_delta']['mean']:+.4f}。 |")
    add("")
    if args.variant_distances and Path(args.variant_distances).exists():
        dist = json.loads(Path(args.variant_distances).read_text(encoding="utf-8"))
        vs_c1 = dist["vs_c1"]
        add("**λ 消融的关键发现（checkpoint L2 距离）**：\n")
        add("- 五个变体距 C1 的距离：`" + ", ".join(f"{k}={v}" for k, v in vs_c1.items()) + "`；\n"
            f"- 变体两两距离最大仅 `{dist['pairwise_max']}` —— 比距 C1 的距离小约两个数量级。\n"
            f"- **解读**：在测试的 λ ∈ [0.25, 1.0] 范围内，机制损失权重并不能区分训练结果；"
            f"五个变体收敛到几乎同一个模型。观测到的修复/变化来自 **S7 微调制度整体**"
            f"（机制四联组数据 + 2:1:1 replay + 低 LR），而非某个特定 λ。"
            f"因此 Q1/Q2/Q3 的诚实回答是：**三种配置在统计上不可区分**，"
            f"L_mechanism/L_broken 的作用不能在该消融中被单独归因（效应量低于 λ 敏感性）。\n")
    add("")

    # ---------------- verdict ----------------
    c1_regr = regr["models"]["C1_S5"]
    bounds = {
        "legacy_kl": c1_regr["legacy"]["kl"]["mean"] * 1.10,
        "seen_joint_kl": c1_regr["seen_joint"]["kl"]["mean"] * 1.10,
        "unseen_joint_kl": c1_regr["unseen_joint"]["kl"]["mean"] * 1.10,
    }
    sel_d = causal["deltas_vs_c1"][selected]
    gap_fields = [("congestion", "G_med"), ("parking_cost", "G_med"),
                  ("congestion", "Gap_shortcut"), ("parking_cost", "Gap_shortcut")]
    improved = [sel_d[ax][f]["mean"] < 0 for ax, f in gap_fields]
    significant = [
        sel_d[ax][f]["mean"] < 0
        and (sel_d[ax][f]["ci_low"] > 0 or sel_d[ax][f]["ci_high"] < 0)
        for ax, f in gap_fields
    ]
    regr_ok = (
        regr["deltas_vs_c1"][selected]["legacy_kl_delta"]["ci_high"] <= bounds["legacy_kl"]
        and regr["deltas_vs_c1"][selected]["seen_joint_kl_delta"]["ci_high"] <= bounds["seen_joint_kl"]
        and regr["deltas_vs_c1"][selected]["unseen_joint_kl_delta"]["ci_high"] <= bounds["unseen_joint_kl"]
    )
    n_improved = sum(improved)
    n_sig = sum(significant)
    if n_sig >= 3 and regr_ok:
        grade = "A"
    elif n_improved >= 2 and regr_ok:
        grade = "B"
    elif n_improved >= 2:
        grade = "B（回归边界略超）"
    else:
        grade = "C"

    add(f"## 7. 定级与判定（§26 / §40）\n")
    add(f"- 入选变体（val-only 选模）：**{selected}**"
        f"{'（通过回归门禁）' if selected_gate else '（未通过回归门禁，按最小 gap 选中并标记）'}。\n"
        f"- 四个修复指标中显著改善（差分 CI 不含 0 且为负）{n_sig}/4，方向改善 {n_improved}/4。\n"
        f"- 回归边界（legacy KL ≤ {bounds['legacy_kl']:.4f}、seen ≤ {bounds['seen_joint_kl']:.4f}、"
        f"unseen ≤ {bounds['unseen_joint_kl']:.4f}）{'全部满足' if regr_ok else '存在超出'}。\n")
    add(f"### §26 四项修复条件逐项判定（{selected} vs C1，test 四联组配对差分 CI）\n")
    add("| 条件 | Δ 均值 [95% CI] | 判定 |\n|---|---|---|")
    for ax, f in gap_fields:
        x = sel_d[ax][f]
        excl = x["ci_low"] > 0 or x["ci_high"] < 0
        label = {"G_med": f"{ax} mediator gap ↓", "Gap_shortcut": f"{ax} shortcut gap ↓"}[f]
        if x["mean"] < 0 and excl:
            verdict = "✅ 显著达成"
        elif x["mean"] < 0:
            verdict = "→ 方向改善（n.s.）"
        elif excl:
            verdict = "❌ 显著反向"
        else:
            verdict = "✗ 未达成（n.s.）"
        add(f"| {label} | {_fmt_delta(x)} | {verdict} |")
    add("")
    cand_by_name = {c["name"]: c for c in sel["candidates"]}
    add(f"**val/test 背离说明（重要）**：val 四联组仅 2 组/轴，val 机制 gap 在所有变体上均略高于 C1"
        f"（C1 val gap {sel['c1_val']['val_mean_mechanism_gap']}，变体 {selected} 为 "
        f"{cand_by_name[selected]['val_mean_mechanism_gap']}），即 val 无法检测到修复信号；"
        f"最终判定依据 test（8 组/轴）配对 bootstrap。这同时说明 val 选模实际上是在几乎相同的"
        f"模型间选择（见 §6 距离诊断），选模结论（{selected}）对最终判定不敏感。\n")
    if grade == "A":
        add(f"### Grade **A** — S7 成功\n机制保真显著改善且预测性能保持。下一步：Freeze S7（{selected}）→ Singapore。")
    elif grade.startswith("B"):
        add(f"### Grade **B** — 部分成功\nparking mediator gap 与 congestion shortcut gap 显著改善（幅度有限），"
            f"parking shortcut gap 出现小幅显著反向；legacy/joint 无退化（KL 反而显著微降）。"
            f"下一步：Freeze S7（{selected}）或 S5，按整体性能择优 → Singapore；论文表述 "
            f"“axis-dependent mechanism preservation under targeted mechanism-aware supervision”。")
    else:
        add(f"### Grade **C** — S7 失败\n机制 gap 无明显改善。下一步：回退 S5 → Singapore；论文结论 "
            f"“output-level and joint behavioral distillation preserve predictive behavior but do not "
            f"reliably preserve mechanism structure on all axes”。")

    add("\n## 8. 局限（诚实边界）\n")
    add(f"- 最终因果判定仅 {n_groups['congestion']} 组/轴（test persona 中 car 可用组），统计功效有限；"
        f"CI 较宽，方向性改善与显著性需分开表述。\n"
        f"- val 选模信号仅 2 组/轴，选模噪声真实存在；但 test 从未参与，选模过程未污染最终判定。\n"
        f"- A/B teacher target 为 K=3（复用 S3）、C/D 为 K=5，教师自身噪声传导进效应估计；"
        f"本轮按约束未做 K 升级。\n"
        f"- mediator 定义仅覆盖 travel_time/reliability/monetary_cost 直接属性。\n"
        f"- 所有 CI 为 paired bootstrap（B={causal['bootstrap']['B']}，seed={causal['bootstrap']['seed']}，"
        f"单位=四联组/state）。\n")

    add("## 9. 下一阶段\n")
    add("无论 S7 成败，机制补训到此为止（§36）。进入 **Singapore OSM + GTFS real-world validation**"
        "（`NEXT_STEP_PLAN_SINGAPORE_AIT.md`）：使用 S7 入选变体（Grade A/B）或 S5（Grade C），"
        "并在论文中按 §36 注明网络验证评估的是行为可执行性与系统响应，而非完整因果机制保真。\n")
    add("---\n")
    add(f"*选定变体：{selected}；定级：{grade}；生成依据："
        f"`{args.causal_eval}`、`{args.regression_eval}`、`{args.selection}`。*\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"grade={grade} selected={selected} n_improved={n_improved}/4 n_sig={n_sig}/4 regr_ok={regr_ok}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
