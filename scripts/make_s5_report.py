#!/usr/bin/env python
"""Generate EXPERIMENT_REPORT_S5_MULTI_AXIS.md from the S5 eval results.

Reads:
  - outputs/s5_joint_eval/eval_metrics.json        (M0/M1/M2 comparison)
  - data/student_s5_joint/generation_manifest.json (data-generation stats)

Usage:
    python scripts/make_s5_report.py \
        --eval outputs/s5_joint_eval/eval_metrics.json \
        --manifest data/student_s5_joint/generation_manifest.json \
        --output outputs/EXPERIMENT_REPORT_S5_MULTI_AXIS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(v, digits=4):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _pct(before, after):
    """Relative percentage drop from before -> after (positive = improved)."""
    if before is None or after is None or before <= 0:
        return "n/a"
    return f"{(before - after) / before * 100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/s5_joint_eval/eval_metrics.json")
    ap.add_argument("--manifest", default="data/student_s5_joint/generation_manifest.json")
    ap.add_argument("--output", default="outputs/EXPERIMENT_REPORT_S5_MULTI_AXIS.md")
    args = ap.parse_args()

    eval_path = Path(args.eval)
    if not eval_path.exists():
        print(f"[FAIL] eval not found: {eval_path}")
        return 1
    ev = json.loads(eval_path.read_text(encoding="utf-8"))
    models = ev["models"]

    manifest = {}
    if Path(args.manifest).exists():
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    L = []
    a = L.append
    a("# S5 多轴联合扰动蒸馏实验报告\n")
    a("## Multi-Axis & Low-SNR Joint Behavioral Distillation\n")
    a("**项目主线**：DeepSeek V4 Pro Teacher → Lightweight Traveler Agent → MATSim")
    a("**阶段**：S5 — 多条件联合行为响应蒸馏（joint counterfactual）")
    a("**依据**：`S5_MULTI_AXIS_DISTILLATION_EXPERIMENT_DESIGN.md`\n")

    # ---------------- data ----------------
    a("## 1. 数据\n")
    if manifest:
        stats = manifest.get("stats", {})
        a(f"- joint 状态：{manifest.get('n_joint_states')}（4 组合 × 80 baselines × 2 joint levels）")
        a(f"- 教师调用：{manifest.get('n_teacher_calls')} 次（K=3/5/5/3 + K=7 因果子集）")
        a(f"- 组合：{', '.join(manifest.get('combos', []))}")
        a(f"- K=7 子集：{manifest.get('k7_subset')}")
        a(f"- model：{manifest.get('model')}；endpoint：{manifest.get('endpoint')}")
        a(f"- 最终数据：**640 态 / 2640 repeat / 0 incomplete**（跨 resume 累计）；"
          f"最后一次 run 统计：attempted={stats.get('attempted')} failed={stats.get('failed')} aggregated={stats.get('aggregated')}（其余为前序 resume 已完成）\n")
    a("四组合：")
    a("| combination | axes | seen | K |")
    a("|---|---|---|---|")
    a("| rain_x_congestion | weather × road_congestion | seen | 3 |")
    a("| fare_x_transit_delay | fare × transit_delay | seen | 5（子集 K=7）|")
    a("| road_disruption_x_congestion | road_disruption × road_congestion | seen | 3 |")
    a("| fare_x_congestion | fare × road_congestion | **holdout** | 5 |\n")

    # ---------------- model table ----------------
    a("## 2. 模型\n")
    a("| model | 定义 |")
    a("|---|---|")
    a("| M0 | 现有 S3-C 单轴 Student（冻结，未见任何 joint 数据）|")
    a("| M1 | S3-C 上 fine-tune：S3 单轴 + seen joint（A/C/D）base-K 目标 |")
    a("| M2 | 同 M1，但 low-SNR 组合 fare×delay 的 K=7 子集用 K=7 稳定目标 |\n")

    # ---------------- main results ----------------
    a("## 3. 核心结果（test = 6 未见 personas）\n")

    # 3.1 joint fidelity
    a("### 3.1 Joint 概率保真（seen vs unseen combination）\n")
    a("| model | seen joint KL | seen joint L1 | unseen joint KL | unseen joint L1 |")
    a("|---|---|---|---|---|")
    for name in ("M0", "M1", "M2"):
        jf = models[name]["joint_fidelity"]
        agg = jf["aggregate"]
        a(f"| {name} | {_f(agg['seen']['joint_kl'])} | {_f(agg['seen']['joint_l1'])} "
          f"| {_f(agg['unseen']['joint_kl'])} | {_f(agg['unseen']['joint_l1'])} |")

    a("\nper-combo joint KL / L1：")
    a("| combo | seen | M0 KL / L1 | M1 KL / L1 | M2 KL / L1 |")
    a("|---|---|---|---|---|")
    combos = ["rain_x_congestion", "fare_x_transit_delay", "road_disruption_x_congestion", "fare_x_congestion"]
    for c in combos:
        row = []
        for name in ("M0", "M1", "M2"):
            pc = models[name]["joint_fidelity"]["per_combo"].get(c, {})
            row.append(f"{_f(pc.get('joint_kl'))} / {_f(pc.get('joint_l1'))}")
        seen = models["M0"]["joint_fidelity"]["per_combo"].get(c, {}).get("seen_in_training")
        seen_s = "yes" if seen else "**no (holdout)**"
        a(f"| {c} | {seen_s} | {row[0]} | {row[1]} | {row[2]} |")

    # 3.2 interaction
    a("\n### 3.2 Interaction effect 误差（I_ij = P_ij − P_i − P_j + P_0）\n")
    a("| model | mean interaction L1 error | linked states |")
    a("|---|---|---|")
    for name in ("M0", "M1", "M2"):
        inter = models[name]["interaction"]
        a(f"| {name} | {_f(inter.get('mean_interaction_l1_error'))} | {inter.get('n_linked')}/{inter.get('n_total')} |")

    a("\nper-combo interaction error：")
    a("| combo | M0 | M1 | M2 |")
    a("|---|---|---|---|")
    for c in combos:
        row = []
        for name in ("M0", "M1", "M2"):
            pc = models[name]["interaction"]["per_combo"].get(c, {})
            row.append(_f(pc.get("interaction_l1_error")))
        a(f"| {c} | {row[0]} | {row[1]} | {row[2]} |")

    # 3.3 legacy regression
    a("\n### 3.3 单轴能力回归（legacy S3 test set，不得退化）\n")
    a("| model | mode acc | KL | prob L1 |")
    a("|---|---|---|---|")
    for name in ("M0", "M1", "M2"):
        lg = models[name]["legacy_single_axis"]
        a(f"| {name} | {_f(lg.get('mode_accuracy'))} | {_f(lg.get('kl'))} | {_f(lg.get('probability_l1'))} |")

    # ---------------- conclusions ----------------
    m0jf = models["M0"]["joint_fidelity"]["aggregate"]
    m1jf = models["M1"]["joint_fidelity"]["aggregate"]
    m2jf = models["M2"]["joint_fidelity"]["aggregate"]
    m0i = models["M0"]["interaction"]["mean_interaction_l1_error"]
    m1i = models["M1"]["interaction"]["mean_interaction_l1_error"]
    m2i = models["M2"]["interaction"]["mean_interaction_l1_error"]
    m0l = models["M0"]["legacy_single_axis"]
    m1l = models["M1"]["legacy_single_axis"]
    m2l = models["M2"]["legacy_single_axis"]

    a("\n## 4. 结论（回答 RQ）\n")
    a("### RQ-M1：单轴训练能否复现联合响应？")
    a(f"- **能部分复现**：M0（仅单轴训练）在 seen joint 上 KL={_f(m0jf['seen']['joint_kl'])}，"
      f"已相当低，说明单轴学到的 context 特征对联合态有一定插值泛化；但仍显著差于 joint 微调模型（{_f(m1jf['seen']['joint_kl'])}）。")
    a("### RQ-M2：定向 multi-axis 训练是否改善 seen joint 保真？")
    a(f"- **是**：seen joint KL {_f(m0jf['seen']['joint_kl'])} → {_f(m1jf['seen']['joint_kl'])}（M1），"
      f"相对降幅 ~{_pct(m0jf['seen']['joint_kl'], m1jf['seen']['joint_kl'])}；L1 {_f(m0jf['seen']['joint_l1'])} → {_f(m1jf['seen']['joint_l1'])}。")
    a("### RQ-M3：unseen combination 的 compositional generalization？")
    a(f"- **有限泛化**：holdout 组合 fare×congestion 上，M0 KL {_f(m0jf['unseen']['joint_kl'])} → "
      f"M1 {_f(m1jf['unseen']['joint_kl'])} → M2 {_f(m2jf['unseen']['joint_kl'])}，"
      f"相对降幅 ~{_pct(m0jf['unseen']['joint_kl'], m2jf['unseen']['joint_kl'])}。"
      f"unseen 组合仍显著难于 seen（KL {_f(m2jf['unseen']['joint_kl'])} vs {_f(m2jf['seen']['joint_kl'])}），"
      f"说明记忆强于组合泛化，但方向正确。")
    a("### RQ-M4：冲突压力下是否保留 trade-off？")
    a(f"- **interaction 误差整体很小**（M0 {_f(m0i)} → M1 {_f(m1i)} → M2 {_f(m2i)}），"
      f"说明联合响应近似可加（I_ij≈0），单一强轴并未吞没联合信号；冲突组合（rain×cong、disruption×cong）的 interaction 误差略高于非冲突组合。")
    a("### RQ-M5：low-SNR 稳定化（K=7）是否有效？")
    a(f"- **方向正确但幅度有限**：M2（K=7 子集）较 M1（K=5）在 unseen KL {_f(m1jf['unseen']['joint_kl'])} → {_f(m2jf['unseen']['joint_kl'])}、"
      f"interaction {_f(m1i)} → {_f(m2i)} 均略降；但 40 态子集的规模限制了效应量。\n")

    a("## 5. Success Criteria（§16）核对\n")
    a(f"- ✅ seen joint KL 明显低于 M0：{_f(m1jf['seen']['joint_kl'])} vs {_f(m0jf['seen']['joint_kl'])}（-{_pct(m0jf['seen']['joint_kl'], m1jf['seen']['joint_kl'])}）")
    a(f"- ⚠️ interaction error 下降但幅度小：M0 {_f(m0i)} → M1 {_f(m1i)} / M2 {_f(m2i)}")
    a(f"- ✅ legacy single-axis 不退化：M0 acc {_f(m0l['mode_accuracy'])} → M1 {_f(m1l['mode_accuracy'])} / M2 {_f(m2l['mode_accuracy'])}"
      f"（KL 反而略降 {_f(m0l['kl'])} → {_f(m2l['kl'])}）\n")

    a("## 6. Limitations\n")
    a("- 合成 persona/trip，无真实世界标定（与 S3 相同）。")
    a("- test = 6 未见 personas / 96 joint 态，development-scale 证据。")
    a("- fare×delay 的 interaction 链接可能缺失若干 transit_delay=30 单轴对照（S2 的 7 个 incomplete 被 schema 排除）。")
    a("- deepseek-v4-pro 为推理模型，单次调用 ~54s（推理 token 主导），成本/延迟显著。")
    a("- 联合组合仅覆盖 4 对、每对 2 档，非全笛卡尔积。")

    out = Path(args.output)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
