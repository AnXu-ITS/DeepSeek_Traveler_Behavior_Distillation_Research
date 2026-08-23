#!/usr/bin/env python
"""Assemble the final v0.3 experiment report from training outputs.

Combines the teacher-side analysis (analyze_teacher_dataset.py), the three
student runs' evaluation metrics and split manifests, and produces a single
markdown report with the A/B/C comparison and persona-holdout generalization
assessment.

Usage:
    python scripts/make_v0_3_report.py \
        --teacher-analysis outputs/teacher_analysis_v0_3.json \
        --outputs outputs \
        --report outputs/EXPERIMENT_REPORT_v0_3.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_ROWS = [
    ("mode_accuracy", "mode accuracy"),
    ("kl", "KL(P_T || P_S)"),
    ("probability_l1", "probability L1"),
    ("departure_mae", "departure MAE (min)"),
    ("departure_sign_agreement", "departure sign agreement"),
]
ELASTICITY_ROWS = [
    ("delta_p_l1", "counterfactual |dP_T - dP_S|"),
    ("sign_agreement", "counterfactual sign agreement"),
    ("heterogeneity", "heterogeneity |dP_T - dP_S| (C only)"),
]


def _fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _load_run(outputs: Path, name: str) -> dict:
    m = json.loads((outputs / name / "evaluation_metrics.json").read_text(encoding="utf-8"))
    s = json.loads((outputs / name / "split_manifest.json").read_text(encoding="utf-8"))
    return {
        "metrics": m,
        "split": s,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-analysis", required=True)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--report", default="outputs/EXPERIMENT_REPORT_v0_3.md")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    ta = json.loads(Path(args.teacher_analysis).read_text(encoding="utf-8"))

    runs = {lbl: _load_run(outputs, f"student_v0_3_{lbl.lower()}") for lbl in ("A", "B", "C")}

    lines = []
    a = lines.append
    a("# v0.3 实验报告：官方直连 DeepSeek V4 Pro 蒸馏（persona-holdout）\n")
    a("> 蒸馏出行意图 · 个体行为蒸馏研究 · 生成日期：2026-08-21\n")

    # ---- teacher dataset ----
    ov = ta["overview"]
    a("## 1. 教师数据集\n")
    a(f"- 规模：{ov['n_samples']} states = {ov['n_baselines']} baselines + "
      f"{ov['n_counterfactuals']} counterfactuals")
    a(f"- 构成：{ov['n_personas']} personas × {ov['n_trips']} trips × "
      f"{'、'.join(ov['axes'])}")
    a(f"- 教师 selected-mode 分布：{ov['selected_mode_distribution']}")
    noise = ta.get("teacher_sampling_noise") or {}
    if noise:
        a(f"- 教师采样噪声（K=3 repeat 的 per-state pairwise L1）："
          f"mean = {noise.get('mean_pairwise_l1')}，max = {noise.get('max_pairwise_l1')} "
          f"—— 这是 Student 分布误差的合理下限参考\n")

    het = ta.get("baseline_heterogeneity") or {}
    if het:
        a("### 1.1 教师人群异质性（baseline 概率跨 persona 的 std）\n")
        a("| mode | mean P | std | n |")
        a("|---|---|---|---|")
        for m, v in het.items():
            a(f"| {m} | {v['mean']} | {v['std']} | {v['n']} |")
        a("")

    curves = ta.get("per_axis_response_curves") or {}
    if curves:
        a("### 1.2 教师逐轴响应（mean ΔP vs baseline）\n")
        for axis, rows in curves.items():
            a(f"**{axis}**\n")
            for r in rows:
                deltas = ", ".join(f"{m} {d:+.3f}" for m, d in sorted(r["mean_delta_p_vs_baseline"].items()))
                a(f"- level {r['level']} (n={r['n']}): {deltas}")
            a("")

    # ---- splits ----
    a("## 2. 切分（persona-holdout）\n")
    a("| split | A | B | C |")
    a("|---|---|---|---|")
    for field in ("counts", "persona_overlap"):
        row = "| " + field + " |"
        for lbl in ("A", "B", "C"):
            v = runs[lbl]["split"].get(field)
            row += f" {v} |"
        a(row)
    for lbl in ("A", "B", "C"):
        personas = runs[lbl]["split"].get("personas", {})
        a(f"\n- **{lbl}** personas: train={sorted(personas.get('train', []))} "
          f"val={sorted(personas.get('val', []))} test={sorted(personas.get('test', []))}")
    a("")

    # ---- test metrics ----
    a("## 3. 测试集指标（test = 完全未见的 personas）\n")
    a("| metric | A | B | C |")
    a("|---|---|---|---|")
    for key, name in METRIC_ROWS:
        vals = [runs[lbl]["metrics"]["test"].get(key) for lbl in ("A", "B", "C")]
        a(f"| {name} | {_fmt(vals[0])} | {_fmt(vals[1])} | {_fmt(vals[2])} |")
    a("")

    a("### 3.1 行为弹性与异质性（test）\n")
    a("| metric | A | B | C |")
    a("|---|---|---|---|")
    vals_a = runs["A"]["metrics"].get("counterfactual_preview", {}).get("mean_abs_delta_p_diff")
    vals_b = runs["B"]["metrics"].get("counterfactual_preview", {}).get("mean_abs_delta_p_diff")
    vals_c = runs["C"]["metrics"].get("counterfactual_preview", {}).get("mean_abs_delta_p_diff")
    a(f"| counterfactual \\|dP_T - dP_S\\| | {_fmt(vals_a)} | {_fmt(vals_b)} | {_fmt(vals_c)} |")
    sa_a = runs["A"]["metrics"].get("counterfactual_sign_agreement", {}).get("mean_sign_agreement")
    sa_b = runs["B"]["metrics"].get("counterfactual_sign_agreement", {}).get("mean_sign_agreement")
    sa_c = runs["C"]["metrics"].get("counterfactual_sign_agreement", {}).get("mean_sign_agreement")
    a(f"| counterfactual sign agreement | {_fmt(sa_a)} | {_fmt(sa_b)} | {_fmt(sa_c)} |")
    het_c = runs["C"]["metrics"].get("heterogeneity_preview", {}).get("mean_abs_delta_p_diff")
    a(f"| heterogeneity \\|dP_T - dP_S\\| | n/a | n/a | {_fmt(het_c)} |")
    a("")

    # ---- persona breakdown ----
    a("### 3.2 逐 persona 测试明细（未见人群泛化）\n")
    a("| persona | n | A acc | A L1 | B acc | B L1 | C acc | C L1 |")
    a("|---|---|---|---|---|---|---|---|")
    pb = {}
    for lbl in ("A", "B", "C"):
        pb[lbl] = {row["persona_id"]: row for row in runs[lbl]["metrics"].get("persona_breakdown", [])}
    all_personas = sorted({p for lbl in ("A", "B", "C") for p in pb[lbl]})
    for p in all_personas:
        row = f"| {p} |"
        d0 = pb["A"].get(p)
        row += f" {d0['n_samples']} |" if d0 else " n/a |"
        for lbl in ("A", "B", "C"):
            d = pb[lbl].get(p)
            if d is None:
                row += " n/a | n/a |"
            else:
                row += f" {d['mode_accuracy']} | {d['probability_l1']} |"
        a(row)
    a("")

    # ---- conclusions ----
    a("## 4. 结论\n")
    a("（训练完成后由实验者填写；表格数据见上。）\n")
    a("## 5. 复现命令\n")
    a("```powershell")
    a(".venv\\Scripts\\python.exe scripts\\run_v0_3_experiment.py "
      "--dataset data/student_v0_3/aggregated_teacher_dataset.jsonl "
      "--repeats data/student_v0_3/repeat_records.jsonl --outputs outputs")
    a("```\n")
    a("## 6. 已知边界\n")
    a("- 合成 persona/trip，无真实世界标定；development-scale 数据集。")
    a("- v0.3-B 的弹性损失为分解式（方向 + 幅度，无 plain L1）。")
    a("- C 的异质性评估依赖同 split 内多个 persona 的同情境配对。")

    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
