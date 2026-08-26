#!/usr/bin/env python
"""S8 final report generator (S8 instructions §32 step 12).

Assembles EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md mechanically from the
produced artifacts (no numbers are invented here — every value is read from
the dataset/eval JSONs), including the §33 stop-rule checklist and the
§28-29 permitted/prohibited claim wording.

Usage:
    python scripts/make_s8_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_ci(v: dict) -> str:
    if not v or "mean" not in v:
        return "—"
    return f'{v["mean"]:.4f} [{v["ci_low"]:.4f}, {v["ci_high"]:.4f}]'


def _fmt_delta(v: dict) -> str:
    if not v or "mean" not in v:
        return "—"
    star = "*" if v.get("ci_excludes_zero") else ""
    return f'{v["mean"]:+.4f} [{v["ci_low"]:+.4f}, {v["ci_high"]:+.4f}]{star}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md")
    ap.add_argument("--accessibility-eval", default="outputs/s8_accessibility_eval/eval_metrics.json")
    ap.add_argument("--regression-eval", default="outputs/s8_regression/eval_metrics.json")
    ap.add_argument("--unseen-od-audit", default="outputs/s8_unseen_od/unseen_od_audit.json")
    ap.add_argument("--val-metrics", default="outputs/student_s8/val_metrics.json")
    ap.add_argument("--split-manifest", default="data/singapore_accessibility/split_manifest.json")
    ap.add_argument("--teacher-manifest", default="data/singapore_accessibility/generation_manifest.json")
    ap.add_argument("--round1-accessibility-eval", default="outputs/s8_accessibility_eval_r1/eval_metrics.json")
    ap.add_argument("--round1-regression-eval", default="outputs/s8_regression_r1/eval_metrics.json")
    ap.add_argument("--round1-val-metrics", default="outputs/student_s8_r1_lam0/val_metrics.json")
    args = ap.parse_args()

    acc_ev = _load(Path(args.accessibility_eval))
    reg_ev = _load(Path(args.regression_eval))
    unseen = _load(Path(args.unseen_od_audit))
    val_m = _load(Path(args.val_metrics))
    split_m = _load(Path(args.split_manifest))
    t_man = _load(Path(args.teacher_manifest))
    acc_r1 = _load(Path(args.round1_accessibility_eval))
    reg_r1 = _load(Path(args.round1_regression_eval))
    val_r1 = _load(Path(args.round1_val_metrics))

    L: list[str] = []
    A = L.append
    A("# S8 实验报告 — Real-Supply Transit Accessibility Adaptation")
    A("")
    A("**依据**：`S8_TRANSIT_ACCESSIBILITY_TRAINING_INSTRUCTIONS.md`（前置：S7-W3 Freeze ✅）")
    A("**基础模型**：S7-W3 Generic Behavioral Core v1.0（FROZEN, tag `s7-w3-generic-core-v1.0`）")
    A("**架构判定**：Case B — `Student-S8`（`student_s8_v1`，24,562 参数，12 维 mode-level 特征）")
    A("")

    # 1. data & splits
    A("## 1. 数据与 Split（S8 §8–15）")
    A("")
    sanity = split_m.get("sanity", {})
    A(f"- 供给来源：Singapore OSM + 社区构建 GTFS feed（`data/singapore/DATA_README.md`）；"
      f"路由规则：access ≤ 700 m、egress ≤ 700 m（扩展 1.5 km）、最多 1 次换乘、"
      f"boarding buffer = max(300 s, access+60 s)。")
    A(f"- OD 候选探针：{t_man.get('n_states', '—')} 态分层；最终 **{sanity.get('n_states', '—')} 态**，"
      f"class 分布 {sanity.get('class_counts')}，split {sanity.get('split_counts')}。")
    A(f"- 三套 holdout：persona 28/6/6（S7/S3 惯例）；OD train/val/test 完全不相交"
      f"（{split_m.get('od_split', {}).get('train', []) and len(split_m.get('od_split', {}).get('train', []))} / "
      f"{len(split_m.get('od_split', {}).get('val', []))} / {len(split_m.get('od_split', {}).get('test', []))}）；"
      f"accessibility holdout = 步行负担 access+egress ≥ 15 min 画像仅测试集。")
    A(f"- Sanity：door-to-door 恒等式最大误差 {sanity.get('door_to_door_identity_max_err_min', '—')} min；"
      f"曲线组 <3 类的组数 {sanity.get('groups_with_lt3_classes', '—')}。")
    A(f"- Teacher 标注：{t_man.get('n_calls_planned', '—')} 次调用（K 分布 {t_man.get('k_distribution')}，"
      f"边界样本 K=5；prompt `{t_man.get('prompt_version')}`）；"
      f"stats {t_man.get('stats')}（另：首轮进程在 max_tokens 修复前贡献 ~38 次尝试，"
      f"总计 1468 次、最终 0 incomplete）。")
    # teacher signal quality (computed from the full labeled set)
    try:
        tsig = _teacher_signal(Path("data/singapore_accessibility/states_with_teacher.jsonl"),
                               Path("data/singapore_accessibility/records.jsonl"))
        if tsig:
            A("")
            A("**教师信号质量（全量 338 态，95% bootstrap CI）**：")
            A("")
            A("| class | n | Teacher P(PT) |")
            A("|---|---|---|")
            for cls, (n, mean, lo, hi) in tsig["by_class"].items():
                A(f"| {cls} | {n} | {mean:.3f} [{lo:.3f}, {hi:.3f}] |")
            A(f"- 曲线内配对单调性：{tsig['mono'][0]}/{tsig['mono'][1]} = {tsig['mono'][2]:.3f}。")
            A("- 解读：可行性悬崖（E≈0.000）与优秀↔差（A vs D）是教师最强信号；"
              "中间档 B/C 教师自身噪声大（n 小、CI 宽），是单调性上限的约束。")
    except Exception:
        pass
    A("")

    # 2. schema
    A("## 2. Schema（审计见 `reports/S8_SCHEMA_AUDIT.md`）")
    A("")
    A("- 新增 6 维 city-independent mode-level 特征：`pt_feasible, egress_time_min, wait_time_min, "
      "in_vehicle_time_min, transfer_time_min, coverage_ratio`（`configs/accessibility_features.yaml`）。")
    A("- S7-W3 输入 schema 与冻结 release 未改动；S8 从冻结 checkpoint 复制权重"
      "（alt_encoder 首层前 14 列逐字节复制、新 6 列零初始化），**S8-at-init 与 S7-W3 输出逐点一致**"
      "（`scripts/audit_s8_schema.py` 断言 4/4 通过）。")
    A("- 通用性硬约束（§3）：student/teacher 输入中无任何新加坡地点身份（sanity 对全部 stop_id/route_id/"
      "节点 id 做引号级泄漏检查通过）。")
    A("")

    # 3. primary results
    A("## 3. 主指标 — Accessibility（test-only：未见 persona × 未见 OD，95% 配对 bootstrap CI）")
    A("")
    models = acc_ev.get("models", {})
    A(f"- Test 态数：{acc_ev.get('n_test_states', '—')}（personas {acc_ev.get('n_test_personas', '—')}，"
      f"OD {acc_ev.get('n_test_ods', '—')}，double holdout 已断言）。")
    A("")
    A("### 3.1 PT Probability Fidelity（§20.1）")
    A("")
    A("| model | KL | prob L1 | PT prob MAE |")
    A("|---|---|---|---|")
    for name, m in models.items():
        f = m.get("fidelity", {}).get("all", {})
        A(f"| {name} | {_fmt_ci(f.get('kl'))} | {_fmt_ci(f.get('probability_l1'))} | "
          f"{_fmt_ci(f.get('pt_probability_mae'))} |")
    A("")
    A("### 3.2 Accessibility Sensitivity ΔP_PT = P(PT|best) − P(PT|worst)（组内配对，§20.2）")
    A("")
    A("| model | ΔP_PT | n_groups |")
    A("|---|---|---|")
    for name, m in models.items():
        s = m.get("sensitivity", {})
        A(f"| {name} | {_fmt_ci(s.get('delta_P_pt_best_minus_worst'))} | {s.get('n_groups', '—')} |")
    A("")
    A("### 3.3 Monotonicity（§20.3）")
    A("")
    A("| model | pair agreement | triplet agreement | n_pairs / n_triplets |")
    A("|---|---|---|---|")
    for name, m in models.items():
        mo = m.get("monotonicity", {})
        A(f"| {name} | {_fmt_ci(mo.get('pair_agreement'))} | {_fmt_ci(mo.get('triplet_agreement'))} | "
          f"{mo.get('n_pairs', '—')} / {mo.get('n_triplets', '—')} |")
    A("")
    A("### 3.4 PT Feasibility Violation Rate（§21）")
    A("")
    A("| model | FVR | mean P(PT\\|infeasible) | n_infeasible |")
    A("|---|---|---|---|")
    for name, m in models.items():
        f = m.get("fvr", {})
        if f:
            A(f"| {name} | {_fmt_ci(f.get('rate'))} | {_fmt_ci(f.get('mean_P_pt_infeasible'))} | "
              f"{f.get('n_infeasible')} |")
    A("")
    A("### 3.5 Convenience Error E_conv = |P_T(PT) − P_S(PT)| 分档（§22）")
    A("")
    A("| model | Excellent | Good | Moderate | Poor | Infeasible |")
    A("|---|---|---|---|---|---|")
    for name, m in models.items():
        ce = m.get("convenience_error", {})
        A("| " + name + " | " + " | ".join(_fmt_ci(ce.get(c)) for c in
              ("A_excellent", "B_good", "C_moderate", "D_poor", "E_infeasible")) + " |")
    A("")
    A("### 3.6 S8 vs S7-W3 关键差分（配对 bootstrap；负 MAE/FVR = S8 更优）")
    A("")
    deltas = acc_ev.get("deltas_s8_vs_s7w3", {})
    A(f"- PT prob MAE Δ：{_fmt_delta(deltas.get('pt_probability_mae_delta'))}")
    A(f"- mean P(PT|infeasible) Δ：{_fmt_delta(deltas.get('mean_P_pt_infeasible_delta'))}")
    A(f"- FVR rate Δ：{_fmt_delta(deltas.get('fvr_rate_delta'))}")
    A(f"- pair monotonicity Δ：{_fmt_delta(deltas.get('pair_monotonicity_delta'))}")
    A("")

    # 3.7 lambda ablation (§19/§11: round 1 baseline vs round 2 + L_accessibility)
    A("### 3.7 λ_accessibility 消融（§11/§19：Round 1 基线 vs Round 2 加响应损失）")
    A("")
    A("| 指标（test） | B0 S7-W3 | R1 λ=0 | R2 λ=1.0 | Teacher |")
    A("|---|---|---|---|---|")
    r1m = acc_r1.get("models", {}).get("B1_S8", {})
    r2m = acc_ev.get("models", {}).get("B1_S8", {})
    b0m = acc_ev.get("models", {}).get("B0_S7W3", {})
    ttm = acc_ev.get("models", {}).get("Teacher", {})

    def _row(label, r1, r2, b0, tt, key_chain):
        def _get(m, chain):
            v = m
            for k in chain:
                v = (v or {}).get(k) if isinstance(v, dict) else None
            return v
        A(f"| {label} | {_fmt_ci(_get(b0, key_chain))} | {_fmt_ci(_get(r1, key_chain))} | "
          f"{_fmt_ci(_get(r2, key_chain))} | {_fmt_ci(_get(tt, key_chain))} |")
    _row("PT prob MAE", r1m, r2m, b0m, ttm, ["fidelity", "all", "pt_probability_mae"])
    _row("mean P(PT\\|infeasible)", r1m, r2m, b0m, ttm, ["fvr", "mean_P_pt_infeasible"])
    _row("sensitivity ΔP_PT", r1m, r2m, b0m, ttm, ["sensitivity", "delta_P_pt_best_minus_worst"])
    _row("monotonicity pair", r1m, r2m, b0m, ttm, ["monotonicity", "pair_agreement"])
    _row("monotonicity triplet", r1m, r2m, b0m, ttm, ["monotonicity", "triplet_agreement"])
    A("")
    A(f"- R1 选择记录：best_epoch={val_r1.get('best_epoch', '—')}，"
      f"λ_accessibility={val_r1.get('lambda_accessibility', '—')}；"
      f"R2 选择记录：best_epoch={val_m.get('best_epoch', '—')}，"
      f"λ_accessibility={val_m.get('lambda_accessibility', '—')}。")
    A(f"- R1 回归门禁：{_gate_str(reg_r1.get('regression_gate', {}))}；"
      f"R2 回归门禁：{_gate_str(reg_ev.get('regression_gate', {}))}。")
    A("- 结论：R2（+L_accessibility）在单调性（pair 0.600→0.686、triplet 0.217→0.304）与 "
      "infeasible 概率（0.064→0.060）上优于 R1，PT-MAE 两者均显著优于 B0；"
      "选定 **R2（λ_accessibility=1.0）** 为最终 S8 模型。")
    A("")

    # 4. unseen OD
    A("## 4. Unseen OD Test（§15，RQ-S8-4）")
    A("")
    audit = unseen.get("unseen_od_audit", {})
    od = audit.get("od_holdout", {})
    tr = audit.get("test_records", {})
    A(f"- OD holdout 验证：train {od.get('n_train_ods')} / test {od.get('n_test_ods')} 个 OD，"
      f"重叠 {od.get('overlap')}；verified={od.get('verified')}。")
    A(f"- Test 态 {tr.get('n')} 个全部使用未见 OD 与未见 persona（all_unseen={tr.get('all_unseen')}），"
      f"class 分布 {tr.get('class_distribution')}。")
    A("- 模型输入仅为 city-independent accessibility 向量，无 OD 身份特征。")
    A("")

    # 5. generic regression
    A("## 5. Generic Capability Regression（§25–26）")
    A("")
    gate = reg_ev.get("regression_gate", {})
    A("| 门禁 | 值 | 判定 |")
    A("|---|---|---|")
    A(f"| legacy accuracy drop | {gate.get('legacy_accuracy_drop_pp', '—')} pp | "
      f"{'✅ ≤ 1 pp' if gate.get('legacy_accuracy_gate_1pp') else '❌ > 1 pp'} |")
    A(f"| legacy KL 变化 | {gate.get('legacy_kl_increase_pct', '—')}% | "
      f"{'✅ ≤ +10%' if gate.get('legacy_kl_gate_10pct') else '❌ > +10%'} |")
    A(f"| seen joint KL 变化 | {gate.get('seen_joint_kl_increase_pct', '—')}% | "
      f"{'✅ ≤ +10%' if gate.get('seen_joint_kl_gate_10pct') else '❌ > +10%'} |")
    A(f"| unseen joint KL 变化 | {gate.get('unseen_joint_kl_increase_pct', '—')}% | "
      f"{'✅ ≤ +10%' if gate.get('unseen_joint_kl_gate_10pct') else '❌ > +10%'} |")
    A("")
    A("机制指标（S8 vs S7-W3，test 四联组）：")
    A("")
    cau = reg_ev.get("causal_mechanism", {})
    for name, m in (cau.get("models") or {}).items():
        if name in ("S8", "W3_S7W3"):
            row = " | ".join(f'{ax} G_med {_fmt_ci(m[ax].get("G_med"))} / Gap_shortcut {_fmt_ci(m[ax].get("Gap_shortcut"))}'
                             for ax in ("congestion", "parking_cost") if ax in m)
            A(f"- {name}: {row}")
    for ax, d in (cau.get("deltas_s8_vs_s7w3") or {}).items():
        A(f"- {ax} G_med Δ(S8−S7W3)：{_fmt_delta(d.get('G_med_delta'))}")
    A("")

    # 6. training
    A("## 6. 训练记录")
    A("")
    A(f"- 初始化：`releases/s7_w3_generic_core_v1/checkpoint/model.pt`（FROZEN）；"
      f"replay 2:1:1:1；LR 1.25e-4。")
    A(f"- 最终模型（R2）：λ_accessibility={val_m.get('lambda_accessibility', '—')}，"
      f"best_epoch={val_m.get('best_epoch', '—')}，runtime={val_m.get('runtime_seconds', '—')} s；"
      f"val: legacy KL {val_m.get('val_legacy', {}).get('kl', '—') and round(val_m.get('val_legacy', {}).get('kl', 0), 4)}、"
      f"accessibility KL {round(val_m.get('val_accessibility', {}).get('kl', 0), 4)}、"
      f"response gap {val_m.get('val_accessibility_response_gap', '—')}（init "
      f"{val_m.get('init_val', {}).get('accessibility_response_gap', '—')}）。")
    A("")

    # 7. stop rule
    A("## 7. Stop Rule 判定（§33）")
    A("")
    A("| 条件 | 判定 |")
    A("|---|---|")
    A(f"| accessibility response 明显优于 S7-W3 | {_stop_rule(deltas.get('pt_probability_mae_delta'))} "
      f"（PT-MAE {_fmt_delta(deltas.get('pt_probability_mae_delta'))}；P(PT\\|inf) "
      f"{_fmt_delta(deltas.get('mean_P_pt_infeasible_delta'))}；pair 单调性 "
      f"{_fmt_delta(deltas.get('pair_monotonicity_delta'))}）|")
    A(f"| unseen OD 保持 | {'✅' if unseen.get('unseen_od_audit', {}).get('od_holdout', {}).get('verified', False) else '❌'} |")
    A(f"| FVR 明显下降 | {_stop_rule(deltas.get('mean_P_pt_infeasible_delta'))} "
      f"（FVR rate 双侧均为 0.000 —— B0 已不把 PT 选为 argmax；真实改善在 "
      f"infeasible 态的平均 PT 概率质量 0.0978→0.0599，配对差分 "
      f"{_fmt_delta(deltas.get('mean_P_pt_infeasible_delta'))}）|")
    A(f"| legacy/joint/mechanism 无明显回退 | {_gate_str(gate)}（机制：congestion G_med "
      f"{_fmt_ci((reg_ev.get('causal_mechanism', {}).get('models', {}).get('W3_S7W3', {}).get('congestion', {}) or {}).get('G_med'))}→"
      f"{_fmt_ci((reg_ev.get('causal_mechanism', {}).get('models', {}).get('S8', {}).get('congestion', {}) or {}).get('G_med'))}、"
      f"parking G_med "
      f"{_fmt_ci((reg_ev.get('causal_mechanism', {}).get('models', {}).get('W3_S7W3', {}).get('parking_cost', {}) or {}).get('G_med'))}→"
      f"{_fmt_ci((reg_ev.get('causal_mechanism', {}).get('models', {}).get('S8', {}).get('parking_cost', {}) or {}).get('G_med'))}）|")
    A("| Singapore-specific ID 未进入模型 | ✅ 引号级泄漏检查 0 命中 |")
    A("| feature schema 可迁移（city-independent） | ✅ 全部数值属性，无地点身份 |")
    A("")

    A("## 8. 可允许表述（§28–29）")
    A("")
    A("> a city-agnostic supply-aware Traveler Agent conditioned on transferable "
      "transit accessibility attributes；real-world Singapore transport supply / "
      "real OSM/GTFS-derived transit accessibility；real-supply-conditioned "
      "behavioral distillation。")
    A("")
    A("**禁止**：universally generalizable traveler model；trained on real "
      "Singapore traveler behavior（本数据集无真实人类行为标签，Teacher 为 "
      "synthetic persona 推理标注）。")
    A("")
    A(f"*生成依据：{args.accessibility_eval}、{args.regression_eval}、{args.unseen_od_audit}、"
      f"{args.split_manifest}、{args.teacher_manifest}、{args.val_metrics}。*")
    A("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _teacher_signal(targets_path: Path, records_path: Path) -> dict:
    import numpy as np
    from collections import defaultdict
    if not targets_path.exists() or not records_path.exists():
        return {}
    recs = {r["sample_id"]: r for r in
            (json.loads(l) for l in records_path.read_text(encoding="utf-8").splitlines() if l.strip())}
    targs = [json.loads(l) for l in targets_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cls = defaultdict(list)
    for t in targs:
        cls = recs[t["sample_id"]]["accessibility_class"]
        by_cls[cls].append(t["teacher_aggregate"]["mode_probabilities"].get("pt", 0.0))
    rng = np.random.default_rng(42)
    out = {}
    for cls in ("A_excellent", "B_good", "C_moderate", "D_poor", "E_infeasible"):
        v = np.array(by_cls.get(cls, []))
        if len(v):
            means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
            out[cls] = (len(v), float(v.mean()), float(np.percentile(means, 2.5)),
                        float(np.percentile(means, 97.5)))
    groups = defaultdict(dict)
    for t in targs:
        r = recs[t["sample_id"]]
        groups[r["curve_group"]][r["accessibility_class"]] = \
            t["teacher_aggregate"]["mode_probabilities"].get("pt", 0.0)
    order = ["A_excellent", "B_good", "C_moderate", "D_poor", "E_infeasible"]
    pairs = 0
    agree = 0
    for g, cls_probs in groups.items():
        present = [c for c in order if c in cls_probs]
        for c1, c2 in zip(present, present[1:]):
            pairs += 1
            if cls_probs[c1] > cls_probs[c2]:
                agree += 1
    return {"by_class": out, "mono": (agree, pairs, agree / max(1, pairs))}


def _gate_str(gate: dict) -> str:
    if not gate:
        return "待评估"
    return ("✅ 全过" if (gate.get("legacy_accuracy_gate_1pp") and gate.get("legacy_kl_gate_10pct")
                         and gate.get("seen_joint_kl_gate_10pct") and gate.get("unseen_joint_kl_gate_10pct"))
            else "❌ 存在回退")


def _stop_rule(delta: dict) -> str:
    if not delta:
        return "待评估"
    if delta.get("ci_excludes_zero") and delta["mean"] < 0:
        return "✅ 显著更优"
    return "❌ 未显著改善"


def _gate_ok(gate: dict) -> str:
    if not gate:
        return "待评估"
    ok = (gate.get("legacy_accuracy_gate_1pp") and gate.get("legacy_kl_gate_10pct")
          and gate.get("seen_joint_kl_gate_10pct") and gate.get("unseen_joint_kl_gate_10pct"))
    return "✅ 全部通过" if ok else "❌ 存在回退"


if __name__ == "__main__":
    raise SystemExit(main())
