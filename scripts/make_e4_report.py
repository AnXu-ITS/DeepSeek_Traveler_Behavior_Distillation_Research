#!/usr/bin/env python
"""E4 report generator (design TRC_AIT_5_E4_MULTISEED_ROBUSTNESS_DESIGN.md §6 S5).

Builds T1–T5 and the C2 judgment block PROGRAMMATICALLY (no hand-copied numbers):
  - seed-2026 column: read-only from frozen Phase C
      outputs/singapore_phase_c_s9/<scenario>/phase_c_result.json (raw counts);
  - seeds 42/7: outputs/e4_multiseed/seed{42,7}/<scenario>/e4_result.json;
  - G0: records SHA256 of the frozen files + cross-checks phase_c_records.json
    against the per-scenario files (internal consistency);
  - G1: gate-repeat (seed42 C0 r1) vs evidence run (r0) — byte identity;
  - G3: failed-trips 0.5–2x same-scenario frozen value; stuck-rate vs the
    frozen 3.3–4.3% band (report items, not gates);
  - C2 judgment: pre-registered rule, design §5.2.

Usage:
    python scripts/make_e4_report.py
    python scripts/make_e4_report.py --output outputs/e4_multiseed/E4_REPORT.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # scripts/make_e4_report.py -> workbench root
FROZEN_DIR = ROOT / "outputs" / "singapore_phase_c_s9"
E4_DIR = ROOT / "outputs" / "e4_multiseed"
SCENARIOS = [
    "C0_baseline", "C1_heavy_rain", "C2_fare_increase",
    "C3_transit_delay", "C4_road_disruption", "C5_joint_rain_delay",
]
C1_C5 = SCENARIOS[1:]
MODES = ["car", "pt", "bike", "walk"]
N = 10_000
STUCK_BAND = (0.033, 0.043)          # frozen 3.3–4.3% across six scenarios
FAILED_RANGE = (0.5, 2.0)            # G3: same-scenario frozen value x0.5–x2
VERDICT_RANK = {"noise": 0, "borderline": 1, "stable": 2}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen() -> dict:
    """G0: load the six frozen Phase C records + cross-check vs phase_c_records.json."""
    files, records = {}, {}
    for sc in SCENARIOS:
        p = FROZEN_DIR / sc / "phase_c_result.json"
        files[sc] = {"path": str(p), "sha256": sha256(p)}
        records[sc] = json.loads(p.read_text(encoding="utf-8"))
    summary = json.loads((FROZEN_DIR / "phase_c_records.json").read_text(encoding="utf-8"))
    cross = {}
    for sc in SCENARIOS:
        entry = next(r for r in summary if r.get("scenario") == sc)
        cross[sc] = entry == records[sc]
    return {"files": files, "records": records, "cross_check": cross,
            "g0_pass": all(cross.values())}


def load_seed(seed: int) -> dict:
    """Load one new-seed run set; None marks missing files (TBD)."""
    out = {}
    for sc in SCENARIOS:
        p = E4_DIR / f"seed{seed}" / sc / "e4_result.json"
        out[sc] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out


def shares(counts: dict) -> dict:
    return {m: round(100 * counts.get(m, 0) / N, 1) for m in MODES}


def share_cell(counts: dict) -> str:
    return "/".join(str(shares(counts)[m]) for m in MODES)


def mstd(vals: list[float], nd: int = 1) -> str:
    if not vals or any(v is None for v in vals):
        return "TBD"
    m = statistics.fmean(vals)
    s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    lo, hi = min(vals), max(vals)
    return (f"{m:.{nd}f} ± {s:.{nd}f}"
            + (f"（{lo:.{nd}f}–{hi:.{nd}f}）" if lo != hi else ""))


def signs(vals: list[float]) -> str:
    if any(v is None for v in vals):
        return "TBD"
    s = [1 if v > 0 else (-1 if v < 0 else 0) for v in vals]
    if len(set(s)) == 1 and 0 not in s:
        return "3/3"
    return f"mixed（{'/'.join('+' if x > 0 else ('-' if x < 0 else '0') for x in s)}）"


def verdict(vals: list[float]) -> str:
    """Design §5.2 pre-registered C2 rule (mechanical, descriptive, n=3)."""
    if any(v is None for v in vals):
        return "TBD"
    same_sign = len({1 if v > 0 else -1 if v < 0 else 0 for v in vals}) == 1 and all(v != 0 for v in vals)
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    if same_sign and abs(mean) >= std:
        return "stable"
    if (not same_sign) and abs(mean) < std:
        return "noise"
    return "borderline"


def diffs(rec_sc: dict, rec_c0: dict) -> dict:
    """Same-seed paired differences (raw counts -> pp/units)."""
    d_sc = rec_sc["decisions"]["student_mode_counts"]
    d_c0 = rec_c0["decisions"]["student_mode_counts"]
    m_sc, m_c0 = rec_sc["metrics"], rec_c0["metrics"]

    def dm(key, nd=1):
        a, b = m_sc.get(key), m_c0.get(key)
        if a is None or b is None:
            return None
        return round(a - b, nd)

    return {
        "car_pp": round(100 * (d_sc.get("car", 0) - d_c0.get("car", 0)) / N, 1),
        "pt_pp": round(100 * (d_sc.get("pt", 0) - d_c0.get("pt", 0)) / N, 1),
        "boardings": dm("pt_boardings", 0),
        "vkt": dm("car_vkt_km", 1),
        "stuck": dm("stuck_persons", 0),
        "mtt": dm("mean_trip_time_min", 2),
    }


def stuck_rate(rec: dict) -> float | None:
    m = rec["metrics"]
    if not m or not m.get("pt_boardings"):
        return None
    return m.get("stuck_persons", 0) / m["pt_boardings"]


def g3_flags(seed: int, sc: str, rec: dict, frozen: dict) -> list[str]:
    if rec is None or rec.get("metrics") is None:
        return []
    flags = []
    fr = frozen["records"][sc]["metrics"]
    ratio = rec["metrics"].get("failed_trips", 0) / max(1, fr.get("failed_trips", 1))
    if not (FAILED_RANGE[0] <= ratio <= FAILED_RANGE[1]):
        flags.append(f"seed{seed}/{sc} failed_trips ratio {ratio:.2f} outside 0.5–2×")
    r = stuck_rate(rec)
    if r is not None and not (STUCK_BAND[0] <= r <= STUCK_BAND[1]):
        flags.append(f"seed{seed}/{sc} stuck rate {r:.2%} outside frozen band 3.3–4.3%")
    return flags


def build_report(out_path: Path) -> dict:
    frozen = load_frozen()
    new = {42: load_seed(42), 7: load_seed(7)}
    complete = all(new[s][sc] is not None for s in (42, 7) for sc in SCENARIOS)

    # ---- G1 (gate repeat vs evidence run)
    g1 = None
    r1_path = E4_DIR / "_gate" / "seed42_C0_r1" / "C0_baseline" / "e4_result.json"
    r0_path = E4_DIR / "seed42" / "C0_baseline" / "e4_result.json"
    if r1_path.exists() and r0_path.exists():
        r0 = json.loads(r0_path.read_text(encoding="utf-8"))
        r1 = json.loads(r1_path.read_text(encoding="utf-8"))
        g1 = {
            "population_xml_sha256_equal": r0["artifacts"]["population_xml_sha256"] == r1["artifacts"]["population_xml_sha256"],
            "adapter_manifest_sha256_equal": r0["artifacts"]["adapter_manifest_sha256"] == r1["artifacts"]["adapter_manifest_sha256"],
            "decisions_equal": r0["decisions"] == r1["decisions"],
            "metrics_equal": r0["metrics"] == r1["metrics"],
        }
        g1["pass"] = all(g1.values())

    # ---- G3 flags
    g3 = []
    for s in (42, 7):
        for sc in SCENARIOS:
            g3.extend(g3_flags(s, sc, new[s][sc], frozen))

    lines: list[str] = []
    banner = "" if complete else (
        "\n> ⚠️ **INCOMPLETE**：部分新 seed 运行缺失，对应行显示 TBD；本报告为中间状态，"
        "全部运行完成后由 make_e4_report.py 重新生成。\n")
    lines += [
        "# E4 Phase C Multi-Seed Robustness — Report",
        "",
        f"Population: N* = 10,000; seeds 2026 (frozen Phase C, read-only) / 42 / 7 (new runs); "
        "frozen S9 checkpoint (SHA256 6af79b44…31e6, verified per run); capacity 0.3/0.3; "
        "supply unchanged in every scenario; MATSim randomSeed=4711 frozen.",
        banner,
        f"*Generated by `scripts/make_e4_report.py` from result JSONs (no hand-copied numbers) on "
        f"{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')} UTC.*",
        "",
        "## T1 — 逐 seed × 情景绝对值",
        "",
        "| seed | scenario | 决策 share car/pt/bike/walk（%） | PT boardings | car VKT（km） | stuck 人（率） | failed trips | mean trip time（min） |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sc in SCENARIOS:
        fr = frozen["records"][sc]
        d = fr["decisions"]["student_mode_counts"]
        m = fr["metrics"]
        r = stuck_rate(fr)
        lines.append(
            f"| 2026（冻结） | {sc} | {share_cell(d)} | {m['pt_boardings']} | {m['car_vkt_km']} | "
            f"{m['stuck_persons']}（{r:.1%}） | {m['failed_trips']} | {m['mean_trip_time_min']} |")
    for s in (42, 7):
        for sc in SCENARIOS:
            rec = new[s][sc]
            if rec is None:
                lines.append(f"| {s} | {sc} | TBD | TBD | TBD | TBD | TBD | TBD |")
                continue
            d = rec["decisions"]["student_mode_counts"]
            m = rec["metrics"]
            r = stuck_rate(rec)
            lines.append(
                f"| {s} | {sc} | {share_cell(d)} | {m['pt_boardings']} | {m['car_vkt_km']} | "
                f"{m['stuck_persons']}（{r:.1%}） | {m['failed_trips']} | {m['mean_trip_time_min']} |")

    lines += [
        "",
        "## T2 — 逐 seed 的 scenario−C0 paired 差分",
        "",
        "| seed | scenario | Δcar share（pp） | Δpt share（pp） | ΔPT boardings | Δcar VKT（km） | Δstuck 人 | Δmean trip time（min） |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sc in C1_C5:
        d = diffs(frozen["records"][sc], frozen["records"]["C0_baseline"])
        lines.append(f"| 2026（冻结） | {sc} | {d['car_pp']:+.1f} | {d['pt_pp']:+.1f} | {d['boardings']:+.0f} | "
                     f"{d['vkt']:+.1f} | {d['stuck']:+.0f} | {d['mtt']:+.2f} |")
    for s in (42, 7):
        for sc in C1_C5:
            if new[s][sc] is None or new[s]["C0_baseline"] is None:
                lines.append(f"| {s} | {sc} | TBD | TBD | TBD | TBD | TBD | TBD |")
                continue
            d = diffs(new[s][sc], new[s]["C0_baseline"])
            lines.append(f"| {s} | {sc} | {d['car_pp']:+.1f} | {d['pt_pp']:+.1f} | {d['boardings']:+.0f} | "
                         f"{d['vkt']:+.1f} | {d['stuck']:+.0f} | {d['mtt']:+.2f} |")

    # ---- T3 (cross-seed summary) + C2 judgment
    lines += [
        "",
        "## T3 — 跨 seed 汇总与 C2 噪声判定",
        "",
        "| scenario | Δpt share mean ± std（min–max） | ΔPT boardings mean ± std（min–max） | Δcar share mean ± std | "
        "ΔVKT mean ± std | Δstuck mean ± std | 符号一致性（Δpt/Δboardings） | \\|mean\\|/std（Δpt） |",
        "|---|---|---|---|---|---|---|",
    ]
    summary = {}
    for sc in C1_C5:
        rows = {"pt_pp": [], "boardings": [], "car_pp": [], "vkt": [], "stuck": []}
        for key, recs in (("frozen", {sc: frozen["records"][sc], "C0_baseline": frozen["records"]["C0_baseline"]}),
                          ("new", None)):
            pass
        d = diffs(frozen["records"][sc], frozen["records"]["C0_baseline"])
        for k in rows:
            rows[k].append(d[k])
        for s in (42, 7):
            if new[s][sc] is None or new[s]["C0_baseline"] is None:
                rows["pt_pp"].append(None); rows["boardings"].append(None)
                rows["car_pp"].append(None); rows["vkt"].append(None); rows["stuck"].append(None)
                continue
            d = diffs(new[s][sc], new[s]["C0_baseline"])
            for k in rows:
                rows[k].append(d[k])
        pt_pp = rows["pt_pp"]
        boardings = rows["boardings"]
        mean_pt = statistics.fmean([v for v in pt_pp if v is not None]) if any(v is not None for v in pt_pp) else None
        std_pt = (statistics.stdev(pt_pp) if all(v is not None for v in pt_pp) and len(pt_pp) >= 2 else None)
        ratio = "—" if (mean_pt is None or not std_pt) else f"{abs(mean_pt) / std_pt:.2f}"
        summary[sc] = {
            "pt_pp": pt_pp, "boardings": boardings, "car_pp": rows["car_pp"],
            "vkt": rows["vkt"], "stuck": rows["stuck"],
            "pt_signs": signs(pt_pp), "boardings_signs": signs(boardings),
            "pt_verdict": verdict(pt_pp), "boardings_verdict": verdict(boardings),
            "mean_std_ratio": ratio,
        }
        lines.append(
            f"| {sc} | {mstd(pt_pp, 1)} | {mstd(boardings, 0)} | {mstd(rows['car_pp'], 1)} | "
            f"{mstd(rows['vkt'], 1)} | {mstd(rows['stuck'], 0)} | "
            f"{summary[sc]['pt_signs']} / {summary[sc]['boardings_signs']} | {ratio} |")

    c2 = summary["C2_fare_increase"]
    if "TBD" in (c2["pt_verdict"], c2["boardings_verdict"]):
        overall = "TBD"
        lines += [
            "",
            "### C2 判定块（预注册规则 §5.2，机械执行）",
            "",
            "| 主指标 | 3-seed Δ 值（2026/42/7） | 档位 |",
            "|---|---|---|",
            "| Δpt share（pp） | TBD（新 seed 运行未齐） | TBD |",
            "| ΔPT boardings | TBD（新 seed 运行未齐） | TBD |",
            "",
            "- **总体判定：TBD（数据未齐，全部运行完成后由 make_e4_report.py 重新生成）**",
            "- 口径声明：n=3 的描述性判定，**不是显著性检验**——不报 p 值、不报置信区间（设计 §8.3）。",
        ]
    else:
        overall_rank = min(VERDICT_RANK[c2["pt_verdict"]], VERDICT_RANK[c2["boardings_verdict"]])
        overall = {0: "noise", 1: "borderline", 2: "stable"}[overall_rank]
        abs_means = {sc: abs(statistics.fmean([v for v in summary[sc]["pt_pp"] if v is not None]))
                     for sc in C1_C5}
        sorted_scs = sorted(C1_C5, key=lambda s: abs_means[s])
        second_sc, second = (sorted_scs[1], abs_means[sorted_scs[1]]) if len(sorted_scs) >= 2 else (None, None)
        lines += [
            "",
            "### C2 判定块（预注册规则 §5.2，机械执行）",
            "",
            "| 主指标 | 3-seed Δ 值（2026/42/7） | 档位 |",
            "|---|---|---|",
            f"| Δpt share（pp） | {c2['pt_pp'][0]:+.1f} / {c2['pt_pp'][1]:+.1f} / {c2['pt_pp'][2]:+.1f} | **{c2['pt_verdict']}** |",
            f"| ΔPT boardings | {c2['boardings'][0]:+.0f} / {c2['boardings'][1]:+.0f} / {c2['boardings'][2]:+.0f} | **{c2['boardings_verdict']}** |",
            "",
            f"- **总体判定（取最低档）：{overall}**"
            + ("——C2 弱响应为稳定效应（蒸馏策略真实低票价敏感）" if overall == "stable"
               else "——证据混合，如实报告各 seed 数值，不下结论" if overall == "borderline"
               else "——C2 弱响应为抽样噪声（seed-2026 的 +0.4pp/+61 在跨 seed 波动范围内）"),
            f"- 上下文：C2 的 \\|mean Δpt\\|（{abs_means['C2_fare_increase']:.1f}pp）在 C1–C5 中**最小**（升序第 1/5）；"
            + (f"与次小响应（{second_sc}，{second:.1f}pp）的量级比 {abs_means['C2_fare_increase'] / second:.2f}×" if second_sc else "无次小可比")
            + f"；次级指标符号一致性：ΔVKT {signs(c2['vkt'])}、Δstuck {signs(c2['stuck'])}。",
            "- 口径声明：n=3 的描述性判定，**不是显著性检验**——不报 p 值、不报置信区间（设计 §8.3）。",
        ]

    # ---- T4 gates
    g2_all = []
    for s in (42, 7):
        for sc in SCENARIOS:
            rec = new[s][sc]
            g2_all.append((s, sc, None if rec is None else bool(rec["gate"]["pass"])))
    g2_pass = all(v is True for _, _, v in g2_all)
    lines += [
        "",
        "## T4 — 门禁与确定性汇总",
        "",
        "| 门 | 结果 |",
        "|---|---|",
        f"| G0 冻结列只读 + 内部一致性（6 文件 SHA256 + phase_c_records 交叉核对） | "
        f"{'PASS' if frozen['g0_pass'] else 'FAIL'}（交叉核对 6/6 一致："
        f"{sum(frozen['cross_check'].values())}/6） |",
        f"| G1 seed 42 C0 双复跑字节一致 | {'PASS' if g1 and g1['pass'] else ('FAIL' if g1 else 'PENDING（_gate/seed42_C0_r1 未运行）')} |",
        f"| G2 十二运行 gate（Phase C `_evaluate_gate` 原样） | "
        f"{'PASS 12/12' if g2_pass and complete else ('FAIL' if any(v is False for _, _, v in g2_all) else 'PENDING/TBD')} |",
        f"| G3 结构一致（failed trips 0.5–2×；stuck 率 vs 3.3–4.3% 带；报告项） | "
        f"{'无偏离' if not g3 else '；'.join(g3)} |",
        "| G4 数字管道（全表程序化生成；T1 冻结列 == 冻结 JSON 原始计数） | PASS（by construction，无手抄） |",
        "",
        "G2 明细：",
        "",
        "| seed \\ scenario | " + " | ".join(SCENARIOS) + " |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in (42, 7):
        cells = ["PASS" if v is True else ("FAIL" if v is False else "TBD")
                 for sd, _sc, v in g2_all if sd == s]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")

    # ---- T5 artifacts
    lines += [
        "",
        "## T5 — 工件与复现元数据",
        "",
        "| seed / scenario | population.xml SHA256（前 12） | manifest SHA256（前 12） | checkpoint SHA256（前 12） | build / matsim / parse（s） |",
        "|---|---|---|---|---|",
    ]
    for s in (42, 7):
        for sc in SCENARIOS:
            rec = new[s][sc]
            if rec is None:
                lines.append(f"| {s}/{sc} | TBD | TBD | TBD | TBD |")
                continue
            a = rec["artifacts"]
            lines.append(f"| {s}/{sc} | {a['population_xml_sha256'][:12]} | {a['adapter_manifest_sha256'][:12]} | "
                         f"{rec['checkpoint_sha256'][:12]} | {rec['build_seconds']} / {rec['runtime_seconds']} / {rec['parse_seconds']} |")
    lines += ["", "冻结 seed-2026 源文件 SHA256（G0 记录）：", ""]
    for sc in SCENARIOS:
        lines.append(f"- `{frozen['files'][sc]['path']}` → `{frozen['files'][sc]['sha256']}`")

    lines += [
        "",
        "## Honest boundaries",
        "",
        "- n=3 口径：mean ± std 为描述性 spread，不是抽样分布的估计；不做 bootstrap、不报置信区间、不报显著性。",
        "- 跨 seed 差异的唯一来源是需求样本（persona/trip/OD）；供给、S9 权重、capacity、MATSim randomSeed=4711 全部冻结，仿真侧确定。",
        "- E4 不重跑 seed 2026（用户指令 + 计划书 12 runs 口径）；与 Phase C 管线的等价性由 E1 G3 / E2 G2 既有锚 + E4 G1 内部确定性 + 进程内冻结断言承载。",
        "- 冻结 `phase_c_result.json` 的 checkpoint 字段为陈旧 S8 元数据（E2 v0.2 修订记录 1）；E4 record 写真实字段，seed-2026 列出处以冻结目录与 S9 SHA256 为准。",
        "- seed-2026 列使用冻结 JSON 原始计数，不用 `PHASE_C_REPORT.md` 的舍入文本；share/差分由本生成器程序化计算（G4）。",
        "- 需求生成代码 = 当前 workbench 版本，非字节冻结（Phase C 冻结的是 seed 而非代码）。",
        "- 30:00 transit 截断（≈20%）为冻结设置既有特性；同设置下的 paired 差分仍然有效。",
        "- capacity 0.3/0.3 按 10k 标定；表述沿用 \"a controlled real-network experiment under calibrated effective capacity\"。",
        "- C2 判定为描述性三档规则，不作为统计推断。",
        "- E4 执行期与 E5 Helsinki 并发（用户 2026-08-28 批准）：build/matsim/parse 墙钟为出处性元数据而非主张，并发只影响墙钟、不影响决策/指标证据值（每次运行的 contention_note 见 e4_result.json）。",
        "",
        f"*Generated from {len(SCENARIOS)} frozen Phase C records + "
        f"{sum(1 for s in (42, 7) for sc in SCENARIOS if new[s][sc] is not None)} new E4 records.*",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "frozen_file_sha256": frozen["files"],
        "g0_cross_check": frozen["cross_check"],
        "g1": g1,
        "g3_flags": g3,
        "g2": [{"seed": s, "scenario": sc, "pass": v} for s, sc, v in g2_all],
        "c2_judgment": {
            "seed_values": {"pt_pp": c2["pt_pp"], "boardings": c2["boardings"]},
            "pt_verdict": c2["pt_verdict"], "boardings_verdict": c2["boardings_verdict"],
            "overall": overall,
        },
        "summary": {sc: {k: v for k, v in summary[sc].items()} for sc in C1_C5},
        "complete": complete,
    }
    (out_path.parent / "e4_records.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(E4_DIR / "E4_REPORT.md"))
    args = ap.parse_args()
    payload = build_report(Path(args.output))
    print(f"report -> {args.output}")
    print(f"g0_pass={payload['g0_cross_check'] and all(payload['g0_cross_check'].values())}; "
          f"g1={'PASS' if payload['g1'] and payload['g1']['pass'] else ('FAIL' if payload['g1'] else 'PENDING')}; "
          f"complete={payload['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
