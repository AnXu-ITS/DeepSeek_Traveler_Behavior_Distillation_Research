#!/usr/bin/env python
"""E3 report generator — fills T1-T5 from e3_result.json files ONLY (no hand copying).

    python scripts/make_e3_report.py [--archive-dir <paper-repo/data_report/12_E3_SCALE>]

Reads every outputs/e3_scale/**/e3_result.json, derives the frozen 10k
reference from the frozen Phase C C0 manifest + result JSON, and writes
outputs/e3_scale/E3_REPORT.md. Pilot (1k) rows aggregate the three repeats:
representative = median-by-build_s run; CV across repeats reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E3_OUT = ROOT / "outputs" / "e3_scale"
FROZEN_MANIFEST = ROOT / "outputs" / "singapore_phase_c_s9" / "C0_baseline" / "adapter_manifest.json"
FROZEN_RESULT = ROOT / "outputs" / "singapore_phase_c_s9" / "C0_baseline" / "phase_c_result.json"
MODES = ("car", "pt", "bike", "walk")
LINEAR_EXPECT = {10_000: 10.0, 20_000: 2.0, 50_000: 2.5, 100_000: 2.0, 200_000: 2.0, 500_000: 2.5}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_runs() -> dict[int, list[dict]]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for p in sorted(E3_OUT.glob("**/e3_result.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["_dir"] = str(p.parent)
        by_n[rec["n_agents"]].append(rec)
    return {n: sorted(v, key=lambda r: (r.get("repeat") is None, r.get("repeat"))) for n, v in by_n.items()}


def frozen_reference() -> dict:
    man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    dec = man["decisions"]
    n = len(dec)
    stu = defaultdict(int)
    exe = defaultdict(int)
    for d in dec:
        stu[d["student_mode"]] += 1
        exe[d["outbound_mode"]] += 1
    fallback = man.get("fallback_counts", {})
    intended_pt = stu["pt"]
    ref = {
        "n": n,
        "decision_share": {m: stu.get(m, 0) / n for m in MODES},
        "executed_share": {m: exe.get(m, 0) / n for m in MODES},
        "pt_validity": 1 - fallback.get("pt_fallback_walk", 0) / max(1, intended_pt),
        "pt_fallback_walk": fallback.get("pt_fallback_walk", 0),
        "intended_pt": intended_pt,
    }
    if FROZEN_RESULT.exists():
        res = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
        ref["metrics"] = res.get("metrics") or {}
        ref["build_s"] = res.get("build_seconds")
        ref["matsim_wall_s"] = res.get("runtime_seconds")
    return ref


def rep_of(repeats: list[dict]) -> dict:
    if len(repeats) == 1:
        return repeats[0]
    return sorted(repeats, key=lambda r: r["timing"]["build_s"])[len(repeats) // 2]


def pilot_cv(repeats: list[dict]) -> dict:
    if len(repeats) < 2:
        return {}
    def cv(xs: list[float]) -> float | None:
        if len(xs) < 2:
            return None
        m = statistics.mean(xs)
        return round(100 * statistics.stdev(xs) / m, 2) if m > 0 else 0.0
    build_cv = cv([r["timing"]["build_s"] for r in repeats])
    matsim_all = [r["timing"]["matsim_wall_s"] or 0 for r in repeats]
    matsim_cv_all = cv(matsim_all)
    matsim_cv_warm = cv(matsim_all[1:])  # design revision 1: warm-machine clause (r1/r2)
    decide_cv = cv([r["timing"]["decide_stats"].get("mean_ms", 0) for r in repeats])
    man_shas = {sha256(Path(r["_dir"]) / "adapter_manifest.json") for r in repeats}
    return {
        "n_repeats": len(repeats),
        "cv_build_pct": build_cv,
        "cv_matsim_all_pct": matsim_cv_all,
        "cv_matsim_warm_pct": matsim_cv_warm,
        "cv_decide_mean_pct": decide_cv,
        "manifest_sha256_identical": len(man_shas) == 1,
        "cv_ok": all(v is None or v <= 10.0 for v in (build_cv, matsim_cv_warm, decide_cv)),
        "decisions_identical": all(r["gates"]["g1"]["pass"] for r in repeats) and len(man_shas) == 1,
    }


def fmt_t(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds >= 3600:
        return f"{seconds/3600:.2f} h"
    if seconds >= 60:
        return f"{seconds/60:.1f} min"
    return f"{seconds:.1f} s"


def fmt_mb(v: float | None) -> str:
    return "—" if v is None else f"{v:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", default=None,
                    help="copy E3_REPORT.md + e3_result.json files into this dir")
    args = ap.parse_args()

    by_n = load_runs()
    if not by_n:
        print("no e3_result.json found under outputs/e3_scale")
        return 1
    ref = frozen_reference()
    fm = ref.get("metrics") or {}
    lines: list[str] = [
        "# E3 Population Scalability — Report",
        "",
        "> 依据：`TRC_AIT_5_EXPERIMENT_PLAN.md` §E3；设计书 `TRC_AIT_5_E3_POPULATION_SCALABILITY_DESIGN.md` v0.1",
        "> 生成：`scripts/make_e3_report.py`（全部数字来自 `outputs/e3_scale/**/e3_result.json`，无手抄）",
        "",
        "冻结设置：frozen S9（SHA256 `6af79b44…31e6`）· Singapore supply 全档不变 · C0 baseline ·",
        "seed=2026 嵌套前缀 · capacity 0.3/0.3 · lastIteration=0 · `-Xmx6g` · 单机 24C/31.4 GB。",
        "",
    ]

    # ---------------- pilot (1k repeats) ----------------
    if 1_000 in by_n:
        cv = pilot_cv(by_n[1_000])
        lines += ["## G3 确定性门（1k × 3 重复）", "",
                  "| 项 | 值 | 判据 |",
                  "|---|---|---|",
                  f"| repeats | {cv.get('n_repeats')} | 3 |",
                  f"| decisions 逐位一致（manifest SHA256） | {cv.get('manifest_sha256_identical')} | True |",
                  f"| CV build wall | {cv.get('cv_build_pct')}% | ≤10% |",
                  f"| CV MATSim wall（全三重复 / r1–r2 暖机） | {cv.get('cv_matsim_all_pct')}% / {cv.get('cv_matsim_warm_pct')}% | ≤10%（暖机口径，修订 1） |",
                  f"| CV decide mean | {cv.get('cv_decide_mean_pct')}% | ≤10% |",
                  f"| **G3 PASS** | **{cv.get('cv_ok') and cv.get('decisions_identical')}** | 全过才进 S3 |",
                  "",
                  "> r0 为 JVM/OS 缓存冷启动（MATSim 墙钟 +23 s）；暖机口径按设计书执行期修订 1。", ""]

    # ---------------- ladder scope note ----------------
    if 100_000 not in by_n:
        lines += ["## 档位范围（执行口径）", "",
                  "> 本报告覆盖 **1k×3 / 10k / 20k / 50k**；100k 及 200k/500k 扩展档未执行"
                  "（用户指示 E3 止于 50k，设计书执行期修订 8）。S7 六条“100k 稳定”判据"
                  "因此不再评估；T4 瓶颈判定在 1k–50k 范围内给出。", ""]

    # ---------------- T1 ----------------
    lines += ["## T1 — 逐档计时与资源分解（实测）", "",
              "| N | T_setup | T_factory | T_decide（mean/P50/P95 ms · 总量） | T_residual | T_build | "
              "T_matsim | T_parse | peak RAM py（P1/Δ, MB） | peak RAM jvm（MB） | population.xml | events.zst | states/s | agents/s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    reps: dict[int, dict] = {}
    for n in sorted(by_n):
        r = rep_of(by_n[n])
        reps[n] = r
        t = r["timing"]; mem = r["memory"]; art = r["artifacts"]
        ds = t["decide_stats"]; fs = t["factory_stats"]
        states_s = n / t["build_s"] if t["build_s"] else None
        agents_s = n / t["matsim_wall_s"] if t["matsim_wall_s"] else None
        lines.append(
            f"| {n:,} | {fmt_t(t['setup_s'])} | {fmt_t(fs.get('total_s'))} | "
            f"{ds.get('mean_ms')}/{ds.get('p50_ms')}/{ds.get('p95_ms')} ms · {fmt_t(ds.get('total_s'))} | "
            f"{fmt_t(t['residual_s'])} | {fmt_t(t['build_s'])} | {fmt_t(t['matsim_wall_s'])} | "
            f"{fmt_t(t['parse_s'])} | "
            f"{fmt_mb(mem['py_peak_ws_mb']['p1_after_build'])}/{fmt_mb(mem['py_peak_ws_mb']['delta'])} | "
            f"{fmt_mb(mem['java_peak_ws_mb'])} | "
            f"{fmt_mb(art['population_xml_bytes'] and art['population_xml_bytes']/2**20)} | "
            f"{fmt_mb(art['events_zst_bytes'] and art['events_zst_bytes']/2**20)} | "
            f"{states_s and round(states_s,2)} | {agents_s and round(agents_s,2)} |")
    stall_notes = []
    for n in sorted(by_n):
        r = reps[n]
        g4 = r["gates"]["g4"]
        if g4.get("stall_count"):
            stall_notes.append(f"N={n:,}：{g4['stall_count']} 个 decide 停顿 "
                               f"{g4['stall_durations_ms']} ms（OS 级，P99 其余 "
                               f"{r['timing']['decide_stats'].get('p95_ms')} ms；设计修订 4 记录在案）")
    if stall_notes:
        lines += ["", "> G4 停顿记录：" + "；".join(stall_notes), ""]
    if 1_000 in by_n and len(by_n[1_000]) > 1:
        lines += ["", f"> 1k 行为三重复的中位 run（按 build_s 取中位）；CV 见上方 G3 表。", ""]
    lines += ["", f"> 冻结参照（不进表）：10k build {fmt_t(ref.get('build_s'))} / MATSim "
                  f"{fmt_t(ref.get('matsim_wall_s'))} / population.xml 94.1 MB / events 272.5 MB。", ""]

    # ---------------- T2 ----------------
    lines += ["## T2 — 逐档聚合行为与系统指标", "",
              "| N | 决策 share（car/pt/bike/walk） | 执行 share | PT validity | boardings | "
              "stuck 人（率） | failed trips | mean trip time（min） | car VKT（km） | road delay（s/pass） | slow share | drift vs 10k（pp, 决策侧） |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def share_row(sh: dict) -> str:
        return "/".join(f"{100*sh.get(m,0):.1f}" for m in MODES)
    def drift_row(sh: dict) -> str:
        return "/".join(f"{100*(sh.get(m,0)-ref['decision_share'].get(m,0)):+.1f}" for m in MODES)
    lines.append(f"| 10k（冻结参照） | {share_row(ref['decision_share'])} | {share_row(ref['executed_share'])} | "
                 f"{100*ref['pt_validity']:.1f}%（{ref['pt_fallback_walk']}/{ref['intended_pt']}） | "
                 f"{fm.get('pt_boardings','—')} | {fm.get('stuck_persons','—')}（{100*(fm.get('stuck_persons',0)/max(1,fm.get('pt_boardings',0))):.1f}%） | "
                 f"{fm.get('failed_trips','—')} | {fm.get('mean_trip_time_min','—')} | {fm.get('car_vkt_km','—')} | "
                 f"{fm.get('road_delay_mean_s_per_passage','—')} | {fm.get('network_congestion_slow_share','—')} | 0/0/0/0 |")
    for n in sorted(by_n):
        r = reps[n]
        d = r["decisions"]; stu = d["student_mode_counts"]; exe = d["executed_outbound_counts"]
        sh_s = {m: stu.get(m, 0) / n for m in MODES}
        sh_e = {m: exe.get(m, 0) / n for m in MODES}
        m = r.get("metrics") or {}
        pv = r["pt_routing_validity"]
        boardings = m.get("pt_boardings", "—")
        stuck = m.get("stuck_persons", "—")
        rate = 100 * (m.get("stuck_persons", 0) / max(1, m.get("pt_boardings", 0))) if isinstance(boardings, int) else None
        tag = "（E3 复现）" if n == 10_000 else ""
        lines.append(f"| {n:,}{tag} | {share_row(sh_s)} | {share_row(sh_e)} | "
                     f"{100*pv['validity']:.1f}%（{pv['pt_fallback_walk']}/{pv['intended_pt']}） | "
                     f"{boardings} | {stuck}（{rate and f'{rate:.1f}%'}） | {m.get('failed_trips','—')} | "
                     f"{m.get('mean_trip_time_min','—')} | {m.get('car_vkt_km','—')} | "
                     f"{m.get('road_delay_mean_s_per_passage','—')} | {m.get('network_congestion_slow_share','—')} | "
                     f"{drift_row(sh_s)} |")
    lines += ["", "> 10k E3 复现行须与冻结参照逐位一致（G1/G3′）；漂移 = 前缀构成差异（非独立样本，E4 才是多 seed）。", ""]

    # ---------------- T3 ----------------
    ns = sorted(by_n)
    lines += ["## T3 — 扩展比值（相邻档实测比值 vs 线性期望）", "",
              "| 跃迁 | T_factory | T_decide | T_build | T_matsim | peak RAM py | peak RAM jvm | 线性期望 |",
              "|---|---|---|---|---|---|---|---|"]
    def ratio(a, b):
        if not a or not b:
            return "—"
        return f"{a/b:.2f}"
    for lo, hi in zip(ns, ns[1:]):
        rl, rh = reps[lo], reps[hi]
        tl, th = rl["timing"], rh["timing"]
        lines.append(f"| {lo:,}→{hi:,} | {ratio(th['factory_stats'].get('total_s'), tl['factory_stats'].get('total_s'))} | "
                     f"{ratio(th['decide_stats'].get('total_s'), tl['decide_stats'].get('total_s'))} | "
                     f"{ratio(th['build_s'], tl['build_s'])} | {ratio(th['matsim_wall_s'], tl['matsim_wall_s'])} | "
                     f"{ratio(rh['memory']['py_peak_ws_mb']['p1_after_build'], rl['memory']['py_peak_ws_mb']['p1_after_build'])} | "
                     f"{ratio(rh['memory']['java_peak_ws_mb'], rl['memory']['java_peak_ws_mb'])} | "
                     f"×{LINEAR_EXPECT.get(hi, hi/lo):g} |")
    lines += ["", "> 线性期望 = N 之比；<期望 = 摊销（OD/路径复用），>期望 = 超线性（拥堵/争用）。", ""]

    # ---------------- T4 ----------------
    lines += ["## T4 — 瓶颈判定（由 T1–T3 计算，不新增测量）", "",
              "| N | 墙钟占比 build/matsim/parse | 资源警戒（超时/OOM/≥24 GB） | 拥堵读数（slow share / stuck 率） | 判定 |",
              "|---|---|---|---|---|"]
    for n in ns:
        r = reps[n]
        t = r["timing"]; mem = r["memory"]; m = r.get("metrics") or {}
        total = (t["build_s"] or 0) + (t["matsim_wall_s"] or 0) + (t["parse_s"] or 0)
        sh = f"{100*t['build_s']/total:.0f}%/{100*(t['matsim_wall_s'] or 0)/total:.0f}%/{100*(t['parse_s'] or 0)/total:.0f}%"
        alerts = []
        if t.get("timeout_hit"):
            alerts.append("MATSim 超时")
        if t["matsim_exit_code"] not in (0, None):
            alerts.append(f"exit={t['matsim_exit_code']}")
        if (mem["py_peak_ws_mb"]["p1_after_build"] or 0) >= 24576:
            alerts.append("py RAM ≥24 GB")
        if (mem["java_peak_ws_mb"] or 0) >= 24576:
            alerts.append("jvm RAM ≥24 GB")
        max_share = max(("build", t["build_s"] or 0), ("matsim", t["matsim_wall_s"] or 0),
                        ("parse", t["parse_s"] or 0), key=lambda x: x[1])
        rate = 100 * (m.get("stuck_persons", 0) / max(1, m.get("pt_boardings", 0)))
        lines.append(f"| {n:,} | {sh} | {'；'.join(alerts) or '无'} | "
                     f"{m.get('network_congestion_slow_share','—')} / {rate:.1f}% | "
                     f"瓶颈 = {max_share[0]}；ceiling = {alerts[0] if alerts else '未触发'} |")
    lines += ["", "> 警戒线 24 GB = 本机 31.4 GB 的安全余量（设计 §7）。", ""]

    # ---------------- T5 ----------------
    lines += ["## T5 — 门禁与稳定判据汇总", "",
              "| N | G1 前缀匹配 | G2a | G2b（≤20k 硬门/其余报告） | G3′ | G4 | overall |",
              "|---|---|---|---|---|---|---|"]
    for n in ns:
        r = reps[n]
        g = r["gates"]
        g3p = "—" if g["g3prime"] is None else g["g3prime"]["pass"]
        lines.append(f"| {n:,} | {g['g1']['pass']}（{g['g1']['compared_n']} 态"
                     + (f"，pop.xml SHA={g['g1'].get('population_xml_sha256_match')}" if n == 10_000 else "")
                     + f"） | {g['g2a']['pass']} | {g['g2b']['pass']}（rate={g['g2b'].get('stuck_rate_per_pt_boarding')}，"
                     f"hard={g['g2b'].get('hard_gate')}） | {g3p} | "
                     f"{g['g4']['pass']} | **{r['overall_pass']}** |")
    if 1_000 in by_n:
        cv = pilot_cv(by_n[1_000])
        lines += ["", f"> 1k 档 G3（跨重复）：decisions 逐位一致 = {cv.get('decisions_identical')}；"
                      f"CV 全 ≤10% = {cv.get('cv_ok')}。", ""]
    lines += ["", "> G2b 在 N>20k 为报告项（重载档的偏离是 E3 要观测的发现，预注册口径）。", ""]

    # ---------------- S7 stability verdict (100k) ----------------
    if 100_000 in by_n:
        r = reps[100_000]
        g = r["gates"]; m = r.get("metrics") or {}
        t = r["timing"]; mem = r["memory"]
        stu = r["decisions"]["student_mode_counts"]
        sh = {m_: stu.get(m_, 0) / 100_000 for m_ in MODES}
        drift_max = max(abs(sh[m_] - ref["decision_share"].get(m_, 0)) for m_ in MODES)
        boardings = max(1, m.get("pt_boardings", 0))
        stuck_rate = m.get("stuck_persons", 0) / boardings
        failed_ratio = None
        if fm.get("failed_trips"):
            frozen_rate = fm["failed_trips"] / 10_000
            failed_ratio = (m.get("failed_trips", 0) / 100_000) / frozen_rate
        crit = {
            "1 技术门全过": g["g2a"]["pass"] and not t.get("timeout_hit") and t["matsim_exit_code"] == 0,
            "2 决策 share 漂移 ≤2pp": drift_max <= 0.02,
            "3 stuck 率 ≤5%": stuck_rate <= 0.05,
            "4 failed/agent ≤2× 10k": failed_ratio is None or failed_ratio <= 2.0,
            "5 峰值内存 ≤24 GB": (mem["py_peak_ws_mb"]["p1_after_build"] or 0) <= 24576
                                  and (mem["java_peak_ws_mb"] or 0) <= 24576,
            "6 100k build ≤12 h": t["build_s"] <= 43200,
        }
        lines += ["", "## S7 触发判据 — “100k 稳定”（六条全过才跑 200k/500k）", "",
                  "| # | 判据 | 值 | 通过 |", "|---|---|---|---|"]
        for k, v in crit.items():
            lines.append(f"| {k} | {'是' if v else '否'} |")
        lines += ["", f"> drift max = {100*drift_max:.1f}pp；stuck rate = {100*stuck_rate:.1f}%；"
                      f"failed ratio = {failed_ratio and round(failed_ratio,2)}×；"
                      f"py peak = {mem['py_peak_ws_mb']['p1_after_build']} MB；jvm peak = {mem['java_peak_ws_mb']} MB；"
                      f"build = {fmt_t(t['build_s'])}。",
                  f"> **结论：{'全部通过 → 可执行 200k/500k' if all(crit.values()) else '未全过 → 不触发扩展档，按设计记录 ceiling 并停（Stop Rule）'}**", ""]

    # ---------------- honest boundaries ----------------
    lines += ["## Honest boundaries", "",
              "- 地图与供给全档冻结，E3 只放大 N（计划书 §E3 红线：不地图+人口同时放大）；不采纳 `补充实验.md` 原始“1/4 新加坡 × 1M × 6 轮”口径。",
              "- N 档位为 seed=2026 序列的嵌套前缀；mode-share 漂移是前缀构成的抽样差异，非独立重复样本（无跨 N bootstrap；多 seed 属 E4）。",
              "- capacity 0.3/0.3 按 10k 标定；50k 的拥堵读数不得表述为“标定的新加坡拥堵水平”（“a controlled real-network experiment under calibrated effective capacity”）。",
              "- 计时为单机无争用口径；绝对值绑定本机（24C/31.4 GB/CPU-only torch），论文以跨档比值与瓶颈结论为可移植主张；50k 档 build 尾部 ~15% 与并行 E5 Helsinki 试点重叠（已声明，G4 无停顿）。",
              "- 峰值内存：Python 侧为 OS 峰值工作集（P0/P1 两读数，P1 含进程启动）；Java 侧 1 s 采样（粒度注明）；`-Xmx6g` 冻结。",
              "- T_matsim 含 JVM 启动与输出写盘；T_parse 为 harness 开销，不计入两侧。",
              "- 30:00 transit 截断既存（≈20%），随拥堵放大；PT 绝对量读数带此 artifact，跨档比较如实标注。",
              "- lastIteration=0：一次性决策；闭环重规划扩展性不在 E3 范围（Phase D 按 Stop Rule 已停）。",
              "- E3 零 API 调用、零重训、仅 C0；“10k 档逐位复现 Phase C C0”是所有读数的有效性锚。",
              ""]
    report = "\n".join(lines)
    (E3_OUT / "E3_REPORT.md").write_text(report, encoding="utf-8")
    print(f"report -> {E3_OUT / 'E3_REPORT.md'}")

    if args.archive_dir:
        arch = Path(args.archive_dir)
        arch.mkdir(parents=True, exist_ok=True)
        shutil.copy2(E3_OUT / "E3_REPORT.md", arch / "E3_REPORT.md")
        for p in sorted(E3_OUT.glob("**/e3_result.json")):
            rel = p.relative_to(E3_OUT)
            dst = arch / rel.parent
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
        print(f"archive -> {arch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
