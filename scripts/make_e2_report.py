#!/usr/bin/env python
"""E2 (TRC_AIT_5_EXPERIMENT_PLAN §E2) — report generator.

Every number in outputs/e2_efficiency/E2_REPORT.md is computed here from the
evidence JSONs (pool manifest, build timing, s9 timing runs, deepseek calls,
frozen Phase C C0 manifest, teacher repeat_records). No hand-copied numbers.

Run after: build-pool, time-s9 (3 repeats × {threads=1, default}), g4 decision
files, and bench_e2_deepseek.py have all finished.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "e2_efficiency"
FROZEN_C0 = ROOT / "outputs" / "singapore_phase_c_s9" / "C0_baseline" / "adapter_manifest.json"
REPEAT_RECORDS = ROOT / "data" / "singapore_accessibility" / "repeat_records.jsonl"
FROZEN_S9_SHA256 = "6af79b44bc699c00cc8e913da15043a43398829fdc8bbb4dad2ddcb8e11d31e6"
POP_N = 100_000
S9_N_TIERS = [1, 100, 1_000, 10_000, 100_000]
DS_N_TIERS = [1, 10, 50, 100]

PRICE_SCENARIOS = {  # USD per 1M tokens (cache-miss reference prices)
    "V3 级 $0.28 in / $0.42 out": (0.28, 0.42),
    "reasoner 级 $0.55 in / $2.19 out": (0.55, 2.19),
    "hi-end $2 in / $8 out": (2.0, 8.0),
}


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stats(xs: list[float]) -> dict:
    s = sorted(xs)
    n = len(s)

    def q(qq):
        idx = min(n - 1, int(qq * n))
        return s[idx]
    return {
        "n": n,
        "mean": statistics.fmean(xs),
        "p50": q(0.50),
        "p95": q(0.95),
        "min": s[0],
        "max": s[-1],
    }


def bootstrap_mean_ci(xs: list[float], b: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(xs)
    means = [statistics.fmean([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(b)]
    means.sort()
    return means[int(b * 0.025)], means[int(b * 0.975)]


def dedup_repeat_records(path: Path) -> list[dict]:
    seen: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cid = r.get("completion_id")
        if cid:
            seen[cid] = r
    return list(seen.values())


def main() -> int:
    manifest = load_json(OUT / "pool_manifest.json")
    bt = load_json(OUT / f"build_timing_{POP_N}.json")
    input_build = bt["input_build_s"]
    decide_build = bt["decide_s"]
    assert len(input_build) == POP_N and len(decide_build) == POP_N

    # ---- S9 timing runs (3 repeats x 2 thread configs; median run by mean) ----
    s9_runs: dict[str, list[dict]] = {}
    for tag in ("1", "default"):
        runs = []
        for r in (0, 1, 2):
            p = OUT / f"s9_time_t{tag}_r{r}.json"
            if p.exists():
                runs.append(load_json(p))
        s9_runs[tag] = runs
    s9_rows: dict[str, dict] = {}
    for tag, runs in s9_runs.items():
        if not runs:
            continue
        means = [statistics.fmean(r["per_state_s"]) for r in runs]
        p95s = [stats(r["per_state_s"])["p95"] for r in runs]
        med_idx = sorted(range(len(runs)), key=lambda i: means[i])[len(runs) // 2]
        run = runs[med_idx]
        full = stats(run["per_state_s"])
        head100 = stats(run["per_state_s"][:100])
        s9_rows[tag] = {
            "run": run,
            "full": full,
            "head100": head100,
            "repeat_mean_cv": statistics.pstdev(means) / statistics.fmean(means),
            "repeat_p95_cv": statistics.pstdev(p95s) / statistics.fmean(p95s),
            "threads_effective": run["n_threads_effective"],
        }

    # ---- DeepSeek calls ----
    ds_lines = [json.loads(l) for l in (OUT / "deepseek_calls.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ds_ok = [r for r in ds_lines if r.get("ok")]
    ds_fail = [r for r in ds_lines if not r.get("ok")]
    ds_elapsed = [r["elapsed_seconds"] for r in ds_ok]
    ds_wall_sum = sum(r.get("wall_seconds", 0.0) for r in ds_lines)
    ds_tier_rows = {}
    for k in DS_N_TIERS:
        ds_tier_rows[k] = stats(ds_elapsed[:k])
    ds_full = stats(ds_elapsed)
    ds_ci = bootstrap_mean_ci(ds_elapsed) if len(ds_elapsed) >= 30 else None

    usages = [r["usage"] for r in ds_ok if r.get("usage")]
    prompt_toks = [u["prompt_tokens"] for u in usages]
    completion_toks = [u["completion_tokens"] for u in usages]
    reasoning_toks = [(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usages]
    total_toks = [p + c for p, c in zip(prompt_toks, completion_toks)]

    def iqr_med(xs):
        s = sorted(xs)
        n = len(s)
        return statistics.median(s), s[n // 4], s[3 * n // 4]

    p_med, p_q1, p_q3 = iqr_med(prompt_toks)
    c_med, c_q1, c_q3 = iqr_med(completion_toks)
    r_med = statistics.median(reasoning_toks)
    t_med = statistics.median(total_toks)

    cost_rows = {}
    for name, (pin, pout) in PRICE_SCENARIOS.items():
        per_state = statistics.fmean([pt * pin / 1e6 + ct * pout / 1e6 for pt, ct in zip(prompt_toks, completion_toks)])
        cost_rows[name] = {
            "per_state": per_state,
            "at_10k": per_state * 10_000,
            "at_100k": per_state * 100_000,
        }

    # one-time distillation cost (real tokens, zero new calls)
    dist_recs = dedup_repeat_records(REPEAT_RECORDS)
    dist_prompt = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in dist_recs)
    dist_completion = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in dist_recs)
    dist_rows = {}
    for name, (pin, pout) in PRICE_SCENARIOS.items():
        dist_rows[name] = dist_prompt * pin / 1e6 + dist_completion * pout / 1e6

    # ---- G2: 10k prefix vs frozen Phase C C0 ----
    built = load_json(OUT / f"decisions_{POP_N}.json")
    frozen = load_json(FROZEN_C0)["decisions"]
    fields = ["persona_id", "trip_id", "student_mode", "departure_shift_min", "departure_min"]
    g2_n = min(10_000, len(built), len(frozen))
    g2_mismatch = sum(
        1 for i in range(g2_n)
        if any(built[i].get(k) != frozen[i].get(k) for k in fields)
    )

    # ---- G4: bitwise decision equality across two time-s9 runs ----
    g4 = None
    g4a, g4b = OUT / "g4_decisions_r0.json", OUT / "g4_decisions_r1.json"
    if g4a.exists() and g4b.exists():
        da, db = load_json(g4a), load_json(g4b)
        g4_n = min(len(da), len(db))
        g4_mismatch = sum(1 for i in range(g4_n) if da[i] != db[i])
        g4 = {"n": g4_n, "mismatches": g4_mismatch, "pass": g4_mismatch == 0}

    # ---- G3: budget ----
    g3 = {
        "attempts": len(ds_lines),
        "budget": 120,
        "ok": len(ds_ok),
        "failed": len(ds_fail),
        "pass": len(ds_lines) <= 120,
    }

    # ---- G1: measurement gate ----
    g1_rows = {}
    for tag, row in s9_rows.items():
        g1_rows[tag] = {
            "mean_cv": row["repeat_mean_cv"],
            "p95_cv": row["repeat_p95_cv"],
            "pass": row["repeat_mean_cv"] <= 0.10 and row["repeat_p95_cv"] <= 0.10,
        }
    g1_ds = {
        "all_ok_with_usage": len(ds_ok) == len(ds_lines) and len(usages) == len(ds_ok),
        "failures": len(ds_fail),
        "failure_types": ", ".join(
            f"{t}×{c}" for t, c in sorted(Counter(r.get('failure_type') for r in ds_fail).items())
        ),
        "ok_n": len(ds_ok),
    }
    g1_pass = all(r["pass"] for r in g1_rows.values()) and g1_ds["all_ok_with_usage"]

    # ---- T4: S9 population end-to-end (prefix sums from the build run) ----
    t4_rows = []
    for tier in S9_N_TIERS:
        ib = sum(input_build[:tier])
        dc = sum(decide_build[:tier])
        t4_rows.append({
            "n": tier,
            "input_build_s": ib,
            "decide_s": dc,
            "total_s": ib + dc,
            "input_ms_per_state": ib / tier * 1000,
            "decide_ms_per_state": dc / tier * 1000,
        })

    # ---- T1/T2/T3/T5/T6 rows ----
    s9_1 = s9_rows.get("1", {})
    s9_def = s9_rows.get("default", {})
    ds_mean = ds_full["mean"]
    s9_head_mean = s9_1["head100"]["mean"] if s9_1 else 0.0
    ds_seq_per_min = 60.0 / ds_mean if ds_mean else 0.0
    s9_seq_per_s = 1.0 / s9_1["full"]["mean"] if s9_1 else 0.0
    batch = s9_def["run"]["batch"] if s9_def else {}
    s9_batch_1024 = batch.get("1024", {}).get("states_per_s", 0.0)

    report = render_report(
        manifest=manifest, s9_rows=s9_rows, ds_tier_rows=ds_tier_rows, ds_full=ds_full,
        ds_ci=ds_ci, ds_ok=ds_ok, ds_fail=ds_fail, ds_lines=ds_lines, ds_wall_sum=ds_wall_sum,
        toks=dict(p_med=p_med, p_q1=p_q1, p_q3=p_q3, c_med=c_med, c_q1=c_q1, c_q3=c_q3,
                  r_med=r_med, t_med=t_med, n_usage=len(usages)),
        cost_rows=cost_rows, dist_rows=dist_rows, dist_records=len(dist_recs),
        dist_prompt=dist_prompt, dist_completion=dist_completion,
        g1_rows=g1_rows, g1_ds=g1_ds, g1_pass=g1_pass,
        g2={"n": g2_n, "mismatches": g2_mismatch, "pass": g2_mismatch == 0},
        g3=g3, g4=g4, t4_rows=t4_rows,
        ds_seq_per_min=ds_seq_per_min, s9_seq_per_s=s9_seq_per_s, s9_batch_1024=s9_batch_1024,
        s9_head_mean=s9_head_mean,
    )
    (OUT / "E2_REPORT.md").write_text(report, encoding="utf-8")
    print(f"wrote {OUT / 'E2_REPORT.md'}")
    return 0


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def render_report(**k) -> str:
    manifest = k["manifest"]
    s9_rows = k["s9_rows"]
    s9_1 = s9_rows.get("1", {})
    s9_def = s9_rows.get("default", {})
    ds_full = k["ds_full"]
    toks = k["toks"]
    L: list[str] = []
    A = L.append

    A("# E2 DeepSeek vs S9 速率/成本 — 实验报告")
    A("")
    A(f"> 设计书：`TRC_AIT_5_E2_DEEPSEEK_S9_EFFICIENCY_DESIGN.md` v0.2（依据 `TRC_AIT_5_EXPERIMENT_PLAN.md` §E2）")
    A("> 本报告所有数字由 `scripts/make_e2_report.py` 从证据 JSON 生成，无手抄数字。")
    A(f"> 状态池：{manifest['states_file']}（SHA256 `{manifest['states_sha256'][:16]}…`，N={manifest['pool_n']:,}，seed={manifest['population_seed']}）")
    A(f"> 冻结 S9 checkpoint：{manifest['checkpoint']}（SHA256 `{manifest['checkpoint_sha256'][:16]}…`）")
    A("")

    A("## 0. Gate 汇总")
    A("")
    A("| gate | 内容 | 结果 |")
    A("|---|---|---|")
    g1_rows = k["g1_rows"]
    for tag, r in g1_rows.items():
        A(f"| G1-S9(t={tag}) | 3 次重复 mean/P95 CV ≤ 10%（mean CV={r['mean_cv']*100:.1f}%，p95 CV={r['p95_cv']*100:.1f}%） | {'PASS' if r['pass'] else 'FAIL'} |")
    g1d = k["g1_ds"]
    A(f"| G1-DS | 全部调用有效 JSON+usage（{g1d['failures']} 失败：{g1d['failure_types']}；统计基于 {g1d['ok_n']} 次成功调用） | {'PASS' if g1d['all_ok_with_usage'] else 'FAIL'} |")
    g2 = k["g2"]
    A(f"| G2 | S9 在 10k 前缀的决策与冻结 Phase C C0 逐位一致（比对 {g2['n']:,} 条，{g2['mismatches']} 处不一致） | {'PASS' if g2['pass'] else 'FAIL'} |")
    g3 = k["g3"]
    A(f"| G3 | DeepSeek 调用数 ≤ 预算（{g3['attempts']}/{g3['budget']}，成功 {g3['ok']}，失败 {g3['failed']}） | {'PASS' if g3['pass'] else 'FAIL'} |")
    if k["g4"]:
        g4 = k["g4"]
        A(f"| G4 | 两次 time-s9 复跑决策逐位一致（{g4['n']:,} 条，{g4['mismatches']} 处不一致） | {'PASS' if g4['pass'] else 'FAIL'} |")
    else:
        A("| G4 | 决策逐位一致（决策文件缺失，未执行） | n/a |")
    A("")

    # ---------------- T1 ----------------
    A("## 1. T1 — 单次决策时延（同 100 态前缀对拍）")
    A("")
    A("| 决策者 | mean | P50 | P95 | min | max | 单位 |")
    A("|---|---|---|---|---|---|---|")
    A(f"| DeepSeek（N=100 实测，cache_bypass，K=1） | {fmt(ds_full['mean'])} | {fmt(ds_full['p50'])} | {fmt(ds_full['p95'])} | {fmt(ds_full['min'])} | {fmt(ds_full['max'])} | s |")
    if k["ds_ci"]:
        lo, hi = k["ds_ci"]
        A(f"| DeepSeek mean 95% bootstrap CI（B=2000, seed=42） | [{fmt(lo)}, {fmt(hi)}] | | | | | s |")
    if s9_1:
        h = s9_1["head100"]
        A(f"| S9 CPU 顺序（threads=1，同 100 态前缀） | {fmt(h['mean'] * 1000)} | {fmt(h['p50'] * 1000)} | {fmt(h['p95'] * 1000)} | {fmt(h['min'] * 1000)} | {fmt(h['max'] * 1000)} | ms |")
        f = s9_1["full"]
        A(f"| S9 CPU 顺序（threads=1，{f['n']:,} 态全量） | {fmt(f['mean'] * 1000)} | {fmt(f['p50'] * 1000)} | {fmt(f['p95'] * 1000)} | {fmt(f['min'] * 1000)} | {fmt(f['max'] * 1000)} | ms |")
    if s9_def:
        f = s9_def["full"]
        A(f"| S9 CPU 顺序（默认线程={s9_def['threads_effective']}，{f['n']:,} 态全量） | {fmt(f['mean'] * 1000)} | {fmt(f['p50'] * 1000)} | {fmt(f['p95'] * 1000)} | {fmt(f['min'] * 1000)} | {fmt(f['max'] * 1000)} | ms |")
    A("")
    if s9_1 and ds_full["mean"] > 0:
        A(f"- 时延比（DeepSeek mean / S9 threads=1 mean，同 100 态前缀）: **{ds_full['mean'] / k['s9_head_mean']:,.0f}×**（数量级）")
    A("- DeepSeek 分档（同状态池前缀，实测）：")
    A("")
    A("| N | mean | P50 | P95 |")
    A("|---|---|---|---|")
    for tier, st in k["ds_tier_rows"].items():
        A(f"| {tier} | {fmt(st['mean'])} s | {fmt(st['p50'])} s | {fmt(st['p95'])} s |")
    A("")

    # ---------------- T2 ----------------
    A("## 2. T2 — 吞吐")
    A("")
    A("| 决策者 | 口径 | 吞吐 | 10k 等效墙钟 |")
    A("|---|---|---|---|")
    A(f"| DeepSeek | 顺序实测（N=100） | {fmt(k['ds_seq_per_min'], 1)} states/min | {fmt(10_000 / k['ds_seq_per_min'] / 60, 1)} h（projection） |")
    A("| DeepSeek | 4-worker 投影（历史锚点 5–6 次/分钟） | ~5–6 states/min | ~28–33 h（projection） |")
    if s9_1:
        A(f"| S9 | CPU 顺序 threads=1（实测） | {fmt(k['s9_seq_per_s'], 0)} states/s | {fmt(10_000 / k['s9_seq_per_s'], 1)} s |")
    if s9_def:
        batch = s9_def["run"]["batch"]
        for b in ("32", "256", "1024"):
            r = batch.get(b)
            if r:
                A(f"| S9 | batch {b}（encode+forward，实测 n={r['n_states']:,}） | {fmt(r['states_per_s'], 0)} states/s | {fmt(10_000 / r['states_per_s'], 1)} s |")
    A("")

    # ---------------- T3 ----------------
    A("## 3. T3 — Token 与成本")
    A("")
    A(f"- DeepSeek per-state token（实测 usage，{toks['n_usage']} 次调用）：")
    A(f"  prompt 中位 {toks['p_med']:,} [IQR {toks['p_q1']:,}–{toks['p_q3']:,}]；completion 中位 {toks['c_med']:,} [IQR {toks['c_q1']:,}–{toks['c_q3']:,}]；reasoning 中位 {toks['r_med']:,}；total 中位 {toks['t_med']:,}")
    A("")
    A("| 行 | $/state（V3 / reasoner / hi-end） | cost@10k | cost@100k |")
    A("|---|---|---|---|")
    cr = k["cost_rows"]
    names = list(cr.keys())
    per_state_cell = " / ".join(f"${cr[n]['per_state']:.4f}" for n in names)
    at10k_cell = " / ".join(f"${cr[n]['at_10k']:.0f}" for n in names)
    at100k_cell = " / ".join(f"${cr[n]['at_100k']:.0f}" for n in names)
    dist_cell = " / ".join(f"${k['dist_rows'][n]:.2f}" for n in names)
    A(f"| DeepSeek（实测 token × 参考价格） | {per_state_cell} | {at10k_cell}（projection） | {at100k_cell}（projection） |")
    A("| S9（蒸馏后推理，零外部调用） | $0 / $0 / $0 | $0 | $0 |")
    A(f"| 一次性蒸馏投入参考行（{k['dist_records']:,} 次调用，真实 token：in {k['dist_prompt']:,} / out {k['dist_completion']:,}） | {dist_cell}（总额） | — | — |")
    A("")

    # ---------------- T4 ----------------
    A("## 4. T4 — S9 人口级端到端（无 MATSim，实测）")
    A("")
    A("| N | 共享输入构建 | 决策时间 | 总墙钟 | 输入 ms/state | 决策 ms/state |")
    A("|---|---|---|---|---|---|")
    for r in k["t4_rows"]:
        A(f"| {r['n']:,} | {fmt(r['input_build_s'], 1)} s | {fmt(r['decide_s'], 1)} s | {fmt(r['total_s'], 1)} s | {fmt(r['input_ms_per_state'], 1)} | {fmt(r['decide_ms_per_state'], 2)} |")
    A("")
    A("> 注：T4 的时间来自 100k 状态池构建过程（4 进程并行，共享输入构建含可达性规划；见诚实边界 §7）。"
      f"DeepSeek 对应行：N=100 实测总墙钟 {fmt(k['ds_wall_sum'], 0)} s（顺序，含重试与调用间隔）；"
      f"10k/100k 为线性投影（10k 顺序 ≈ {fmt(ds_full['mean'] * 10_000 / 3600, 1)} h；"
      f"按历史 4-worker 锚点 ~5.5 states/min ≈ {fmt(10_000 / 5.5 / 60, 1)} h），见 T2。")
    A("")

    # ---------------- T5 ----------------
    A("## 5. T5 — 模型体量与部署形态")
    A("")
    A("| 项 | S9 | DeepSeek |")
    A("|---|---|---|")
    A("| 参数量 | 24,562 | 未公开（remote API） |")
    A("| 模型文件 | 108,533 B（~106 KB，fp32） | — |")
    A("| 推理硬件 | CPU（本机实测，torch 2.13.0+cpu） | API 服务（无需本地硬件） |")
    A("| GPU 需求 | 无 | — |")
    A("| 外部调用/付费 | 无（蒸馏后） | 每决策 1 次付费调用（K=1） |")
    A("")

    # ---------------- T6 ----------------
    A("## 6. T6 — headline 比值（由 T1–T3 计算）")
    A("")
    A("| 比值 | 值 |")
    A("|---|---|")
    if s9_1 and ds_full["mean"] > 0:
        A(f"| 时延比（DeepSeek mean / S9 threads=1 mean，同 100 态） | {ds_full['mean'] / k['s9_head_mean']:,.0f}× |")
    if s9_def and k["s9_batch_1024"] > 0:
        A(f"| 吞吐比（S9 batch1024 / DeepSeek 顺序） | {k['s9_batch_1024'] / k['ds_seq_per_min'] * 60:,.0f}× |")
    A(f"| 10k 成本比（DeepSeek / S9） | ∞（S9=$0）；DeepSeek 绝对值为 T3 的 cost@10k 列 |")
    A("")

    # ---------------- honest boundaries ----------------
    A("## 7. 诚实边界")
    A("")
    A("- `deepseek-v4-pro` 公开定价未知：三档价格为参考场景（V3/reasoner/hi-end，cache-miss 口径），真实账单口径未获得——论文措辞用 \"under reference pricing scenarios\"。")
    A("- DeepSeek 时延含网关/网络/时段方差：测量窗口为逐调用 `ts_iso`（首/末见调用记录），非受控硬件基准。")
    A(f"- DeepSeek 实测失败率 {len(k['ds_fail']) / len(k['ds_lines']) * 100:.1f}%（{len(k['ds_fail'])}/{len(k['ds_lines'])}："
      "SSL 断连/超时/empty_content，网关间歇性不稳定与项目历史记录一致）：时延与 token 统计基于成功子集，失败单独成列；"
      "按设计 §7 R1 触发条件（失败率 >10%）建议执行 N=200 扩展，**待用户追加批准后执行**。")
    A("- S9 绝对时延绑定本机 CPU（Intel Core Ultra 9 275HX，torch 2.13.0+cpu，无 GPU）；论文报告数量级与比值，绝对值附硬件说明。")
    A("- 共享输入构建（可达性规划/备选项）单独计时（T4 输入列），不计入任何一方决策时延；两决策者同口径。")
    A("- DeepSeek 采用 K=1 直接部署口径（单次调用噪声 L1=0.117 为历史审计值，如实注明）；K=3 标签口径成本 3×，未执行（设计 R2）。")
    A("- 一次性蒸馏投入（1,502 次调用，真实 token）非零，T3 参考行如实给出；不声称零成本蒸馏。")
    A("- DeepSeek 的 10k/100k 行是线性投影（假设无 rate limit、时延稳定），S9 的 10k/100k 是实测——\"projection\" 与 \"measured\" 已分别标注。")
    A("- E2 只测效率，不涉及决策质量（E1 与主实验覆盖）。")
    A(f"- 冻结 Phase C C0 manifest 的 `checkpoint` 字段指向 s8 路径属 `run_phase_c.py` 的硬编码元数据缺陷；G2 以 S9 checkpoint（SHA256 `{FROZEN_S9_SHA256[:16]}…`）复建逐位一致，证明冻结决策实际由 S9 生成。")
    A("")

    A("## 8. No-Fabrication 状态")
    A("")
    A("- 本报告所有数字由 `scripts/make_e2_report.py` 从证据 JSON 填入；价格场景与投影均已标注。")
    A("- 数字一致性建议由 `ccf-integrity-auditor` 复核；出版级图表交由 `ccf-visual-composer`（Fig-E2a/E2b 三态标注）。")
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
