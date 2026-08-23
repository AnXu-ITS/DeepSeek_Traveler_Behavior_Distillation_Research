# Archive — legacy / superseded artifacts

## legacy_20260820（v0.2 接手时归档）

归档于 2026-08-20：历史轮次指令日志、mock 数据/产物、v0.1 单次调用数据集、旧 smoke。

## legacy_20260821（v0.1 时代工具 / 开发期临时产物）

| 原路径 | 原因 |
|---|---|
| `scripts/generate_teacher_dataset.py` | v0.1 单次调用生成器，被 `generate_aggregated_teacher_dataset.py`（K=3 聚合）取代。 |
| `scripts/validate_dataset.py` | v0.1 校验，被 `validate_aggregated_dataset.py` 取代。 |
| `scripts/probe_gateway.py` | 内部网关探活；已切换官方直连，网关弃用。 |
| `scripts/probe_api_cost.py`、`scripts/estimate_dataset_cost.py` | 一次性成本探测工具（网关时代）。 |
| `scripts/run_cache_busted_repeatability.py`、`run_cache_diagnostic.py`、`run_teacher_audit.py`、`run_weather_audit.py`、`analyze_teacher_audit.py` | v0.1 教师审计工具链；结论已固化进当前管线（K=3 聚合）。 |
| `data/matsim_equil_test/` | equil 示例本地拷贝（环境 sanity），可重建。 |
| `data/matsim_smoke/` | Phase 8 门禁 20-persona 冒烟场景，可经 `build_matsim_scenario.py` 重建。 |
| `outputs/teacher_analysis_v0_2.json` | 被 `teacher_analysis_v0_3.json` 取代。 |
| `outputs/v0_3_summary.md` | 被 `EXPERIMENT_REPORT_v0_3.md` 取代。 |
| `outputs/comparison_A_vs_B.md`、`comparison_A_vs_C.md` | v0.2 时代对比，数字已并入 v0.2 报告。 |
| `outputs/api_cost_probe.json` | 网关时代成本探测。 |
| `COST_AND_TIME_ESTIMATE.md` | 网关时代成本预估；当前扩训方案见 `PROGRESS.md` 下一步。 |

## Kept active (do not archive)

- 数据证据：`data/student_v0_2_a`、`data/student_v0_3`、`data/population_experiment`、`data/phase10_loop`
- 报告：`outputs/EXPERIMENT_REPORT_v0_2/v0_3`、`PHASE9/PHASE10` 报告、`comparison_v0_3_*`、`teacher_analysis_v0_3.json`、`teacher_audit_v0_1/`
- 训练产物：`outputs/student_v0_2_{a,b,c}`、`outputs/student_v0_3_{a,b,c}`（含 checkpoints）
- `DeepSeek_Traveler_Behavior_Distillation_Research_Blueprint.md`、`Task_Phase.txt`、`PROGRESS.md`、`README.md`
- `configs/`、`src/`、`scripts/`（活跃脚本）、`tests/`、`tools/`（MATSim release + Java 启动器）
