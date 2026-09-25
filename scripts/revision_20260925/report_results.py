"""Audit experiment completion and assemble an evidence report, not paper prose."""
import collections,csv,json,math
from controlled import *
from offline_audits import rows,RAW
DEST=OUT.parent

def main():
    rs=read_json(OUT/'test_summary.json');summary=[];checks={};hist={};init={}
    for v,w in GRID:
        for seed in SEEDS:
            folder=OUT/name(v,w,seed);status=read_json(folder/'status.json');assert status['status']=='complete' and status['steps']==6360 and status['epochs']==120
            hist[v,w,seed]=[r['schedule_hash'] for r in rows(folder/'history.jsonl')];init[v,w,seed]=read_json(folder/'run.json')['initial_hash']
        for sel in ['static','response']:
            sub=[r for r in rs if r['variant']==v and r['weight']==w and r['selection']==sel]
            summary.append(dict(variant=v,weight=w,selection=sel,n_seeds=len(sub),**{metric:float(np.mean([r[metric] for r in sub])) for metric in ['response_gap','static_kl','departure_mae']},response_seed_sd=float(np.std([r['response_gap'] for r in sub],ddof=1))))
    checks['same_initialization_within_seed']=all(len({init[v,w,s] for v,w in GRID})==1 for s in SEEDS)
    checks['same_training_schedule_within_seed']=all(all(hist[v,w,s]==hist['soft_kl',0.,s] for v,w in GRID) for s in SEEDS)
    checks['all_27_complete']=len(rs)==54
    regression=[]
    for v in ['soft_kl','signed_l1','direction_magnitude']:
        w=0 if v=='soft_kl' else 1
        for s in SEEDS:
            a=torch.load(OUT/name(v,w,s)/'best_static.pt',map_location='cpu',weights_only=False);b=torch.load(ROOT/f'outputs/matched_response_v1/train/{v}_seed{s}/best.pt',map_location='cpu',weights_only=False)
            delta=max(float((a['model_state'][k]-b['model_state'][k]).abs().max()) for k in a['model_state']);regression.append(dict(variant=v,seed=s,same_epoch=a['epoch']==b['epoch'],max_parameter_difference=delta))
    checks['historical_weight1_exact_regression']=all(r['same_epoch'] and r['max_parameter_difference']==0 for r in regression)
    assert all(checks.values()),checks
    write_json(DEST/'training_verification.json',dict(checks=checks,regression=regression,driver_sha256=file_hash(ROOT/'scripts/revision_20260925/controlled.py')))
    write_json(DEST/'weight_selection_summary.json',summary)
    resources=[]
    for stage,root in [('controlled',OUT),('delay_family',DEST/'delay_family/train')]:
        paths=sorted(root.glob('*/status.json'));elapsed=[read_json(p)['elapsed_s'] for p in paths]
        resources.append(dict(stage=stage,runs=len(paths),sum_run_elapsed_s=sum(elapsed),min_run_elapsed_s=min(elapsed),max_run_elapsed_s=max(elapsed),source_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}))
    write_json(DEST/'training_resource_summary.json',dict(stages=resources,interpretation='Sum of recorded per-fit elapsed times, not end-to-end campaign wall time or CPU-seconds. No currency conversion without measured resource billing or a declared tariff.'))
    with (DEST/'weight_selection_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
    family=DEST/'delay_family';family_stats=[]
    if (family/'train/test_summary.json').exists():
        endpoints={e['id']:e for e in rows(family/'bundle/test_endpoints.jsonl')}
        relevant=lambda eid:endpoints[eid]['state']['context']['transit_delay_min']>0 or 'delay' in endpoints[eid].get('audit_group','').lower()
        for sel in ['static','response']:
            avgs={}
            for v,w in [('soft_kl',0),('signed_l1',1)]:
                ps=[{r['id']:r for r in rows(family/'train'/name(v,w,s)/f'test_{sel}/pairs.jsonl') if relevant(r['base']) or relevant(r['cf'])} for s in SEEDS]
                avgs[v]=[dict(r,response_gap=float(np.mean([p[r['id']]['response_gap'] for p in ps]))) for r in ps[0].values()]
                family_stats.append(dict(variant=v,selection=sel,n_pairs=len(avgs[v]),response_gap=float(np.mean([r['response_gap'] for r in avgs[v]]))))
            family_stats.append(dict(variant='signed_minus_soft',selection=sel,paired=paired_cluster_difference(avgs['signed_l1'],avgs['soft_kl'],'response_gap',10000,20260925)))
        write_json(family/'delay_only_test_summary.json',family_stats)
    api=read_json(DEST/'synthetic/pilot/status.json') if (DEST/'synthetic/pilot/status.json').exists() else {}
    formal=read_json(DEST/'synthetic/formal/status.json') if (DEST/'synthetic/formal/status.json').exists() else {}
    formal_audit=read_json(DEST/'synthetic/formal/acquisition_audit.json') if (DEST/'synthetic/formal/acquisition_audit.json').exists() else {}
    done=formal.get('complete',False) and formal_audit.get('all_checks_passed',False)
    physical=read_json(DEST/'physical_supply/comparisons.json') if (DEST/'physical_supply/comparisons.json').exists() else {}
    write_json(DEST/'stage_status.json',dict(A='complete',B=dict(local_family_training='complete' if family_stats else 'running',pilot=api,formal=formal,status='complete' if done else 'incomplete',blocker=None if done else 'See acquisition attempt ledger and status; do not infer missing responses'),C='deferred by author: no independent new human responses',D=dict(completed=physical.get('completed_runs',0),expected=40),manuscript_writing='not started; stop gate retained'))
    lines=['# 实验进展与写作前报告','', '这是实验记录，不是论文正文修订。原问卷、原始人类回答及历史实验文件保持原样。','',
      '## 已完成：零 API 训练和复核','', '27 次完整重训，54 个按不同验证标准选择的检查点；各次均为 120 epoch / 6,360 updates。九个历史同配置模型参数和选中 epoch 完全复现。',
      '另完成 4 个 MNL 正则化候选拟合；两个选择标准均选择 L2=0.0001。MNL choice 与 departure 的选择规则保持区分。','',
      '| 目标 | 权重 | 选择 | 平均响应误差 | 训练种子 SD | 静态宏 KL |','|---|---:|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["variant"]} | {r["weight"]:g} | {r["selection"]} | {r["response_gap"]:.6f} | {r["response_seed_sd"]:.6f} | {r["static_kl"]:.6f} |')
    lines+=['',f'逐次训练耗时之和：27 个受控拟合 {resources[0]["sum_run_elapsed_s"]:.2f} 秒，6 个家族留出拟合 {resources[1]["sum_run_elapsed_s"]:.2f} 秒。不是端到端 campaign wall-clock 或 CPU-seconds；没有资源账单/费率，不虚构美元训练成本。']
    lines+=['','这些均值不是独立总体样本：原测试集只有 6 个独立人设。配对人设区间见 `controlled/paired_comparisons.json`，权重网格的比较为敏感性分析，不能事后选最优测试权重当作预注册主结果。','',
      '组合留出审计：训练/验证未引用 fare×congestion 留出端点；测试有 24 个该组合端点。构成它的单独因素在训练中存在，因此结论只适用于组合留出，不是全部干预类别未见。','',
      '重复 Teacher 分析覆盖 373 个端点、350 对响应。K=3 使用 1 vs 1，K=5 使用 2 vs 2，保留共享端点和方法间相同分配。30 个分配相互相关，不能视作 30 次独立试验，也不能当作精确噪声上限。','',
      '旧执行实验逐阶段统计：先在每个人内平均配对分配种子，再整体重采样人，分别计算概率调整、分配、路径、上车对响应的增量。','',
      '## 干预类别留出','', '另将正 transit_delay 的端点及整个 delay 机制四元组从训练、验证和所有关联单元移除，重新拟合特征统计，再训练 soft-KL / signed-L1 × 3 种子，共 6 个模型。各自 120 epoch / 5,280 updates；同一留出实验内预算一致，不能与全数据训练混称同更新数。历史测试的 delay-only 结果是回顾性诊断；新模型参考样本评估另列。','',
      '## API 与新增数据','',f'当前 pilot 有效响应：{api.get("valid",0)}/432。请求实际模型 deepseek-v4.1-flash，提供方 OpenCode Go；不能记为 V4 Pro 或 DeepSeek 官方直供。',
      '作者设置 Global 后预检成功。保留最初两次 HTTP 400、全部连接故障和无效响应；仅补采缺失的有效 state-repeat 单元，传输恢复修订不改变模型、提示词、温度、输出上限或情境。',
      f'正式响应：{formal.get("valid",0)}/{formal.get("required","尚未锁定")}。正式人数按 pilot 中逐人主要差值 SD 和半宽 0.01 计算，30–120 人，pilot 与正式人设完全分离。计划公式不保证最终达到该精度。',
      '已收到 usage 的失败/无效回答也计入成本；未返回 usage 的调用记为未知，不能按零费用处理。按 token 估算的套餐额度价值、订阅费和实际账单分别报告。','',
      '独立人类修复验证按作者回复暂缓，现有问卷不能充当新增独立验证。','',
      '## 物理供给实验','',f'已核验 {physical.get("completed_runs",0)}/40 个模型×情境×配对分配种子组合。',
      '按线路与停站序列分组隔班删除，原 13,655 趟保留 6,924 趟。供给索引与 MATSim 时刻表共同修改。四臂为原行为/原供给、仅感知延迟/原供给、冻结原行为/受扰供给、按受扰 LOS 重算行为/受扰供给。',
      '时间沿用历史网络运行口径，不能据此声称校准过的步行、骑行效率改善；主要判断仍是初始人群分母下的概率、分配、路径、实际上车和完成情况。','',
      '## 写作停止点','', '尚未开始把新结果写入论文。已完成的阶段按证据单独列出；独立人类修复仍延期，不能把其他实验的完成改述为独立人类修复已经验证。']
    if family_stats:
        lines+=['','## 必须保留的类别留出负结果','']
        for sel in ['static','response']:
            ss={r['variant']:r for r in family_stats if r['selection']==sel}
            diff=ss['signed_minus_soft']['paired']
            lines.append(f'{sel} 选择下，58 个 delay 测试对：soft-KL={ss["soft_kl"]["response_gap"]:.6f}，signed-L1={ss["signed_l1"]["response_gap"]:.6f}；signed-minus-soft={diff["mean"]:.6f}，配对区间 {diff["ci"]}。')
        lines.append('signed-L1 在这项类别留出诊断中更差；不能将已覆盖任务的响应改善写成未见干预上的普遍提升。区间仍条件于仅 6 个历史测试人设。')
    if (DEST/'physical_supply/verified_contrasts.json').exists():
        lines+=['','## 四臂供给实验已核验结果','', '| 模型 | 比较 | 原始 PT 概率变化 pp | 实际上车变化 pp | 完成率变化 pp | 完成/时限时间变化 min |','|---|---|---:|---:|---:|---:|']
        for r in read_json(DEST/'physical_supply/verified_contrasts.json'):
            m=r['metrics'];lines.append(f'| {r["model"]} | {r["arm"]} minus {r["baseline"]} | {m["raw_pt_pp"]["mean"]:.3f} | {m["boarded_pt_pp"]["mean"]:.3f} | {m["completion_pp"]["mean"]:.3f} | {m["completion_or_horizon_minutes"]["mean"]:.3f} |')
        lines+=['','40 个组合包括 12 个逐人核验后复用的历史运行与 28 个新 MATSim 运行。每个组合的初始分母为 1,000 人，配对分配种子为 5 个；不能把 40,000 次执行记录视作 40,000 个独立人。区间、各阶段及行程时间见 `physical_supply`。',
            '本次各组出程均完成，因此完成/时限时间等于本次观测的出程时长。冻结行为、减少班次后上车比例未变，但模拟时长发生变化；这不等于供给没有影响，也不能将负时长差解释为现实减班收益。该时间结果沿用未校准的历史自由流网络执行口径。']
    for cohort,label in [('pilot','独立 pilot：仅用于样本量规划'),('formal','正式新合成人设实验')]:
        folder=DEST/'synthetic'/cohort
        if not (folder/'acquisition_audit.json').exists():continue
        a=read_json(folder/'analysis.json');audit=read_json(folder/'acquisition_audit.json');u=audit['usage']
        lines+=['',f'## {label}','',f'{a["n"]} 个人设；{audit["valid"]} 个有效回答。主比较为全数据、static 选模下 signed-L1 减 soft-KL 的逐人响应误差差值，负数有利于 signed-L1：均值 {a["mean"]:.6f}，95% 配对人设区间 [{a["ci95"][0]:.6f}, {a["ci95"][1]:.6f}]。',
            '每个人设内平均三个训练种子及 11 个干预对，再重采样人设；不能把调用次数、训练种子或共享 baseline 当独立样本。所有次要比较为描述性、点态区间。']
        if cohort=='pilot':lines.append(f'规划 SD={a["sd"]:.6f}，锁定正式人数 {a["formal_n_required"]}；pilot 不并入正式测试。')
        lines+=['',f'记录请求 {u["total_attempts"]} 次，其中 {u["usage_reported_attempts"]} 次有 usage，{u["usage_unknown_attempts"]} 次 usage 未知。已记录输入 {u["input_tokens"]:,}、输出 {u["output_tokens"]:,}、合计 {u["total_tokens"]:,} tokens；缓存输入 {u["cached_input_tokens"]:,} 已包含在输入中，reasoning 已包含在输出中。',
            f'按请求 UTC 峰/闲时及缓存量估算已记录部分：USD {u["request_time_rate_quota_estimate_usd"]:.4f} 套餐额度；全峰时、无缓存保守重估 USD {u["known_usage_peak_no_cache_usd"]:.4f}。不包括未知用量，不是实际账单；账户剩余额度未核实。Go 订阅 USD 10/月不应再次按每次请求叠加。',
            '', '| 训练数据 | 方法 | 选模 | 平均响应误差 | 95% 人设区间 |','|---|---|---|---:|---|']
        for r in read_json(folder/'descriptive_summary.json'):
            if r['metric']=='response_gap' and r['card']=='ALL':lines.append(f'| {r["family"]} | {r["variant"]} | {r["selection"]} | {r["mean"]:.6f} | [{r["ci_low"]:.6f}, {r["ci_high"]:.6f}] |')
        lines+=['','逐强度/干预、交互、不可行 PT 概率和 departure MAE 见同目录 `descriptive_summary.csv`；逐人配对差见 `paired_contrasts.csv`。输入覆盖的 192 项审计见 `intensity_support_audit.json`。不同数值强度不自动等于独立的未见类别。',
            '新参考来自 Go Flash，与历史 Pro 来源不同。因此这项结果是跨参考模型的一致性评估，不是原 Pro 压缩误差，也不是人类行为准确率。实验固定一个数值行程，不能据此声称跨城市或真实人群泛化。']
    if (DEST/'mnl_selection/extended_sensitivity_summary.json').exists():
        lines+=['','## 优化与 MNL 方向诊断','', '保留 135 条固定诊断批次的梯度记录，分别列出 KL、加权 departure、加权 response 范数和 KL-response 余弦；见 `audit/gradient_summary.json`。这些是沿训练轨迹的诊断，不能单凭梯度大小认定某个损失有因果优势。',
            '', '| 选择 | 扰动 | 可行端点数 | PT 概率平均变化 | PT 概率增加比例 |','|---|---|---:|---:|---:|']
        for r in read_json(DEST/'mnl_selection/extended_sensitivity_summary.json')['rows']:
            if r['subset']=='service_feasible':lines.append(f'| {r["selection"]} | {r["perturbation"]} | {r["n_endpoints"]} | {r["mean_pt_probability_change"]:.6f} | {r["fraction_pt_increases"]:.3f} |')
        lines+=['','接驳扰动同时增加 access 与 total PT time 各 5 分钟，避免只改分量而不改总量。该 MNL 使用无单调约束的共同特征，系数和概率方向不应解释为已识别的经济偏好或时间价值。']
    if (DEST/'synthetic/service_metadata_diagnostic.json').exists():
        sm=read_json(DEST/'synthetic/service_metadata_diagnostic.json')
        lines+=['','## API 服务元数据变化：事后诊断','',
            '请求模型、温度、提示词、输出上限均未改变，但部分正式响应的 reasoning_tokens 回报为 0，usage 字段结构和延迟也与此前不同。返回模型名保持一致且没有不可变后端指纹。这是返回特征的变化，不能据此断言后台换模型或实际上完全没有内部推理。',
            '', '| 批次 | 报告的 reasoning | 调用数 | 输出 tokens 中位数 |','|---|---|---:|---:|']
        for r in sm['call_summary']:lines.append(f'| {r["cohort"]} | {r["regime"]} | {r["n_calls"]} | {r["median_output_tokens"]:.0f} |')
        lines+=['','按每个人设的全部 36 次有效响应将其分为全 positive、全 zero 或 mixed，主差值的描述性分组如下。','', '| 批次 | 分组 | 人设数 | signed-minus-soft 主差值 |','|---|---|---:|---:|']
        for r in sm['primary_persona_groups']:lines.append(f'| {r["cohort"]} | {r["regime"]} | {r["n_personas"]} | {r["mean"]:.6f} |')
        lines+=['','该诊断在查看正式方法比较结果前、观察到服务元数据异常后登记，因此是事后基础设施敏感性分析。没有删样本、改主分析或改样本量。分组与采集时间、人设顺序混杂，不能推导推理模式/服务后端的因果效果。新结果只适用于本次实际服务混合条件，不能称为同一不可变后端下的严格复现；pilot 方差对正式精度的规划也需结合实际区间与服务差异解释。']
    if done:lines+=['','A、B 与 D 已完成；C 按作者要求延期。现在停在写作前，等待作者阅读本报告后决定论文论证范围。']
    (DEST/'EXPERIMENT_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(read_json(DEST/'stage_status.json')),flush=True)

if __name__=='__main__':main()
