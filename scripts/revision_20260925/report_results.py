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
    physical=read_json(DEST/'physical_supply/comparisons.json') if (DEST/'physical_supply/comparisons.json').exists() else {}
    write_json(DEST/'stage_status.json',dict(A='complete',B=dict(local_family_training='complete' if family_stats else 'running',pilot=api,formal='not started' if not (DEST/'synthetic/formal/analysis.json').exists() else 'complete',blocker='Provider requires Global privacy region; pending author setting change' if not api.get('complete') else None),C='deferred by author: no independent new human responses',D=dict(completed=physical.get('completed_runs',0),expected=40),manuscript_writing='not started; stop gate retained'))
    lines=['# 实验进展与写作前报告','', '这是实验记录，不是论文正文修订。原问卷、原始人类回答及历史实验文件保持原样。','',
      '## 已完成：零 API 训练和复核','', '27 次完整重训，54 个按不同验证标准选择的检查点；各次均为 120 epoch / 6,360 updates。九个历史同配置模型参数和选中 epoch 完全复现。',
      '另完成 4 个 MNL 正则化候选拟合；两个选择标准均选择 L2=0.0001。MNL choice 与 departure 的选择规则保持区分。','',
      '| 目标 | 权重 | 选择 | 平均响应误差 | 训练种子 SD | 静态宏 KL |','|---|---:|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["variant"]} | {r["weight"]:g} | {r["selection"]} | {r["response_gap"]:.6f} | {r["response_seed_sd"]:.6f} | {r["static_kl"]:.6f} |')
    lines+=['','这些均值不是独立总体样本：原测试集只有 6 个独立人设。配对人设区间见 `controlled/paired_comparisons.json`，权重网格的比较为敏感性分析，不能事后选最优测试权重当作预注册主结果。','',
      '组合留出审计：训练/验证未引用 fare×congestion 留出端点；测试有 24 个该组合端点。构成它的单独因素在训练中存在，因此结论只适用于组合留出，不是全部干预类别未见。','',
      '重复 Teacher 分析覆盖 373 个端点、350 对响应。K=3 使用 1 vs 1，K=5 使用 2 vs 2，保留共享端点和方法间相同分配。30 个分配相互相关，不能视作 30 次独立试验，也不能当作精确噪声上限。','',
      '旧执行实验逐阶段统计：先在每个人内平均配对分配种子，再整体重采样人，分别计算概率调整、分配、路径、上车对响应的增量。','',
      '## 干预类别留出','', '另将正 transit_delay 的端点及整个 delay 机制四元组从训练、验证和所有关联单元移除，重新拟合特征统计，再训练 soft-KL / signed-L1 × 3 种子，共 6 个模型。各自 120 epoch / 5,280 updates；同一留出实验内预算一致，不能与全数据训练混称同更新数。历史测试的 delay-only 结果是回顾性诊断；新模型参考样本评估另列。','',
      '## API 与新增数据','',f'当前 pilot 有效响应：{api.get("valid",0)}/432。请求实际模型 deepseek-v4.1-flash，提供方 OpenCode Go；不能记为 V4 Pro 或 DeepSeek 官方直供。',
      '预检遇到 HTTP 400：Go 要求工作区 Privacy 使用 Global regions。已经停止自动重试，等待作者修改或选择暂缓。失败请求未报告 token 用量；这不能替代账户账单。',
      '计划 pilot 约 1.84M tokens（包含 10% 余量）；按旧平均用量与 Go Flash 价格约 0.7–1.5 USD 的套餐额度。套餐订阅费与按 token 计算的额度价值分开报告。正式人数按 pilot 中逐人主要差值 SD 和半宽 0.01 计算，30–120 人，pilot 与正式人设完全分离。','',
      '独立人类修复验证按作者回复暂缓，现有问卷不能充当新增独立验证。','',
      '## 物理供给实验','',f'已核验 {physical.get("completed_runs",0)}/40 个模型×情境×配对分配种子组合。',
      '按线路与停站序列分组隔班删除，原 13,655 趟保留 6,924 趟。供给索引与 MATSim 时刻表共同修改。四臂为原行为/原供给、仅感知延迟/原供给、冻结原行为/受扰供给、按受扰 LOS 重算行为/受扰供给。',
      '时间沿用历史网络运行口径，不能据此声称校准过的步行、骑行效率改善；主要判断仍是初始人群分母下的概率、分配、路径、实际上车和完成情况。','',
      '## 写作停止点','', '尚未开始把新结果写入论文。待其余可执行实验与完整性核验结束后，再向作者汇报；新 API 的阻塞单独列明，不能声称全部实验已经完成。']
    if family_stats:
        lines+=['','## 必须保留的类别留出负结果','']
        for sel in ['static','response']:
            ss={r['variant']:r for r in family_stats if r['selection']==sel}
            diff=ss['signed_minus_soft']['paired']
            lines.append(f'{sel} 选择下，58 个 delay 测试对：soft-KL={ss["soft_kl"]["response_gap"]:.6f}，signed-L1={ss["signed_l1"]["response_gap"]:.6f}；signed-minus-soft={diff["mean"]:.6f}，配对区间 {diff["ci"]}。')
        lines.append('signed-L1 在这项类别留出诊断中更差；不能将已覆盖任务的响应改善写成未见干预上的普遍提升。区间仍条件于仅 6 个历史测试人设。')
    if (DEST/'physical_supply/verified_contrasts.json').exists():
        lines+=['','## 四臂供给实验已核验结果','', '| 模型 | 比较 | 原始 PT 概率变化 pp | 实际上车变化 pp | 完成率变化 pp |','|---|---|---:|---:|---:|']
        for r in read_json(DEST/'physical_supply/verified_contrasts.json'):
            m=r['metrics'];lines.append(f'| {r["model"]} | {r["arm"]} minus {r["baseline"]} | {m["raw_pt_pp"]["mean"]:.3f} | {m["boarded_pt_pp"]["mean"]:.3f} | {m["completion_pp"]["mean"]:.3f} |')
        lines+=['','40 个组合包括 12 个逐人核验后复用的历史运行与 28 个新 MATSim 运行。每个组合的初始分母为 1,000 人，配对分配种子为 5 个；不能把 40,000 次执行记录视作 40,000 个独立人。区间、各阶段及行程时间见 `physical_supply`。']
    (DEST/'EXPERIMENT_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(read_json(DEST/'stage_status.json')),flush=True)

if __name__=='__main__':main()
