"""Same-respondent attribution with common cluster resamples; no model fitting."""
from pathlib import Path
from collections import defaultdict
import json,csv,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/revision_20260921/teacher'
SUR=ROOT/'outputs/revision_20260921/survey'
MODES=['car','pt','bike','walk']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def csvout(p,x):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)

def interval(x,idx,family=1):
    boot=x[idx].mean(1)
    return {'mean':float(x.mean()),'low':float(np.quantile(boot,.025)),
            'high':float(np.quantile(boot,.975)),
            'family_low':float(np.quantile(boot,.05/(2*family))),
            'family_high':float(np.quantile(boot,1-.05/(2*family)))}

def metrics(p,y):
    valid=y>=0; pp=p[valid]; yy=y[valid]; Y=np.eye(4)[yy]
    pred=pp.argmax(1);cm=np.zeros((4,4),int);np.add.at(cm,(yy,pred),1)
    rec=np.divide(cm.diagonal(),cm.sum(1),out=np.full(4,np.nan),where=cm.sum(1)>0)
    return dict(n_choices=int(valid.sum()),accuracy=float((pred==yy).mean()),
       balanced_accuracy=float(np.nanmean(rec)),brier=float(np.mean(np.sum((pp-Y)**2,1))),
       nll_clipped=float(-np.log(np.maximum(pp[np.arange(len(yy)),yy],1e-8)).mean()),
       nll_raw='Infinity' if (pp[np.arange(len(yy)),yy]==0).any() else float(-np.log(pp[np.arange(len(yy)),yy]).mean()),
       epsilon=1e-8,
       true_zero_labels=int((pp[np.arange(len(yy)),yy]==0).sum()),
       human_shares=Y.mean(0).tolist(),model_shares=pp.mean(0).tolist(),confusion=cm.tolist())

def main():
    selected=rows(OUT/'selected_states.jsonl')
    aggregates={(r['city'],str(r['respondent_id']),str(r['task'])):r for r in rows(OUT/'aggregates.jsonl')}
    calls={(r['city'],str(r['respondent_id']),str(r['task']),r['repeat']):r for r in rows(OUT/'calls.jsonl') if r.get('valid')}
    specs=read(SUR/'contrasts.json');cfg=read(OUT/'config.json')
    response_rows=[];metric_rows=[];repeat_rows=[];all_detail={};closure=0.
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    for city in ['Singapore','Shanghai']:
        grid=read(SUR/f'grid_{city.lower()}.json');tasks=grid['tasks']
        ids=list(dict.fromkeys(str(r['respondent_id']) for r in selected if r['city']==city))
        for rid in ids:
            for task in tasks:
                if aggregates.get((city,rid,task),{}).get('repeats') != 3:
                    raise RuntimeError('Incomplete Teacher group: analysis requires all selected tasks and repeats')
        index=[grid['respondents'].index(rid) for rid in ids]
        y=np.array(grid['labels'])[index]
        T=np.array([[aggregates[city,rid,t]['mean_probabilities'] for t in tasks] for rid in ids])
        TR=np.array([[[[calls[city,rid,t,k]['action']['mode_probabilities'].get(m,0.) for m in MODES] for t in tasks] for rid in ids] for k in range(3)])
        P={'teacher':T}
        source_dir=SUR/'predictions'/city.lower()/cfg['profile']
        for p in sorted(source_dir.glob('*.npz')):
            P[p.stem]=np.load(p)['probabilities'][index]
        assert len(P)==15, (city,list(P))
        for fam in ['soft_kl','ce_kl','signed_l1','direction_magnitude']:
            P[fam+'_seedmean']=np.mean([P[f'{fam}_seed{s}'] for s in [42,2026,7]],0)
        bycity={}
        for model,p in P.items():
            if model.endswith('_seedmean'):continue
            m=metrics(p,y);bycity[model]=m
            metric_rows.append(dict(city=city,model=model,n_respondents=len(ids),
                       **{k:v for k,v in m.items() if not isinstance(v,list)},
                       pt_share_bias=m['model_shares'][1]-m['human_shares'][1]))
        for contrast,s in specs[city].items():
            tt=[tasks.index(t) for t in s['weights']];w=np.array(list(s['weights'].values()))
            mm=[MODES.index(m) for m in s['modes']]
            keep=(y[:,tt]>=0).all(1)
            YY=np.eye(4)[np.maximum(y,0)]
            h=(YY[:,tt,:][:,:,mm].sum(2)*w).sum(1)[keep]
            tresp=(T[:,tt,:][:,:,mm].sum(2)*w).sum(1)[keep]
            n=int(keep.sum());idx=rng.integers(n,size=(cfg['bootstrap_draws'],n),dtype=np.int32)
            family=len(specs[city]);ci_h=interval(h,idx,family);ci_t=interval(tresp,idx,family)
            ci_th=interval(tresp-h,idx,family)
            for model,p in P.items():
                sr=(p[:,tt,:][:,:,mm].sum(2)*w).sum(1)[keep]
                closure=max(closure,float(np.max(np.abs((sr-h)-((tresp-h)+(sr-tresp))))))
                vals={'human':ci_h,'teacher':ci_t,'teacher_minus_human':ci_th,
                      'student':interval(sr,idx,family),'student_minus_teacher':interval(sr-tresp,idx,family),
                      'student_minus_human':interval(sr-h,idx,family)}
                row=dict(city=city,contrast=contrast,model=model,n=n)
                for key,stat in vals.items():
                    for k,v in stat.items():row[f'{key}_{k}']=v
                # Absolute discrepancy of aggregate response, not mean individual error.
                row.update(abs_teacher_bias=abs(float((tresp-h).mean())),
                           abs_student_bias=abs(float((sr-h).mean())),
                           absolute_bias_change=abs(float((sr-h).mean()))-abs(float((tresp-h).mean())))
                # The absolute aggregate-bias contrast is nonlinear: evaluate it
                # on each common respondent resample instead of adding bounds.
                boot_abs_change=np.abs((sr-h)[idx].mean(1))-np.abs((tresp-h)[idx].mean(1))
                row.update(absolute_bias_change_low=float(np.quantile(boot_abs_change,.025)),
                           absolute_bias_change_high=float(np.quantile(boot_abs_change,.975)),
                           absolute_bias_change_family_low=float(np.quantile(boot_abs_change,.05/(2*family))),
                           absolute_bias_change_family_high=float(np.quantile(boot_abs_change,1-.05/(2*family))))
                response_rows.append(row)
            for k in range(3):
                tr=(TR[k][:,tt,:][:,:,mm].sum(2)*w).sum(1)[keep]
                repeat_rows.append(dict(city=city,contrast=contrast,repeat=k,n=n,
                               teacher_response=float(tr.mean()),teacher_bias=float((tr-h).mean())))
        bycity['n_respondents']=len(ids);bycity['tasks']=tasks;all_detail[city]=bycity
    assert closure<1e-12
    csvout(OUT/'same_task_responses.csv',response_rows)
    csvout(OUT/'same_task_choices.csv',metric_rows)
    csvout(OUT/'same_task_repeat_responses.csv',repeat_rows)
    dump(OUT/'same_task_details.json',all_detail)
    summary={'additive_identity_max_abs_error':closure,'respondents':{c:v['n_respondents'] for c,v in all_detail.items()},
        'bootstrap_draws':cfg['bootstrap_draws'],'bootstrap_seed':cfg['bootstrap_seed'],
        'unit':'respondent; same draws across all models and decomposition terms',
        'family':'Bonferroni within5 Singapore /8 Shanghai contrasts; no claim of global FWER across all methods',
        'model_seedmean':'mean response across three separately fitted models, not an accuracy evaluation of an ensemble',
        'scope':'stratified selected existing respondents; contemporary API model; Shanghai reference numerical encoding of qualitative questionnaire',
        'prompt_version':cfg['prompt_version'],'profile':cfg['profile'],
        'unused_metadata_policy':cfg.get('unused_metadata_policy','Original task/place metadata and city-bearing persona IDs retained; diagnostic interface'),
        'attribution_boundary':'Shared numerical and categorical raw attributes and availability, not identical language versus numerical encoders. Contemporary Teacher differences do not isolate historical compression effects.',
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'selected_states.jsonl',OUT/'calls.jsonl',OUT/'aggregates.jsonl',Path(__file__)]}}
    dump(OUT/'analysis_manifest.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--condition',choices=['survey_options','ownership'],default='survey_options');ap.add_argument('--prompt',choices=['generic','accessibility'],default='generic');a=ap.parse_args()
    if a.prompt=='accessibility':OUT=ROOT/'outputs/revision_20260921/teacher_accessibility_neutral'
    if a.condition=='ownership':OUT=OUT.with_name(OUT.name+'_ownership')
    main()
