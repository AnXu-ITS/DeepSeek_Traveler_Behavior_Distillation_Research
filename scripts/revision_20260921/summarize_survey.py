"""Assemble model/seed comparisons and the questionnaire provenance ledger."""
from pathlib import Path
import csv,json,sys
import numpy as np
from prepare_survey import ROOT,OUT,SG,SH,MODES,readj,rows,writej,writejl,sha
from evaluate_survey import VARIANTS,SEEDS,MODELS,Scorer,csvout

def main():
    # Refresh mapping prose without changing any task state or label.
    audit=readj(OUT/'mapping_audit.json')
    for city,v in audit['city_summary'].items():
        if 'undecided' in v:v['unscored_nonmode']=v.pop('undecided')
    audit['unscored_semantics']='Shanghai two original responses say none of the options suitable (unable), not an explicitly chosen mode. No imputed human label.'
    audit['feature_sources']['Shanghai']['displayed']='Qualitative conditions; explicit values PT fare=6 yuan, parking fee=30 yuan, transfers=1. All four mode options displayed to all eligible respondents.'
    audit['feature_sources']['Shanghai']['unmeasured']=['Household size=2 reference (sensitivity1,4)','Numerical travel times, distance, departure, PT component splits, baseline fares/costs, rain intensity and intervention magnitudes are reference encoding from cards.json, not displayed online.','Parking30 is shown as parking fee, whereas model has an undivided car monetary-cost attribute. Composite taxi/private-car utility remains unresolved.']
    audit['instrument_audit']=dict(questionnaire_text_sha256=sha(ROOT/'docs/plans/shanghai_survey_v3/field/上海出行方式调查_问卷.md'),html_sha256=sha(ROOT/'docs/plans/shanghai_survey_v3/field/上海出行方式调查_原始页面.html'),
        images={p.name:dict(sha256=sha(p),content='Decorative background or generic survey banner; no task tables, times, prices, or instructions') for p in (OUT/'instrument_audit').glob('header_image_*')},
        conclusion='Singapore task numbers verified against displayed questionnaire. Shanghai requires declared numerical encoding of qualitative tasks; an exact same-numeric-task human claim is unsupported.')
    writej(OUT/'mapping_audit.json',audit)
    # Per-task field provenance, preserving actual displayed facts versus numeric references.
    primary=rows(OUT/'task_states.jsonl');seen=set();ledger=[]
    for r in primary:
        key=(r['city'],r['task'])
        if key in seen:continue
        seen.add(key);st=r['state'];city,task=key
        for alt in st['alternatives']:
            for field in ['available','travel_time_min','monetary_cost','access_time_min','transfers','pt_feasible','egress_time_min','wait_time_min','in_vehicle_time_min','transfer_time_min','coverage_ratio','reliability_delay_min','weather_exposure']:
                source='answer-independent reference encoding'
                if city=='Singapore' and field in ['travel_time_min','monetary_cost']:source='displayed numeric task attribute (car drive/taxi cost mapped by access and license)'
                if city=='Singapore' and alt['mode']=='pt' and field in ['access_time_min','transfers']:source='displayed numeric task attribute'
                if field=='available':source='all four displayed mode options; ownership rule separate sensitivity'
                if city=='Shanghai' and ((task=='F1' and alt['mode']=='pt' and field=='monetary_cost') or (task=='A_TRANSFER' and alt['mode']=='pt' and field=='transfers')):source='displayed intervention value'
                if city=='Shanghai' and task=='P1' and alt['mode']=='car' and field=='monetary_cost':source='displayed parking30 mapped to undivided car cost; taxi/private-car construct mismatch'
                ledger.append(dict(city=city,task=task,mode=alt['mode'],field=field,encoded_value=alt[field],source=source))
    csvout(OUT/'task_field_sources.csv',ledger)
    scores=list(csv.DictReader((OUT/'model_scores.csv').open(encoding='utf-8-sig')))
    summaries=[]
    for city in ['Singapore','Shanghai']:
        profiles=sorted({r['profile'] for r in scores if r['city']==city})
        for profile in profiles:
            for model in ['s9','mnl_s']+VARIANTS:
                rs=[r for r in scores if r['city']==city and r['profile']==profile and (r['model']==model or r['model'].startswith(model+'_seed'))]
                if not rs:continue
                row=dict(city=city,profile=profile,model=model,n_training_seeds=len(rs))
                for metric in ['accuracy','balanced_accuracy','brier','nll_clipped','pt_bias','mean_absolute_response_bias']:
                    x=np.array([float(r[metric]) for r in rs]);row[metric+'_mean']=float(x.mean());row[metric+'_sd']=float(x.std(ddof=1)) if len(x)>1 else 0.
                summaries.append(row)
    csvout(OUT/'seed_summary.csv',summaries)
    # All primary-choice probability predictions for direct Teacher joins.
    lookup={(r['city'],r['respondent_id'],r['task']):r for r in primary}
    full=[];comparisons=[];family_response={};family_choice_intervals=[]
    for city in ['Singapore','Shanghai']:
        family_response[city]={}
        grid=readj(OUT/f'grid_{city.lower()}.json');ids=grid['respondents'];tasks=grid['tasks'];y=np.array(grid['labels']);scorer=Scorer(city,ids,tasks,y)
        for profile in ['survey_options','ownership','car_relaxed']:
            family_response[city][profile]={}
            preds={name:np.load(OUT/f'predictions/{city.lower()}/{profile}/{name}.npz')['probabilities'] for name in MODELS}
            if profile=='survey_options':
                for name,P in preds.items():
                    for i,rid in enumerate(ids):
                        for j,task in enumerate(tasks):
                            full.append(dict(city=city,respondent_id=rid,task=task,state_id=lookup[city,rid,task]['state_id'],model=name,probabilities=P[i,j].tolist()))
            valid=y>=0;Y=np.eye(4)[np.maximum(y,0)];rowmean=lambda x:np.where(valid,x,0).sum(1)/valid.sum(1)
            for family in ['s9','mnl_s']+VARIANTS:
                names=[family] if family in ['s9','mnl_s'] else [f'{family}_seed{s}' for s in SEEDS]
                pmean=np.mean([preds[n] for n in names],axis=0)
                rs=scorer.score(pmean,ci=True)['responses']
                for contrast,r in rs.items():
                    original=[readj(OUT/f'predictions/{city.lower()}/{profile}/{n}.json')['responses'][contrast] for n in names]
                    r['training_seed_sd']=float(np.std([x['model'] for x in original],ddof=1)) if len(names)>1 else 0.
                    r['n_training_seeds']=len(names)
                family_response[city][profile][family]=dict(responses=rs,aggregation='Response of mean probabilities across fixed training seeds; bootstrap resamples shared respondents, not seeds',n_training_seeds=len(names))
                writej(OUT/f'predictions/{city.lower()}/{profile}/{family}_seedmean_response.json',family_response[city][profile][family])
                metric_units={'accuracy':[],'brier':[],'nll_clipped':[]}
                for n in names:
                    p=preds[n];truep=p[np.arange(len(ids))[:,None],np.arange(10)[None,:],np.maximum(y,0)]
                    metric_units['accuracy'].append(rowmean((p.argmax(2)==y).astype(float)))
                    metric_units['brier'].append(rowmean(((p-Y)**2).sum(2)))
                    metric_units['nll_clipped'].append(rowmean(-np.log(np.maximum(truep,1e-8))))
                for metric,units in metric_units.items():
                    interval=scorer.interval(np.mean(units,axis=0))
                    family_choice_intervals.append(dict(city=city,profile=profile,model=family,metric=metric,n_training_seeds=len(names),respondent_mean=interval['mean'],ci95_low=interval['ci95'][0],ci95_high=interval['ci95'][1],aggregation='arithmetic mean of per-seed per-respondent metrics, not ensemble classification'))
            for method in ['ce_kl','signed_l1','direction_magnitude']:
                for seed in SEEDS:
                    A=preds[f'{method}_seed{seed}'];B=preds[f'soft_kl_seed{seed}']
                    vals=dict(accuracy=rowmean((A.argmax(2)==y).astype(float)-(B.argmax(2)==y)),brier=rowmean(((A-Y)**2).sum(2)-((B-Y)**2).sum(2)),
                              pt_bias=rowmean(A[:,:,1]-B[:,:,1]))
                    for metric,v in vals.items():
                        interval=scorer.interval(v);comparisons.append(dict(city=city,profile=profile,method=method,seed=seed,reference='soft_kl_same_seed',metric=metric,mean=interval['mean'],ci95_low=interval['ci95'][0],ci95_high=interval['ci95'][1]))
                    for name,tt,w,m,keep,h in scorer.conditions:
                        da=(A[:,tt,:][:,:,m].sum(2)*w).sum(1);db=(B[:,tt,:][:,:,m].sum(2)*w).sum(1)
                        v=da-db;interval=scorer.interval(v,keep)
                        comparisons.append(dict(city=city,profile=profile,method=method,seed=seed,reference='soft_kl_same_seed',metric='response_bias_'+name,mean=interval['mean'],ci95_low=interval['ci95'][0],ci95_high=interval['ci95'][1]))
    writejl(OUT/'primary_model_predictions.jsonl',full);csvout(OUT/'paired_model_comparisons.csv',comparisons)
    writej(OUT/'family_responses.json',family_response);csvout(OUT/'family_choice_intervals.csv',family_choice_intervals)
    writej(OUT/'summary_manifest.json',dict(files={p.name:sha(p) for p in [OUT/'task_states.jsonl',OUT/'human_labels.jsonl',OUT/'model_scores.csv',OUT/'model_responses.csv',OUT/'seed_summary.csv',OUT/'paired_model_comparisons.csv',OUT/'primary_model_predictions.jsonl',OUT/'task_field_sources.csv',OUT/'family_responses.json',OUT/'family_choice_intervals.csv']},
       bootstrap='10000 resamples seed917; same respondents and same resampling matrix for every model and training seed within each city; complete task panels retained',
       multiplicity='Shanghai8 and Singapore5 contrast-family Bonferroni intervals accompany nominal95 intervals. Paired model comparisons exploratory95 intervals; no multiplicity-adjusted superiority claim.',
       seed_uncertainty='Individual training seeds and sample SD reported separately from respondent resampling. No survey data used for fitting or selection.'))
    print('Survey summary complete',len(full),'prediction rows;',len(comparisons),'paired comparisons',flush=True)

if __name__=='__main__':main()
