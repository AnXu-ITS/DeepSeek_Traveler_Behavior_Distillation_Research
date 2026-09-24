"""Read-only re-analysis checks for the complete neutral accessibility Teacher.

No API calls and no mutation of Teacher calls, aggregates or analysis outputs.
Writes a separate audit. Missing groups withhold result validation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from collections import Counter,defaultdict
from pathlib import Path
import csv,hashlib,json
import numpy as np
from prepare_survey import ROOT,OUT as SUR,MODES,readj,rows,writej,sha
from analyze_prompt_sensitivity import snapshot_jsonl,object_sha
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.teacher.validator import TeacherResponseValidator
from traveler_distillation.teacher.prompts_s8 import S8_SYSTEM_PROMPT

MAIN=ROOT/'outputs/revision_20260921/teacher_accessibility_neutral_ownership'
DEST=ROOT/'outputs/revision_20260921/teacher_validation'
FAMILIES=['soft_kl','ce_kl','signed_l1','direction_magnitude']

def key(r):return r['city'],str(r['respondent_id']),str(r['task'])
def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def main():
    DEST.mkdir(exist_ok=True)
    selected=rows(MAIN/'selected_states.jsonl');lookup={key(r):r for r in selected};assert len(selected)==len(lookup)==480
    config=readj(MAIN/'config.json');prompt=(MAIN/'system_prompt.txt').read_text(encoding='utf-8');template=(MAIN/'user_prompt_template.txt').read_text(encoding='utf-8')
    assert prompt==S8_SYSTEM_PROMPT and config['profile']=='ownership' and config['repeats']==3
    old_input_audit=readj(MAIN/'independent_student_feature_audit.json')
    assert old_input_audit['status']=='passed' and old_input_audit['before_main_teacher_calls']
    assert old_input_audit['source_sha256'][str((MAIN/'selected_states.jsonl').relative_to(ROOT))]==sha(MAIN/'selected_states.jsonl')
    targets={k+(r,) for k in lookup for r in range(3)};calls,partial,snapshot_sha,nbytes=snapshot_jsonl(MAIN/'calls.jsonl')
    successes=defaultdict(list);failures=Counter();out_of_cohort=[];hashfail=[];max_p_sum_error=0.;validator=TeacherResponseValidator()
    for i,c in enumerate(calls,1):
        k=key(c)+(int(c['repeat']),)
        if k not in targets:out_of_cohort.append(i);continue
        r=lookup[k[:3]];assert c['state_sha256']==r['state_sha256']==object_sha(r['state'])
        st=UniversalTravelerState.model_validate(r['state']);user=template.replace('{state_json}',st.model_dump_json(indent=2))
        caps=[c['max_tokens']] if 'max_tokens' in c else config['retry_token_budgets']
        requests=[dict(model=config['model'],messages=[dict(role='system',content=prompt),dict(role='user',content=user)],temperature=config['temperature'],max_tokens=cap) for cap in caps]
        if not any(object_sha(r)==c['request_sha256'] for r in requests):hashfail.append(i)
        if c.get('valid'):
            a=UniversalTravelerAction.model_validate(c['action']);check=validator.validate(st,a);assert check.valid,check.reason
            max_p_sum_error=max(max_p_sum_error,abs(sum(a.mode_probabilities.values())-1))
            successes[k].append(c)
        else:failures[c.get('failure_type',c.get('validation_reason','unknown'))]+=1
    missing=targets-set(successes);duplicates={k:len(v) for k,v in successes.items() if len(v)!=1}
    needfiles=['aggregates.jsonl','analysis_manifest.json','same_task_responses.csv','same_task_choices.csv','same_task_repeat_responses.csv','same_task_details.json']
    absent=[name for name in needfiles if not (MAIN/name).exists()]
    preliminary=dict(n_selected_states=480,expected_repeat_targets=1440,successful_repeat_targets=len(successes),missing_repeat_targets=len(missing),duplicate_success_targets=len(duplicates),out_of_cohort_records=out_of_cohort,request_hash_failures=hashfail,partial_final_line=partial,source_calls_snapshot_sha256=snapshot_sha,source_calls_snapshot_bytes=nbytes,failure_counts=dict(failures),missing_analysis_outputs=absent)
    if missing or duplicates or out_of_cohort or hashfail or partial or absent:
        writej(DEST/'primary_audit_status.json',dict(status='waiting_or_input_inconsistency',**preliminary));print(json.dumps(preliminary));return 2
    aggregates={key(r):r for r in rows(MAIN/'aggregates.jsonl')};assert set(aggregates)==set(lookup)
    request_models=Counter();completion_ids=[];aggregate_errors=[]
    for k,r in aggregates.items():
        rr=[successes[k+(rep,)][0] for rep in range(3)];pp=np.array([[c['action']['mode_probabilities'].get(m,0.) for m in MODES] for c in rr]);dd=np.array([c['action']['departure_time_shift_min'] for c in rr])
        assert r['repeats']==3 and r['state_sha256']==lookup[k]['state_sha256']
        error=max(float(np.max(abs(pp.mean(0)-r['mean_probabilities']))),float(np.max(abs(pp.std(0,ddof=1)-r['sd_probabilities']))),float(abs(dd.mean()-r['mean_departure'])))
        assert error<1e-12;aggregate_errors.append(error)
        request_models.update(c['model'] for c in rr);completion_ids.extend(c['completion_id'] for c in rr)
    assert len(set(completion_ids))==1440
    response={(r['city'],r['contrast'],r['model']):r for r in csvread(MAIN/'same_task_responses.csv')};choice={(r['city'],r['model']):r for r in csvread(MAIN/'same_task_choices.csv')};repeats={(r['city'],r['contrast'],int(r['repeat'])):r for r in csvread(MAIN/'same_task_repeat_responses.csv')}
    assert len(response)==247 and len(choice)==30 and len(repeats)==39
    specs=readj(SUR/'contrasts.json');rng=np.random.default_rng(config['bootstrap_seed']);checks=[];max_error=0.;abs_ci_max_error=0.;closure=0.;case_counts={}
    def verify_value(actual,expected):
        nonlocal max_error
        error=abs(float(actual)-float(expected));max_error=max(max_error,error);assert error<1e-11,(actual,expected)
    for city in ['Singapore','Shanghai']:
        grid=readj(SUR/f'grid_{city.lower()}.json');tasks=grid['tasks'];ids=list(dict.fromkeys(str(r['respondent_id']) for r in selected if r['city']==city));assert len(ids)==24
        pos=[grid['respondents'].index(i) for i in ids];y=np.array(grid['labels'])[pos];valid=y>=0;onehot=np.eye(4)[np.maximum(y,0)]
        T=np.array([[aggregates[city,rid,t]['mean_probabilities'] for t in tasks] for rid in ids]);TR=np.array([[[[successes[(city,rid,t,rep)][0]['action']['mode_probabilities'].get(m,0.) for m in MODES] for t in tasks] for rid in ids] for rep in range(3)])
        P={'teacher':T}
        for p in sorted((SUR/'predictions'/city.lower()/'ownership').glob('*.npz')):P[p.stem]=np.load(p)['probabilities'][pos]
        assert len(P)==15
        for family in FAMILIES:P[family+'_seedmean']=np.mean([P[f'{family}_seed{s}'] for s in [42,2026,7]],axis=0)
        for model,prob in P.items():
            if model.endswith('_seedmean'):continue
            row=choice[city,model];p=prob[valid];labels=y[valid];pred=p.argmax(1);Y=np.eye(4)[labels];cm=np.zeros((4,4),int);np.add.at(cm,(labels,pred),1)
            supported=cm.sum(1)>0;truep=p[np.arange(len(p)),labels]
            vals=dict(n_respondents=24,n_choices=len(p),accuracy=(pred==labels).mean(),balanced_accuracy=(cm.diagonal()[supported]/cm.sum(1)[supported]).mean(),brier=np.square(p-Y).sum(1).mean(),nll_clipped=-np.log(np.maximum(truep,1e-8)).mean(),true_zero_labels=(truep==0).sum(),pt_share_bias=p[:,1].mean()-Y[:,1].mean(),epsilon=1e-8)
            for f,v in vals.items():verify_value(row[f],v)
            if (truep==0).any():assert row['nll_raw']=='Infinity'
            else:verify_value(row['nll_raw'],-np.log(truep).mean())
        case_counts[city]={}
        for contrast,spec in specs[city].items():
            ti=[tasks.index(t) for t in spec['weights']];weights=np.array(list(spec['weights'].values()));modeindices=[MODES.index(m) for m in spec['modes']]
            keep=(y[:,ti]>=0).all(1);n=int(keep.sum());case_counts[city][contrast]=n
            h=(onehot[:,ti,:][:,:,modeindices].sum(2)*weights).sum(1)[keep]
            t=(T[:,ti,:][:,:,modeindices].sum(2)*weights).sum(1)[keep]
            indices=rng.integers(n,size=(config['bootstrap_draws'],n),dtype=np.int32);family=len(specs[city]);q=[.025,.975,.05/(2*family),1-.05/(2*family)]
            for model,p in P.items():
                s=(p[:,ti,:][:,:,modeindices].sum(2)*weights).sum(1)[keep];row=response[city,contrast,model];assert int(row['n'])==n
                vectors={'human':h,'teacher':t,'teacher_minus_human':t-h,'student':s,'student_minus_teacher':s-t,'student_minus_human':s-h}
                # Each bootstrap term is recomputed from per-person endpoint
                # differences; no addition of interval endpoints is permitted.
                for name,v in vectors.items():
                    verify_value(row[name+'_mean'],v.mean());quant=np.quantile(np.mean(v[indices],axis=1),q)
                    for suffix,z in zip(['low','high','family_low','family_high'],quant):verify_value(row[name+'_'+suffix],z)
                verify_value(row['abs_teacher_bias'],abs((t-h).mean()));verify_value(row['abs_student_bias'],abs((s-h).mean()));verify_value(row['absolute_bias_change'],abs((s-h).mean())-abs((t-h).mean()))
                aboot=np.abs(np.mean((s-h)[indices],axis=1))-np.abs(np.mean((t-h)[indices],axis=1));qq=np.quantile(aboot,q)
                for suffix,z in zip(['low','high','family_low','family_high'],qq):
                    error=abs(float(row['absolute_bias_change_'+suffix])-z);abs_ci_max_error=max(abs_ci_max_error,float(error));assert error<1e-11
                closure=max(closure,float(np.max(abs((s-h)-((t-h)+(s-t))))))
            for rep in range(3):
                t_rep=(TR[rep][:,ti,:][:,:,modeindices].sum(2)*weights).sum(1)[keep];row=repeats[city,contrast,rep];assert int(row['n'])==n
                verify_value(row['teacher_response'],t_rep.mean());verify_value(row['teacher_bias'],(t_rep-h).mean())
            checks.append(dict(city=city,contrast=contrast,n=n,n_models=19,all_signed_and_absolute_bias_intervals_recomputed=True))
    manifest=readj(MAIN/'analysis_manifest.json')
    assert manifest['source_hashes'][str((MAIN/'calls.jsonl').relative_to(ROOT))]==snapshot_sha
    assert sha(MAIN/'calls.jsonl')==snapshot_sha,'Calls changed while independent final audit ran'
    summary=dict(status='passed',**preliminary,n_successful_unique_completion_ids=1440,n_aggregate_states=480,n_response_rows=247,n_choice_metric_rows=30,n_repeat_response_rows=39,
        requested_model=config['model'],returned_model_counts=dict(request_models),max_valid_probability_sum_error=max_p_sum_error,aggregate_mean_sd_departure_max_error=max(aggregate_errors),
        choice_and_signed_response_metric_max_error=max_error,absolute_bias_change_interval_max_error=abs_ci_max_error,additive_unit_identity_max_error=closure,per_contrast_n=case_counts,
        before_call_actual_14_extractor_audit_unchanged=True,original_cohort_and_predictions_unchanged=True,no_api_calls=True,checks=checks,
        scope='Conditional empirical comparison on the preselected 24-person-per-city cohort. Contemporary Teacher and original fitted Students share input fields/support; language and numeric encoding are different, so neither signed decomposition nor absolute-bias improvement isolates historical compression causally.',
        source_sha256={str((MAIN/f).relative_to(ROOT)):sha(MAIN/f) for f in ['selected_states.jsonl','calls.jsonl','aggregates.jsonl','same_task_responses.csv','same_task_choices.csv','same_task_repeat_responses.csv','analysis_manifest.json','independent_student_feature_audit.json']},code_sha256=sha(__file__))
    writej(DEST/'primary_result_audit.json',summary);writej(DEST/'primary_audit_status.json',dict(status='passed',n_states=480,n_repeats=1440))
    (DEST/'PRIMARY_RESULT_AUDIT.md').write_text('# Independent primary Teacher result audit\n\nAll 480 preselected states have exactly three successful, request-hash-matched calls (1,440 distinct completion IDs). Failed attempts remain in the append-only source. Aggregated means, sample SDs and departure means reproduce the individual calls.\n\nAll 30 choice rows, 247 signed/absolute-response rows and 39 repeat-response rows were independently recomputed. Confidence bounds for absolute aggregate bias changes use the nonlinear statistic in each shared respondent resample; they are not sums of bounds or individual-level absolute errors. All checks passed to numerical precision. Complete-case denominators are recorded per contrast.\n\nThe previously frozen 6,720 Student-feature comparisons still refer to the unchanged selected states. This verifies numerical input equivalence for each fitted Student, not equivalence of language versus numerical encoders, representative sampling, or causal historical compression effects.\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',n_states=480,n_repeats=1440,metric_max_error=max_error,absolute_ci_max_error=abs_ci_max_error,counts=case_counts)))
    return 0

if __name__=='__main__':raise SystemExit(main())
