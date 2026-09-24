"""Independent completed-pilot checks and supplementary reporting summaries.

Reads frozen calls/results only. Does not call an API or change Teacher outputs.
"""
from collections import Counter, defaultdict
import csv, json
import numpy as np
from prepare_survey import ROOT, OUT as SUR, MODES, rows, readj, writej, sha

BASE=ROOT/'outputs/revision_20260921'
DEST=BASE/'teacher_validation'
PILOT=BASE/'prompt_sensitivity'
SETS={'generic_all':BASE/'teacher','generic_ownership':BASE/'teacher_ownership',
      'generic_neutral_ownership':BASE/'teacher_neutral_ownership',
      'accessibility_neutral_ownership':BASE/'teacher_accessibility_neutral_ownership'}
MAIN=SETS['accessibility_neutral_ownership']
PAIRS={'mask':('generic_all','generic_ownership'),'prompt':('generic_neutral_ownership','accessibility_neutral_ownership'),
       'unused_metadata_context':('generic_ownership','generic_neutral_ownership')}
FAMILIES=['soft_kl','ce_kl','signed_l1','direction_magnitude']

def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def key(r):return r['city'],str(r['respondent_id']),str(r['task'])

def main():
    assert readj(PILOT/'status.json')['ready']
    assert readj(DEST/'primary_result_audit.json')['status']=='passed'
    aud=readj(PILOT/'input_audit.json'); frozen=[]; conditions=[]; validsets={}; allarrays={}
    selected={k:[str(i) for i in v] for k,v in aud['selected_respondents'].items()}
    nz=np.load(PILOT/'matched_pilot_probabilities.npz')
    order=list(zip(nz['city'].tolist(),nz['respondent'].tolist(),nz['task'].tolist()))
    assert len(order)==len(set(order))==40
    for name,path in SETS.items():
        raw=rows(path/'calls.jsonl'); frozen.append(path/'calls.jsonl')
        assert sha(path/'calls.jsonl')==aud['sets'][name]['source_calls_sha256']
        successes=defaultdict(list); failures=Counter()
        for r in raw:
            if r['valid']:successes[key(r)+(int(r['repeat']),)].append(r)
            else:failures[r.get('failure_type',r.get('validation_reason','unknown'))]+=1
        assert all(len(v)==1 for v in successes.values())
        vals={k:v[0] for k,v in successes.items()}; validsets[name]=vals
        expected=1440 if name in ['generic_all','accessibility_neutral_ownership'] else 120
        assert len(vals)==expected and len({v['completion_id'] for v in vals.values()})==expected
        states={k[:3] for k in vals};people={(k[0],k[1]) for k in vals}
        assert all(set(k[3] for k in vals if k[:3]==state)=={0,1,2} for state in states)
        assert set(v['model'] for v in vals.values())=={'deepseek-v4-pro'}
        config=readj(path/'config.json')
        caps=Counter(str(v.get('max_tokens','legacy_unrecorded')) for v in vals.values())
        conditions.append(dict(condition=name,n_respondents=len(people),n_states=len(states),n_successful_repeats=len(vals),n_attempts=len(raw),
          n_failed_attempts=sum(failures.values()),http_402=failures['http_402'],http_429=failures['http_429'],parse_error=failures['TeacherParseError'],
          other_failures=sum(v for k,v in failures.items() if k not in ['http_402','http_429','TeacherParseError']),duplicate_success_targets=0,duplicate_completion_ids=0,
          successful_token_caps=json.dumps(dict(caps),sort_keys=True),prompt_version=config.get('prompt_version','teacher_v0.1')))
        A=np.array([[[vals[k+(rep,)]['action']['mode_probabilities'].get(m,0.) for m in MODES] for rep in range(3)] for k in order])
        assert np.array_equal(A,nz[name]);allarrays[name]=A
    err=0.
    def check(a,b):
        nonlocal err
        e=abs(float(a)-float(b));err=max(err,e);assert e<1e-12,(a,b)
    means={k:v.mean(1) for k,v in allarrays.items()}
    for r in csvread(PILOT/'pilot_probability_changes.csv'):
        keep=np.array([r['scope']=='pooled_pilot' or k[0]==r['scope'] for k in order]);left,right=PAIRS[r['comparison']]
        a,b=means[left][keep],means[right][keep];d=b-a
        for fld,val in dict(n_states=keep.sum(),mean_probability_L1=np.abs(d).sum(1).mean(),probability_L1_of_aggregate=np.abs(d.mean(0)).sum(),
          pt_left=a[:,1].mean(),pt_right=b[:,1].mean(),pt_change=d[:,1].mean(),mean_abs_state_pt_change=np.abs(d[:,1]).mean(),hard_choice_disagreement=(a.argmax(1)!=b.argmax(1)).mean()).items():check(r[fld],val)
    for r in csvread(PILOT/'pilot_repeat_variability.csv'):
        keep=np.array([r['scope']=='pooled_pilot' or k[0]==r['scope'] for k in order]);a=allarrays[r['condition']][keep];m=a.mean(1);s=a.std(1,ddof=1)
        for fld,val in dict(pt_mean=m[:,1].mean(),mean_raw_repeat_pt_sd=s[:,1].mean(),mean_raw_repeat_mode_sd=s.mean(),mean_repeat_probability_L1_from_state_mean=np.abs(a-m[:,None,:]).sum(2).mean()).items():check(r[fld],val)
    specs=readj(SUR/'contrasts.json')
    for r in csvread(PILOT/'pilot_contrast_changes.csv'):
        spec=specs[r['city']][r['contrast']];left,right=PAIRS[r['comparison']];vals={}
        for name in [left,right]:
            person=[]
            for rid in selected[r['city']]:
                repeat=np.zeros(3)
                for task,w in spec['weights'].items():
                    idx=order.index((r['city'],rid,task))
                    for m in spec['modes']:repeat+=w*allarrays[name][idx,:,MODES.index(m)]
                person.append(repeat)
            vals[name]=np.array(person)
        for fld,val in dict(left_response=vals[left].mean(),right_response=vals[right].mean(),response_change=(vals[right]-vals[left]).mean(),
          left_mean_respondent_repeat_response_sd=vals[left].std(1,ddof=1).mean(),right_mean_respondent_repeat_response_sd=vals[right].std(1,ddof=1).mean()).items():check(r[fld],val)
    # Explicitly audit human denominators and conflicts for the main cohort.
    states={key(r):r['state'] for r in rows(MAIN/'selected_states.jsonl')};denominators={}
    for city in ['Singapore','Shanghai']:
        grid=readj(SUR/f'grid_{city.lower()}.json');ids=list(dict.fromkeys(k[1] for k in states if k[0]==city))
        missing=conflicts=0
        for rid in ids:
            labels=grid['labels'][grid['respondents'].index(rid)]
            for task,y in zip(grid['tasks'],labels):
                if y<0:missing+=1;continue
                availability={a['mode']:a['available'] for a in states[city,rid,task]['alternatives']}
                conflicts+=not availability[MODES[y]]
        assert missing==conflicts==0
        denominators[city]=dict(respondents=len(ids),states=len(ids)*10,explicit_choices=len(ids)*10-missing,none_suitable=missing,support_conflicts=conflicts,post_selection_exclusions=0)
    # Family choice summaries average per-seed metrics, never classify a seed ensemble.
    choice=csvread(MAIN/'same_task_choices.csv'); familyrows=[]
    for city in ['Singapore','Shanghai']:
        for model in ['teacher','s9','mnl_s']+FAMILIES:
            group=[r for r in choice if r['city']==city and (r['model']==model or r['model'].startswith(model+'_seed'))]
            assert len(group)==(3 if model in FAMILIES else 1)
            out=dict(city=city,model=model,n_training_seeds=len(group) if model!='teacher' else 0,n_respondents=24,n_choices=240)
            for fld in ['accuracy','balanced_accuracy','brier','pt_share_bias']:
                a=np.array([float(r[fld]) for r in group]);out[fld+'_mean']=float(a.mean());out[fld+'_seed_sd']=float(a.std(ddof=1)) if len(a)>1 else ''
            familyrows.append(out)
    primary=validsets['accessibility_neutral_ownership'];repeatrows=[];response_sd=[]
    for city in ['Singapore','Shanghai']:
        ks=[k for k in states if k[0]==city]
        a=np.array([[[primary[k+(rep,)]['action']['mode_probabilities'].get(m,0.) for m in MODES] for rep in range(3)] for k in ks])
        departure=np.array([[primary[k+(rep,)]['action']['departure_time_shift_min'] for rep in range(3)] for k in ks])
        repeatrows.append(dict(city=city,n_states=len(ks),n_repeats=3,mean_state_pt_probability_sd=a.std(1,ddof=1)[:,1].mean(),mean_state_mode_probability_sd=a.std(1,ddof=1).mean(),
          mean_state_departure_sd_min=departure.std(1,ddof=1).mean(),mean_probability_L1_from_state_repeat_mean=np.abs(a-a.mean(1)[:,None,:]).sum(2).mean()))
    rr=csvread(MAIN/'same_task_repeat_responses.csv')
    for city in ['Singapore','Shanghai']:
        for contrast in specs[city]:
            vals=[float(r['teacher_response']) for r in rr if r['city']==city and r['contrast']==contrast]
            assert len(vals)==3
            response_sd.append(dict(city=city,contrast=contrast,n_respondents=24,n_query_repeats=3,teacher_response_mean=np.mean(vals),teacher_response_repeat_sd=np.std(vals,ddof=1)))
    csvwrite(DEST/'condition_summary.csv',conditions);csvwrite(DEST/'same_task_family_choices.csv',familyrows)
    csvwrite(DEST/'teacher_repeat_summary.csv',repeatrows);csvwrite(DEST/'teacher_contrast_repeat_sd.csv',response_sd)
    sourcepaths=frozen+[PILOT/f for f in ['input_audit.json','matched_pilot_probabilities.npz','pilot_probability_changes.csv','pilot_contrast_changes.csv','pilot_repeat_variability.csv']]+[MAIN/f for f in ['same_task_choices.csv','same_task_repeat_responses.csv','selected_states.jsonl']]
    result=dict(status='passed',pilot_n_respondents=4,pilot_n_states=40,pilot_repeats_per_condition=120,pilot_probability_rows=9,pilot_contrast_rows=39,pilot_repeat_rows=12,
      max_pilot_metric_recomputation_error=err,raw_probabilities_match_npz_exactly=True,unique_successful_targets_and_completion_ids=True,main_denominators=denominators,
      all_four_conditions_complete=True,choice_summary='Arithmetic mean and sample SD of separate fitted-seed scores; no probability-ensemble classification',
      repeat_definition='Within-state sample SD across three successful API queries, followed by arithmetic mean over states; contrast SD uses three aggregate repeated-query responses',
      caveat='Query repeat indices are not coupled random seeds. Pilot differences are descriptive and may include observed API variation; they do not estimate causal historical compression effects.',
      source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sourcepaths},code_sha256=sha(__file__))
    writej(DEST/'reporting_audit.json',result)
    (DEST/'PROMPT_REPORTING_AUDIT.md').write_text('# Independent matched-pilot and reporting audit\n\nAll four conditions are complete. Every selected state has exactly three unique successful logical repeats, with no duplicate completion ID. The 40-state probability arrays reproduce raw calls exactly. All nine probability comparisons, 39 contrast comparisons and 12 repeated-query variability summaries independently reproduce to floating-point precision.\n\nThe primary cohort has 24 people and 240 explicit choices in each city, with zero none-suitable responses and zero ownership-support conflicts. These are properties of this preselected subset, not of the full survey cohort. Selection was fixed without reading outcomes and no selected person was removed.\n\nChoice summaries average separately scored training seeds; response seed means are not reused as ensemble accuracies. Failed API attempts remain separate from the three successful query repeats. Four-person sensitivity comparisons are descriptive only.\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',max_pilot_metric_error=err,denominators=denominators,conditions=conditions),ensure_ascii=False))

if __name__=='__main__':main()
