"""Strict paired prompt/mask pilot audit. Reads existing calls; never calls an API.

The cohort is the first two preselected respondents per city, all ten tasks,
and all three logical repeats. No human answers are read. Missing targets are
reported and comparative metrics are withheld until the entire pilot exists.
"""
from pathlib import Path
from collections import Counter,defaultdict
import copy,hashlib,json,sys
import numpy as np
from prepare_survey import ROOT,OUT as SUR,readj,rows,writej,sha,MODES,ORDER
from evaluate_survey import csvout
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.teacher.prompts import SYSTEM_PROMPT
from traveler_distillation.teacher.prompts_s8 import S8_SYSTEM_PROMPT

BASE=ROOT/'outputs/revision_20260921'
DEST=BASE/'prompt_sensitivity'
SETS={'generic_all':BASE/'teacher','generic_ownership':BASE/'teacher_ownership',
      'generic_neutral_ownership':BASE/'teacher_neutral_ownership',
      'accessibility_neutral_ownership':BASE/'teacher_accessibility_neutral_ownership'}
COMPARISONS={'mask':('generic_all','generic_ownership'),
             'prompt':('generic_neutral_ownership','accessibility_neutral_ownership'),
             'unused_metadata_context':('generic_ownership','generic_neutral_ownership')}

def object_sha(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def key(r):return r['city'],str(r['respondent_id']),str(r['task'])
def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [x for k in sorted(set(a)|set(b)) for x in differences(a.get(k),b.get(k),path+'.'+str(k))]
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return [x for i,(u,v) in enumerate(zip(a,b)) for x in differences(u,v,path+f'[{i}]')]
    return [] if a==b else [path]

def snapshot_jsonl(path):
    if not path.exists():return [],False,None,0
    # Read append-only call logs once. A partial final write is retained as
    # an incomplete snapshot indication and never parsed as a failed model.
    data=path.read_bytes();partial=bool(data and not data.endswith(b'\n'))
    text=data.decode('utf-8-sig');lines=text.splitlines()
    if partial:lines=lines[:-1]
    return [json.loads(x) for x in lines if x.strip()],partial,hashlib.sha256(data).hexdigest(),len(data)

def main():
    DEST.mkdir(exist_ok=True)
    unprepared=[name for name,path in SETS.items() if not (path/'selected_states.jsonl').exists()]
    if unprepared:
        writej(DEST/'status.json',dict(ready=False,waiting_for_preparation=unprepared,comparative_statistics_written=False))
        print(json.dumps(dict(status='waiting_for_prepared_conditions',conditions=unprepared)));return 2
    selection=readj(SETS['generic_all']/'selection.json')
    selected={city:list(ids[:2]) for city,ids in selection['respondents'].items()}
    # Fixed questionnaire task orders; do not open grids containing human labels.
    task_orders={'Singapore':[str(i) for i in range(1,11)],'Shanghai':ORDER}
    required=[(city,rid,str(task)) for city in sorted(selected) for rid in selected[city] for task in task_orders[city]]
    targetkeys={(city,rid,task,rep) for city,rid,task in required for rep in range(3)}
    spec=readj(SUR/'contrasts.json');lookup={};configs={};prompts={};templates={};source_hashes={};audits={};bad=[]
    for name,path in SETS.items():
        look={key(r):r for r in rows(path/'selected_states.jsonl')}
        if len(look)!=480:raise ValueError('Expected unchanged 24 per city selection, ten tasks')
        lookup[name]={k:look[k] for k in required}
        sel=readj(path/'selection.json')
        assert sel['respondents']==selection['respondents']
        assert sel['outcomes_read_during_selection'] is False
        configs[name]=readj(path/'config.json')
        prompts[name]=(path/'system_prompt.txt').read_text(encoding='utf-8')
        templates[name]=(path/'user_prompt_template.txt').read_text(encoding='utf-8')
        for fname in ['config.json','selection.json','selected_states.jsonl','system_prompt.txt','user_prompt_template.txt']:
            source_hashes[str((path/fname).relative_to(ROOT))]=sha(path/fname)
        for r in lookup[name].values():assert object_sha(r['state'])==r['state_sha256']
    assert prompts['generic_all']==prompts['generic_ownership']==prompts['generic_neutral_ownership']==SYSTEM_PROMPT
    assert prompts['accessibility_neutral_ownership']==S8_SYSTEM_PROMPT
    assert len(set(templates.values()))==1
    for param in ['model','temperature','repeats','thinking','retry_token_budgets']:
        assert len({json.dumps(c[param],sort_keys=True) for c in configs.values()})==1,param
    state_diff_counts=Counter();identifier_diff_counts=Counter()
    for k in required:
        a=lookup['generic_all'][k]['state'];b=lookup['generic_ownership'][k]['state'];c=lookup['generic_neutral_ownership'][k]['state'];d=lookup['accessibility_neutral_ownership'][k]['state']
        ds=differences(a,b);state_diff_counts.update(ds)
        assert all(x in ['.alternatives[0].available','.alternatives[2].available'] for x in ds)
        assert c==d
        ids=differences(b,c);identifier_diff_counts.update(ids)
        assert all(x in ['.persona.persona_id','.trip.trip_id','.context.context_id','.trip.origin_type','.trip.destination_type'] for x in ids)
        assert 'Singapore' not in json.dumps(c) and 'Shanghai' not in json.dumps(c)
        assert c['trip']['trip_id']=='query' and c['context']['context_id']=='scenario'
        assert c['trip']['origin_type']==c['trip']['destination_type']=='unspecified'
        p=b['persona'];expected={'car':p['car_ownership'] and p['driving_license'],'bike':p['bike_ownership'],'pt':True,'walk':True}
        assert all(x['available']==expected[x['mode']] for x in b['alternatives'])
    arrays={};attempt_rows=[];missing=[]
    for name,path in SETS.items():
        records,partial,calls_snapshot_sha,calls_snapshot_bytes=snapshot_jsonl(path/'calls.jsonl')
        valid=defaultdict(list);failures=Counter();hash_mismatches=[];all_selected_calls=0
        for line,r in enumerate(records,1):
            kk=key(r)+(int(r['repeat']),)
            if kk not in targetkeys:continue
            all_selected_calls+=1
            row=lookup[name][kk[:3]]
            if r['state_sha256']!=row['state_sha256']:bad.append(dict(set=name,reason='state_hash_mismatch',line=line))
            # Verify logged request hashes from exact frozen prompt/state bytes.
            user=templates[name].replace('{state_json}',UniversalTravelerState.model_validate(row['state']).model_dump_json(indent=2))
            caps=[r['max_tokens']] if 'max_tokens' in r else configs[name]['retry_token_budgets']
            matchedcap=None
            for cap in caps:
                request=dict(model=configs[name]['model'],messages=[dict(role='system',content=prompts[name]),dict(role='user',content=user)],temperature=configs[name]['temperature'],max_tokens=cap)
                if object_sha(request)==r['request_sha256']:matchedcap=cap;break
            if matchedcap is None:hash_mismatches.append(line)
            attempt_rows.append(dict(set=name,city=r['city'],respondent_id=r['respondent_id'],task=r['task'],repeat=int(r['repeat']),source_line=line,
                attempt=r.get('attempt'),valid=bool(r.get('valid')),failure_type=r.get('failure_type',r.get('validation_reason','')),http_status=r.get('http_status'),
                finish_reason=r.get('finish_reason'),matched_token_cap=matchedcap,request_sha256=r['request_sha256']))
            if r.get('valid'):
                p=np.array([r['action']['mode_probabilities'].get(m,0.) for m in MODES])
                assert np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<2e-6
                assert all(p[i]==0 for i,a in enumerate(row['state']['alternatives']) if not a['available'])
                valid[kk].append((line,r,p))
            else:failures[r.get('failure_type',r.get('validation_reason','unknown'))]+=1
        duplicate_valid={k:v for k,v in valid.items() if len(v)>1}
        # Multiple successful logical targets are never silently replaced/averaged.
        if duplicate_valid:bad.append(dict(set=name,reason='multiple_valid_records_for_same_logical_repeat',n_targets=len(duplicate_valid)))
        if hash_mismatches:bad.append(dict(set=name,reason='request_hash_cannot_be_reconstructed',source_lines=hash_mismatches))
        absent=sorted(targetkeys-set(valid))
        missing.extend(dict(set=name,city=k[0],respondent_id=k[1],task=k[2],repeat=k[3]) for k in absent)
        complete_people=sum(all((city,rid,str(task),rep) in valid for task in task_orders[city] for rep in range(3)) for city,ids in selected.items() for rid in ids)
        audits[name]=dict(expected_states=40,expected_repeat_targets=120,valid_repeat_targets=len(valid),missing_repeat_targets=len(absent),complete_respondents=complete_people,
            source_attempts_in_pilot=all_selected_calls,failure_counts=dict(failures),duplicate_valid_targets=len(duplicate_valid),partial_final_line_snapshot=partial,
            request_hash_failures=len(hash_mismatches),all_call_rows_in_snapshot=len(records),source_calls_sha256=calls_snapshot_sha,source_calls_snapshot_bytes=calls_snapshot_bytes,
            returned_models=sorted({r['model'] for vs in valid.values() for _,r,_ in vs}),system_fingerprints=sorted({str(r.get('system_fingerprint','unavailable')) for vs in valid.values() for _,r,_ in vs}))
        if not absent and not duplicate_valid:
            arrays[name]=np.array([[valid[k+(rep,)][0][2] for rep in range(3)] for k in required])
    writej(DEST/'input_audit.json',dict(selected_respondents=selected,selection_rule='First two per city in original frozen selection order; no replacement, no human-label reads',
        n_respondents=4,n_tasks=40,n_repeats=3,state_differences_mask=dict(state_diff_counts),state_differences_unused_metadata_context=dict(identifier_diff_counts),prompt_pair_states_exactly_equal=True,user_templates_exactly_equal=True,
        numerical_arrays_equal_except_availability=True,prompt_only_difference='Generic teacher_v0.1 versus teacher_s8_accessibility_v0.1 adds PT semantics',
        city_hint='Mask pair holds original city-bearing identifiers and unused trip metadata fixed. Prompt pair uses exactly identical neutral payloads. Unused-metadata-context diagnostic changes only three identifier fields and origin/destination types; Student does not encode these five fields.',
        currency='No explicit money unit; raw SGD/CNY numeric encodings held identical. No exchange-rate or price correction introduced.',
        component_meaning='Same numerical PT components with exact total conservation; S8 system prompt explains feasibility and door-to-door accounting. Distinct context delay and final total retained.',
        common_model_config={k:configs['generic_all'][k] for k in ['model','temperature','repeats','thinking','retry_token_budgets']},sets=audits,validation_failures=bad,
        source_hashes=source_hashes,code_sha256=sha(__file__)))
    if attempt_rows:csvout(DEST/'pilot_attempt_audit.csv',attempt_rows)
    writej(DEST/'status.json',dict(ready=not missing and not bad,missing=missing,validation_failures=bad,comparative_statistics_written=not missing and not bad))
    if bad:
        print(json.dumps(dict(status='input_validation_failed',issues=bad),ensure_ascii=False));return 3
    if missing:
        print(json.dumps(dict(status='waiting_for_complete_preselected_pilot',missing_by_set={k:v['missing_repeat_targets'] for k,v in audits.items()},no_partial_comparison_computed=True)));return 2
    # No CI or hypothesis test: four selected people are only a descriptive pilot.
    means={name:a.mean(1) for name,a in arrays.items()};summary=[];contrasts=[];repeatrows=[]
    cityindex={city:np.array([k[0]==city for k in required]) for city in selected}
    cityindex['pooled_pilot']=np.ones(len(required),dtype=bool)
    for name,A in arrays.items():
        for scope,keep in cityindex.items():
            mean=means[name][keep];rep=A[keep]
            sd=rep.std(1,ddof=1)
            repeatrows.append(dict(condition=name,scope=scope,n_states=int(keep.sum()),pt_mean=float(mean[:,1].mean()),mean_raw_repeat_pt_sd=float(sd[:,1].mean()),
                mean_raw_repeat_mode_sd=float(sd.mean()),mean_repeat_probability_L1_from_state_mean=float(abs(rep-mean[:,None,:]).sum(2).mean())))
    for comparison,(left,right) in COMPARISONS.items():
        for scope,keep in cityindex.items():
            A=means[left][keep];B=means[right][keep];diff=B-A
            summary.append(dict(comparison=comparison,scope=scope,left=left,right=right,n_states=int(keep.sum()),n_respondents=4 if scope=='pooled_pilot' else 2,
                mean_probability_L1=float(abs(diff).sum(1).mean()),probability_L1_of_aggregate=float(abs(diff.mean(0)).sum()),
                pt_left=float(A[:,1].mean()),pt_right=float(B[:,1].mean()),pt_change=float(diff[:,1].mean()),mean_abs_state_pt_change=float(abs(diff[:,1]).mean()),
                hard_choice_disagreement=float((A.argmax(1)!=B.argmax(1)).mean())))
        for city,ids in selected.items():
            for contrast,sp in spec[city].items():
                modes=[MODES.index(m) for m in sp['modes']]
                conditions={}
                for condition in [left,right]:
                    values=[];repeat_values=[]
                    for rid in ids:
                        idx=[required.index((city,rid,str(task))) for task in sp['weights']];weights=np.array(list(sp['weights'].values()))
                        values.append(float((means[condition][idx][:,modes].sum(1)*weights).sum()))
                        repeat_values.append((arrays[condition][idx][:,:,modes].sum(2)*weights[:,None]).sum(0))
                    conditions[condition]=(np.array(values),np.array(repeat_values))
                l,lr=conditions[left];r,rr=conditions[right]
                contrasts.append(dict(comparison=comparison,city=city,contrast=contrast,modes='|'.join(sp['modes']),n_respondents=2,
                    left_response=float(l.mean()),right_response=float(r.mean()),response_change=float((r-l).mean()),
                    left_mean_respondent_repeat_response_sd=float(lr.std(1,ddof=1).mean()),right_mean_respondent_repeat_response_sd=float(rr.std(1,ddof=1).mean()),
                    interpretation='Three endpoint query repeats indexed into descriptive response replicates, not shared random seeds or independent human observations'))
    csvout(DEST/'pilot_probability_changes.csv',summary);csvout(DEST/'pilot_contrast_changes.csv',contrasts);csvout(DEST/'pilot_repeat_variability.csv',repeatrows)
    np.savez_compressed(DEST/'matched_pilot_probabilities.npz',**arrays,city=np.array([k[0] for k in required]),respondent=np.array([k[1] for k in required]),task=np.array([k[2] for k in required]),modes=np.array(MODES))
    lines=['# Matched prompt and mask diagnostic pilot','',
      'This analysis retains the first two originally selected respondents per city, all ten tasks and three successful logical repeats per task. No participant was replaced, no human answer was read, and failed attempts remain in the raw logs and pilot attempt ledger. This four-person pilot is descriptive and is not a city-representative estimate.','',
      'The mask comparison changes only original-support availability while retaining the generic prompt and the original metadata. The prompt comparison uses exactly the same neutral-metadata original-support state payloads and changes only from the generic prompt to the S8 accessibility prompt. An additional unused-metadata-context comparison neutralizes the three identifier fields and origin/destination types under fixed generic prompt and original support. Students do not encode those five fields. User prompt templates, model and temperature match. Token caps of accepted attempts are audited, and any successful retry at a different cap remains a recorded operational qualification.','',
      '| Comparison | Scope | Mean state probability L1 | PT probability change, pp |',
      '|---|---|---:|---:|']
    for r in summary:lines.append(f"| {r['comparison']} | {r['scope']} | {r['mean_probability_L1']:.4f} | {100*r['pt_change']:+.2f} |")
    lines+=['','The mask effect measures a change of the modeled choice set; it does not validate newly enabled private-car/bicycle utilities. The prompt effect measures sensitivity to explicit PT feasibility and time-component instructions under fixed support. It is not a re-estimate of human behavior or proof that one prompt is more accurate.','',
      'PT component values sum to the same total in each comparison. Currency is not explicitly stated. City-bearing identifiers and unused trip metadata are held fixed in the mask comparison, neutralized before the prompt comparison, and jointly isolated in the unused-metadata-context comparison. These comparisons do not isolate monetary-unit interpretation or the driving/taxi composite, which require separately declared semantic experiments.','',
      'See pilot_repeat_variability.csv for raw repeat SD and L1 around each state mean, pilot_contrast_changes.csv for all five Singapore/eight Shanghai contrasts, and input_audit.json for exact payload/prompt checks and source hashes. No bootstrap CI is presented for four selected respondents.']
    (DEST/'PROMPT_MASK_PILOT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    writej(DEST/'completed_manifest.json',dict(files={str(p.relative_to(DEST)):sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='completed_manifest.json'},interpretation='Descriptive preselected four-person diagnostic; all four conditions complete, no human outcome tuning'))
    print(json.dumps(dict(status='complete',n_people=4,n_states=40,n_repeat_targets_per_condition=120,comparisons=summary),ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
