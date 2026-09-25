"""Recompute persona statistics and audit immutable acquisition evidence.

This is an audit_recompute task. It never selects checkpoints, alters source
responses, imputes missing calls, or declares a paper/Goal acceptance result.
"""
from pathlib import Path
import collections,csv,datetime,itertools,json,platform,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/revision_20260925'))
from synthetic_teacher import DEST,MODEL,ENDPOINT,CARDS,BUNDLE,S8_SYSTEM_PROMPT,build_s8_user_prompt
from controlled import read_json,write_json,write_rows,file_hash,digest
from offline_audits import rows
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.teacher.parser import TeacherResponseParser
from traveler_distillation.teacher.validator import TeacherResponseValidator

CODE='cvpr_workspace/analysis/statistics/synthetic_audit.py'
PROTOCOL='docs/SYNTHETIC_ANALYSIS_PROTOCOL.md'

def interval(x):
    x=np.asarray(x,float);assert x.ndim==1 and len(x)>1 and np.isfinite(x).all()
    b=x[np.random.default_rng(20260925).integers(len(x),size=(10000,len(x)))].mean(1)
    return dict(n_personas=len(x),mean=float(x.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)))

def csvout(path,data):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def usage_report(records):
    known=[];unknown=[];input_tokens=output_tokens=cached_tokens=0
    priced=0.;peak_upper=0.;off_lower=0.
    for rid,r in records:
        u=r.get('usage')
        if not isinstance(u,dict) or 'prompt_tokens' not in u or 'completion_tokens' not in u:
            unknown.append(rid);continue
        pi=u['prompt_tokens'];po=u['completion_tokens'];ca=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)
        assert 0<=ca<=pi and po>=0
        input_tokens+=pi;output_tokens+=po;cached_tokens+=ca;known.append(rid)
        t=datetime.datetime.fromisoformat(r['utc']);peak=t.weekday()<5 and (1<=t.hour<4 or 6<=t.hour<10)
        base=((pi-ca)*.15+ca*.003+po*.60)/1e6
        priced+=base*(2 if peak else 1);off_lower+=base;peak_upper+=(pi*.30+po*1.20)/1e6
    return dict(total_attempts=len(records),usage_reported_attempts=len(known),usage_unknown_attempts=len(unknown),unknown_attempt_ids=unknown,
        input_tokens=input_tokens,output_tokens=output_tokens,cached_input_tokens=cached_tokens,total_tokens=input_tokens+output_tokens,
        request_time_rate_quota_estimate_usd=priced,known_usage_all_offpeak_cache_estimate_usd=off_lower,known_usage_peak_no_cache_usd=peak_upper,
        not_invoice=True,unknown_calls='Unreported usage is unknown, not zero. Bounds cover reported usage only.',
        pricing_time='Request UTC; provider billing timestamp unavailable. Boundary-crossing requests may be charged differently.',
        rates_per_million=dict(offpeak_input=.15,offpeak_output=.60,offpeak_cache=.003,peak_input=.30,peak_output=1.20,peak_cache=.006),
        source='https://opencode.ai/docs/go/',rate_checked_date='2026-09-25',subscription_usd_per_month=10,
        account_remaining_quota='not verified',reasoning_tokens='Already included in completion_tokens; not added again')

def audit(cohort):
    folder=DEST/cohort;status=read_json(folder/'status.json');assert status['complete']
    protocol=read_json(DEST/'acquisition_protocol.json')
    assert file_hash(DEST/'states.jsonl')==protocol['manifest_sha256']
    assert digest(S8_SYSTEM_PROMPT)==protocol['prompt_sha256']
    allstates=rows(DEST/'states.jsonl');stateby={s['id']:s for s in allstates}
    records=[(f'{cohort}/{p.stem}',read_json(p)) for p in sorted(folder.glob('attempt_*.json'))]
    included=[];excluded=[];calls={};completion_ids=set();hashes={}
    for rid,r in records:
        p=DEST/(rid+'.json');hashes[p.relative_to(ROOT).as_posix()]=file_hash(p)
        s=stateby[r['state_id']];assert s['cohort']==cohort and s['state_hash']==digest(s['state'])
        assert r['endpoint']==ENDPOINT and r['requested_model']==MODEL and r['repeat'] in range(3)
        body=dict(model=MODEL,temperature=.2,max_tokens=8192,messages=[dict(role='system',content=S8_SYSTEM_PROMPT),dict(role='user',content=build_s8_user_prompt(json.dumps(s['state'],ensure_ascii=False)))])
        assert digest(body)==r['request_hash']
        if not r.get('valid'):
            excluded.append(dict(run_id=rid,reason=r.get('validation_reason') or r.get('error_type') or f'HTTP {r.get("http_status")}',evidence_refs=[p.relative_to(ROOT).as_posix()]));continue
        assert r['http_status']==200 and r['returned_model']==MODEL
        assert r['completion_id'] and r['completion_id'] not in completion_ids;completion_ids.add(r['completion_id'])
        key=(s['id'],r['repeat']);assert key not in calls;calls[key]=r
        parsed=TeacherResponseParser().parse(r['response']['choices'][0]['message']['content'])
        assert parsed.model_dump(mode='json')==r['action']
        assert TeacherResponseValidator().validate(UniversalTravelerState.model_validate(s['state']),parsed).valid
        included.append(rid)
    personas=sorted({stateby[s]['persona'] for s,k in calls});states=[s for s in allstates if s['persona'] in personas]
    assert len(calls)==status['valid']==status['required']==len(personas)*36
    assert all((s['id'],k) in calls for s in states for k in range(3))
    profiles={digest({k:v for k,v in s['state']['persona'].items() if k!='persona_id'}) for s in states}
    prior={digest({k:v for k,v in e['state']['persona'].items() if k!='persona_id'}) for split in ['train','val','test'] for e in rows(BUNDLE/f'{split}_endpoints.jsonl')}
    assert len(profiles)==len(personas) and not profiles&prior
    pilot_profiles={digest({k:v for k,v in s['state']['persona'].items() if k!='persona_id'}) for s in allstates if s['cohort']=='pilot'}
    if cohort=='formal':assert not profiles&pilot_profiles
    for s in states:
        pt=next(a for a in s['state']['alternatives'] if a['mode']=='pt')
        if pt['pt_feasible']:
            components=sum(pt[k] for k in ['access_time_min','egress_time_min','wait_time_min','in_vehicle_time_min','transfer_time_min'])
            assert abs(components-pt['travel_time_min'])<1e-6
    source=folder/'acquisition_snapshot.json';write_json(source,hashes)
    result=dict(valid=len(calls),personas=len(personas),all_checks_passed=True,checks=['manifest and system prompt hashes','request body reconstruction','unique completions and state-repeat slots','returned model','strict parser and validator recomputation','complete persona groups','novel unique persona profiles','formal-pilot disjointness','feasible PT time decomposition'],source_snapshot_sha256=file_hash(source),usage=usage_report(records))
    write_json(folder/'acquisition_audit.json',result)
    return records,included,excluded,calls,personas

def summarize(cohort,calls,personas):
    folder=DEST/cohort;pr=rows(folder/'predictions.jsonl');pairs=rows(folder/'pair_metrics.jsonl');analysis=read_json(folder/'analysis.json')
    configs=collections.defaultdict(dict)
    for r in pr:configs[r['family'],r['variant'],r['selection'],r['seed']][r['id']]=r
    units=[]
    for (family,variant,selection,seed),by in configs.items():
        assert len(by)==len(personas)*12
        for p in personas:
            b=by[p+':baseline']
            for card in CARDS:
                r=by[p+':'+card];a=[calls[r['id'],k]['action'] for k in range(3)]
                t=np.mean([[v['mode_probabilities'].get(m,0.) for m in ['car','pt','bike','walk']] for v in a],axis=0)
                assert np.allclose(t,r['teacher'],atol=1e-12)
                mask=np.array(r['mask'])|np.array(b['mask']);dt=t-np.array(b['teacher']);ds=np.array(r['student'])-b['student']
                metrics={k:r[k] for k in ['kl','probability_l1','departure_mae']}
                if card!='baseline':metrics['response_gap']=float(abs(dt-ds)[mask].mean())
                if card=='pt_infeasible':metrics.update(infeasible_pt_mass=r['student'][1],top_choice_violation=float(np.argmax(r['student'])==1))
                for metric,value in metrics.items():units.append(dict(family=family,variant=variant,selection=selection,seed=seed,persona=p,card=card,metric=metric,value=value))
            for combo,c1,c2 in [('fare_delay','fare_2_5','delay_45'),('fare_access','fare_2_5','access_20')]:
                rr=[by[p+':'+c] for c in [combo,c1,c2,'baseline']];mask=np.any([r['mask'] for r in rr],axis=0)
                gap=sum(w*(np.array(r['teacher'])-r['student']) for w,r in zip([1,-1,-1,1],rr))
                units.append(dict(family=family,variant=variant,selection=selection,seed=seed,persona=p,card=combo,metric='interaction_gap',value=float(abs(gap)[mask].mean())))
    lookup={(r['family'],r['variant'],r['selection'],r['seed'],r['persona'],r['card']):r['response_gap'] for r in pairs}
    assert all(abs(r['value']-lookup[r['family'],r['variant'],r['selection'],r['seed'],r['persona'],r['card']])<1e-12 for r in units if r['metric']=='response_gap')
    # Average seed replicas within persona before estimating uncertainty.
    averaged=collections.defaultdict(list)
    for r in units:averaged[tuple(r[k] for k in ['family','variant','selection','persona','card','metric'])].append(r['value'])
    av={k:float(np.mean(v)) for k,v in averaged.items()}
    overall=collections.defaultdict(list)
    for (f,v,s,p,c,m),value in av.items():
        if m in ['kl','probability_l1','departure_mae','response_gap']:overall[f,v,s,p,'ALL',m].append(value)
    av.update({k:float(np.mean(v)) for k,v in overall.items()})
    groups=collections.defaultdict(list)
    for (f,v,s,p,c,m),value in av.items():groups[f,v,s,c,m].append(value)
    tables=[dict(family=f,variant=v,selection=s,card=c,metric=m,**interval(vals)) for (f,v,s,c,m),vals in sorted(groups.items())]
    contrasts=[]
    for f in ['full','delay_holdout']:
        for s in ['static','response']:
            for c in ['ALL']+CARDS[1:]:
                diffs=[av[f,'signed_l1',s,p,c,'response_gap']-av[f,'soft_kl',s,p,c,'response_gap'] for p in personas]
                primary=f=='full' and s=='static' and c=='ALL'
                contrasts.append(dict(family=f,selection=s,card=c,primary=primary,**interval(diffs)))
                if primary:
                    assert np.allclose(diffs,[r['signed_minus_softkl'] for r in analysis['primary']],atol=1e-12)
                    assert abs(np.mean(diffs)-analysis['mean'])<1e-12
    noise=[]
    stateby={r['id']:r for r in rows(DEST/'states.jsonl')}
    for p in personas:
        for card in CARDS[1:]:
            base=p+':baseline';cur=p+':'+card
            mask=np.array([a['available'] for a in stateby[cur]['state']['alternatives']])|np.array([a['available'] for a in stateby[base]['state']['alternatives']])
            deltas=[]
            for k in range(3):
                d=[calls[cur,k]['action']['mode_probabilities'].get(m,0.)-calls[base,k]['action']['mode_probabilities'].get(m,0.) for m in ['car','pt','bike','walk']];deltas.append(np.array(d))
            gap=np.mean([abs(deltas[i]-deltas[j])[mask].mean() for i,j in itertools.combinations(range(3),2)])
            noise.append(dict(persona=p,card=card,self_response_gap=float(gap)))
    write_rows(folder/'descriptive_units.jsonl',units);write_json(folder/'descriptive_summary.json',tables);csvout(folder/'descriptive_summary.csv',tables)
    write_json(folder/'paired_contrasts.json',contrasts);csvout(folder/'paired_contrasts.csv',contrasts);write_rows(folder/'repeat_self_disagreement.jsonl',noise)
    return tables,contrasts

def manifest(cohort,records,included,excluded):
    base=f'outputs/revision_20260925/synthetic/{cohort}';folder=ROOT/base
    mids=['response_gap','kl','probability_l1','departure_mae','infeasible_pt_mass','top_choice_violation','interaction_gap']
    metrics=[dict(id=m,name=m,calculation='See deterministic per-unit implementation and frozen analysis protocol',unit='minutes' if m=='departure_mae' else 'nats' if m=='kl' else 'probability',direction='lower',aggregation='Within persona, average neural seeds; equally weighted cards for overall endpoint/response metrics; then equal-weight personas',source_refs=[base+'/predictions.jsonl',CODE]) for m in mids]
    outputs=[dict(id=n,kind='result_table',path=base+'/'+n+'.json',format='json',source_metric_ids=mids if n=='descriptive_summary' else ['response_gap'],evidence_refs=[base+'/descriptive_units.jsonl']) for n in ['descriptive_summary','paired_contrasts']]
    outputs.append(dict(id='audit',kind='statistics_report',path=base+'/acquisition_audit.json',format='json',source_metric_ids=mids,evidence_refs=[base+'/acquisition_snapshot.json']))
    write_json(folder/'environment.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),analysis_sha256=file_hash(ROOT/CODE)))
    write_json(folder/'statistics.yaml',dict(schema_version='1.0',analysis_id='synthetic_reference_'+cohort,version=1,status='accepted',mode='audit_recompute',purpose='Recompute complete-cohort persona statistics and acquisition integrity; not a paper acceptance decision',source_snapshot_ref=base+'/acquisition_snapshot.json',input_run_ids=[r for r,_ in records],included_run_ids=included,excluded_runs=excluded,
        experimental_unit=dict(name='synthetic persona',definition='Complete 12-state x 3-response group; neural seed replicas and shared baseline remain within person',independence_basis='Distinct generated attribute profiles, conditional on one trip and one generative population; no claim of human independence'),grouping_factors=['training family','objective','checkpoint selection','intervention card'],metrics=metrics,
        statistical_methods=[dict(id='paired_persona_bootstrap',applies_to_metric_ids=mids,method='10000 equal-probability persona resamples with replacement, seed 20260925; percentile 95% interval',rationale='Keeps correlated cards, repeated calls and neural seeds within the sampled persona',assumptions=['Generated personas are the sampling units within the fixed synthetic design','Inference is conditional on frozen checkpoints, trip and reference provider'],uncertainty='Persona-level interval conditional on this three-repeat target estimate; does not separately integrate API resampling uncertainty',effect_size='Absolute means and signed-minus-soft probability differences',multiplicity_control='One primary formal comparison; all other intervals pointwise descriptive, no familywise significance claim',implementation_ref=CODE)],
        data_integrity=dict(raw_immutable=True,missing_policy='Require all planned states and three valid replies per state; no imputation',outlier_policy='No outcome-based removal',selection_policy='Frozen historical validation selectors; pilot excluded from formal',checks=read_json(folder/'acquisition_audit.json')['checks'],evidence_refs=[base+'/acquisition_audit.json']),outputs=outputs,
        provenance=dict(analysis_code_refs=[CODE,'scripts/revision_20260925/synthetic_analysis.py'],config_refs=[PROTOCOL,'outputs/revision_20260925/synthetic/acquisition_protocol.json'],environment_refs=[base+'/environment.json'],generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),evidence_refs=[base+'/acquisition_snapshot.json']),supersedes=None))

if __name__=='__main__':
    cohort=sys.argv[1];records,included,excluded,calls,personas=audit(cohort)
    summarize(cohort,calls,personas);manifest(cohort,records,included,excluded)
    print(json.dumps(dict(cohort=cohort,personas=len(personas),valid=len(included),excluded=len(excluded),audit='passed')))
