"""Bounded Go acquisition; credentials only from stdin, never serialized.

Research client identity is truthful. No coding-client impersonation or automatic
paid fallback. Only complete persona groups enter analysis.
"""
from pathlib import Path
import argparse,concurrent.futures,copy,datetime,json,math,sys,threading,time
from controlled import ROOT,OUT,BUNDLE,read_json,write_json,write_rows,file_hash,digest
from offline_audits import rows
import numpy as np
import httpx
from traveler_distillation.generators import PersonaGenerator
from traveler_distillation.config import load_yaml
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.teacher.prompts_s8 import S8_SYSTEM_PROMPT,build_s8_user_prompt
from traveler_distillation.teacher.parser import TeacherResponseParser
from traveler_distillation.teacher.validator import TeacherResponseValidator
DEST=OUT.parent/'synthetic'
MODEL='deepseek-v4.1-flash'
ENDPOINT='https://opencode.ai/zen/go/v1/chat/completions'
CARDS=['baseline','delay_8','delay_45','fare_1_25','fare_2_5','access_5','access_20','fare_delay','fare_access','pt_infeasible','weather_0_25','weather_0_9']

def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'states.jsonl').exists():return
    training=rows(BUNDLE/'train_endpoints.jsonl')
    base=copy.deepcopy(next(e['state'] for e in training if e['source']=='accessibility' and e['accessibility_class']=='A_excellent'))
    prior={digest({k:v for k,v in e['state']['persona'].items() if k!='persona_id'}) for split in ['train','val','test'] for e in rows(BUNDLE/f'{split}_endpoints.jsonl')}
    personas=[]
    for p in PersonaGenerator(seed=20260925,config=load_yaml(ROOT/'configs/generation_v0_1.yaml')).generate(2000):
        sig=digest({k:v for k,v in p.model_dump(mode='json').items() if k!='persona_id'})
        if sig not in prior:personas.append(p);prior.add(sig)
        if len(personas)==132:break
    assert len(personas)==132
    result=[]
    for i,p in enumerate(personas):
        cohort='pilot' if i<12 else 'formal';pid=f'NEW_{cohort}_{i+1:03d}'
        for card in CARDS:
            st=copy.deepcopy(base);st['persona']=p.model_dump(mode='json');st['persona']['persona_id']=pid;st['trip']['trip_id']='fixed_numeric_trip';st['context']['context_id']='numeric_scenario'
            by={a['mode']:a for a in st['alternatives']};by['car']['available']=p.car_ownership and p.driving_license;by['bike']['available']=p.bike_ownership
            pt=by['pt'];delay=8. if card=='delay_8' else 45. if card in ['delay_45','fare_delay'] else 0.
            fare=1.25 if card=='fare_1_25' else 2.5 if card in ['fare_2_5','fare_delay','fare_access'] else 1.
            access=5. if card=='access_5' else 20. if card in ['access_20','fare_access'] else 0.
            if delay:
                st['context']['transit_delay_min']=delay;pt['wait_time_min']+=delay;pt['travel_time_min']+=delay
            st['context']['fare_multiplier']=fare;pt['monetary_cost']*=fare
            if access:pt['access_time_min']+=access;pt['travel_time_min']+=access
            if card=='pt_infeasible':
                pt['pt_feasible']=0.;pt['coverage_ratio']=0.;pt['travel_time_min']=999.
                for k in ['access_time_min','egress_time_min','wait_time_min','in_vehicle_time_min','transfer_time_min']:pt[k]=0.
                pt['transfers']=0
            if card.startswith('weather_'):
                w=.25 if card=='weather_0_25' else .9;st['context']['weather']={'condition':'rain','intensity':w}
                for alt in st['alternatives']:alt['travel_time_min']*=1+w*.15*alt['weather_exposure']
                # Preserve the prompt's PT door-to-door decomposition exactly.
                for k in ['access_time_min','egress_time_min','wait_time_min','in_vehicle_time_min','transfer_time_min']:pt[k]*=1+w*.15*pt['weather_exposure']
            st=UniversalTravelerState.model_validate(st).model_dump(mode='json')
            result.append(dict(id=f'{pid}:{card}',persona=pid,cohort=cohort,card=card,state=st,state_hash=digest(st)))
    write_rows(DEST/'states.jsonl',result)
    write_json(DEST/'acquisition_protocol.json',dict(model=MODEL,endpoint=ENDPOINT,temperature=.2,max_tokens=8192,thinking='provider default; no override',prompt_sha256=digest(S8_SYSTEM_PROMPT),manifest_sha256=file_hash(DEST/'states.jsonl'),workers=8,repeats=3,pilot_attempt_cap=476,pilot_peak_no_cache_quota_cap_usd=3.,formal_peak_no_cache_quota_cap_usd=8.,stopping='stop on 401/402/403, do not retry access or quota denial; at most two attempts per unit; no paid fallback configured',primary_precision_halfwidth=.01,formal_n_rule='max(30, ceil((1.96 * persona SD of prespecified signed-minus-softKL contrast / .01)^2)); if >120 report precision budget gap instead of claiming target achieved',setting='132 novel persona attribute combinations, a single fixed numeric trip and feasible training-input accessibility profile; no geographic OOD claim',families='delay, fare, access burden, fare-delay, fare-access, service infeasibility, rain; distinct intensity cards; no claim all families unseen',delay_interpretation='extra wait, counted once in total travel time',source='synthetic only; no human questionnaire or respondent data sent'))
    print('Prepared 132 personas x 12 states; manifest frozen',flush=True)

def cost(usage):return (usage.get('prompt_tokens',0)*.30+usage.get('completion_tokens',0)*1.20)/1e6

def acquire(cohort,n,preflight=False):
    key=sys.stdin.readline().strip()
    if not key:raise RuntimeError('Credential required on stdin')
    states=[r for r in rows(DEST/'states.jsonl') if r['cohort']==cohort];ids=sorted({r['persona'] for r in states})[:n];states=[r for r in states if r['persona'] in ids]
    dest=DEST/cohort;dest.mkdir(exist_ok=True);existing=list(dest.glob('*.json'));attempts=[read_json(p) for p in existing if p.name.startswith('attempt_')];done={(r['state_id'],r['repeat']) for r in attempts if r.get('valid')}
    jobs=[(s,k) for s in states for k in range(3) if (s['id'],k) not in done]
    if preflight:jobs=jobs[:1]
    # Amendment v2: transport recovery only, frozen model/prompt/states unchanged.
    # Prior attempt records and first valid replies are never replaced.
    cap=len(states)*3*3;budget=3. if cohort=='pilot' else 8.;spent=sum(cost(r.get('usage',{})) for r in attempts);reserved=0.;counter=max([r['attempt'] for r in attempts],default=0);lock=threading.Lock();stop=threading.Event();started=time.monotonic()
    counts={job:sum((r['state_id'],r['repeat'])==job for r in attempts) for job in [(s['id'],k) for s,k in jobs]}
    source_sha=file_hash(__file__);session_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(dest/f'launch_after_{counter:05d}.json',dict(utc=session_utc,driver_sha256=source_sha,manifest_sha256=file_hash(DEST/'states.jsonl'),protocol_version='transport_recovery_v2',attempt_cap=cap,max_attempts_per_slot=8,known_usage_peak_no_cache_cap_usd=budget,prior_attempts=len(attempts),pending=len(jobs),workers=1 if preflight else 8,unknown_usage='Transport failures without provider usage are unknown and are outside the known-usage cost total; attempt caps additionally limit exposure'))
    headers={'Authorization':'Bearer '+key,'User-Agent':'traveler-response-research/1.0','x-opencode-session':'ait-response-revision-20260925-'+cohort}
    client=httpx.Client(timeout=httpx.Timeout(180,connect=30),trust_env=False,limits=httpx.Limits(max_connections=8,max_keepalive_connections=8,keepalive_expiry=60))
    def one(job):
        nonlocal spent,reserved,counter
        state,k=job;body={'model':MODEL,'temperature':.2,'max_tokens':8192,'messages':[{'role':'system','content':S8_SYSTEM_PROMPT},{'role':'user','content':build_s8_user_prompt(json.dumps(state['state'],ensure_ascii=False))}]}
        # UTF-8 bytes upper-bound input tokens conservatively for reservation.
        envelope=(len(json.dumps(body).encode())*.30+8192*1.20)/1e6
        for retry in range(max(0,8-counts[state['id'],k])):
            if retry:time.sleep(min(2**retry,16))
            with lock:
                if stop.is_set() or (dest/'STOP_REQUESTED').exists() or counter>=cap or spent+reserved+envelope>budget:stop.set();return
                counter+=1;idx=counter;reserved+=envelope
            call_start=time.monotonic()
            record=dict(state_id=state['id'],repeat=k,attempt=idx,requested_model=MODEL,endpoint=ENDPOINT,request_hash=digest(body),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),valid=False,driver_sha256=source_sha,protocol_version='transport_recovery_v2')
            try:
                r=client.post(ENDPOINT,headers=headers,json=body)
                record['http_status']=r.status_code
                if r.status_code!=200:
                    # No echoed request headers or secret-bearing exception strings.
                    try:record['provider_error']=r.json().get('error',{})
                    except Exception:record['provider_error']='non-JSON error'
                    if 400<=r.status_code<500:stop.set()
                else:
                    data=r.json();record.update(usage=data.get('usage',{}),completion_id=data.get('id'),returned_model=data.get('model'),system_fingerprint=data.get('system_fingerprint'),response=data)
                    content=data['choices'][0]['message'].get('content') or ''
                    action=TeacherResponseParser().parse(content)
                    vr=TeacherResponseValidator().validate(UniversalTravelerState.model_validate(state['state']),action)
                    record.update(valid=vr.valid,validation_reason=vr.reason,action=action.model_dump(mode='json'))
            except Exception as e:
                record['error_type']=type(e).__name__
                record['error_chain_types']=[]
                cause=e.__cause__
                while cause is not None and len(record['error_chain_types'])<5:
                    record['error_chain_types'].append(type(cause).__name__);cause=cause.__cause__
            record['elapsed_s']=time.monotonic()-call_start
            record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            with lock:
                reserved-=envelope;spent+=cost(record.get('usage',{}));write_json(dest/f'attempt_{idx:05d}.json',record)
                if idx%12==0 or preflight or record.get('http_status') in [401,402,403,429]:print(json.dumps(dict(attempts=idx,quota_peak_no_cache_usd=round(spent,4),last_status=record.get('http_status'),valid=record['valid'],elapsed_s=round(time.monotonic()-started))),flush=True)
            if record['valid'] or stop.is_set():return
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1 if preflight else 8) as pool:list(pool.map(one,jobs))
    finally:client.close()
    records=[read_json(p) for p in dest.glob('attempt_*.json')];valid={(r['state_id'],r['repeat']) for r in records if r.get('valid')}
    write_json(dest/'status.json',dict(required=len(states)*3,valid=len(valid),attempts=len(records),complete=len(valid)==len(states)*3,quota_peak_no_cache_usd=sum(cost(r.get('usage',{})) for r in records),input_tokens=sum(r.get('usage',{}).get('prompt_tokens',0) for r in records),output_tokens=sum(r.get('usage',{}).get('completion_tokens',0) for r in records),returned_models=sorted({str(r.get('returned_model')) for r in records if r.get('valid')})))
    print(json.dumps(read_json(dest/'status.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','preflight','pilot','formal']);p.add_argument('--n',type=int,default=30);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:acquire('formal' if a.command=='formal' else 'pilot',a.n if a.command=='formal' else 12,a.command=='preflight')
