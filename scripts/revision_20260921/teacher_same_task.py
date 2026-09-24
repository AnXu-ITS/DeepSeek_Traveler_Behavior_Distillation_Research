"""Frozen, answer-blind same-task DeepSeek evaluation; resume without replacing calls.

Run with the repository .venv. Credentials are read only from DEEPSEEK_API_KEY
or the existing local .env, never serialized. Selection never reads human labels.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import httpx
import numpy as np
from traveler_distillation.config import load_dotenv
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.teacher.prompts import SYSTEM_PROMPT, build_user_prompt
from traveler_distillation.teacher.prompts_s8 import S8_SYSTEM_PROMPT, TEACHER_S8_PROMPT_VERSION
from traveler_distillation.teacher.parser import TeacherResponseParser
from traveler_distillation.teacher.validator import TeacherResponseValidator

OUT = ROOT / 'outputs/revision_20260921/teacher'
CFG = {'model': 'deepseek-v4-pro', 'temperature': 0.2, 'max_tokens': 8192,
       'timeout_seconds': 300, 'workers': 48, 'repeats': 3,
       'retry_token_budgets': [8192, 16384, 32768],
       'per_city': 24, 'selection_seed': 20260921, 'bootstrap_seed': 20260922,
       'bootstrap_draws': 10000, 'max_attempts': 3,
       'profile': 'survey_options', 'prompt_version': 'teacher_v0.1',
       'thinking': 'API default, no override; historical request fields retained',
       'pricing_source': 'https://api-docs.deepseek.com/quick_start/pricing/',
       'pricing_checked': '2026-09-21',
       'usd_per_million_peak': {'input_miss': 1.32, 'input_hit': 0.044, 'output': 3.96},
       'selection': 'age x car ownership round-robin strata, seeded random order within strata; full 10-task groups; no outcome access'}

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def credentials():
    load_dotenv(ROOT / '.env')
    key = os.environ.get('DEEPSEEK_API_KEY')
    base = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1').rstrip('/')
    if not key:
        raise RuntimeError('DEEPSEEK_API_KEY unavailable')
    if base not in {'https://api.deepseek.com', 'https://api.deepseek.com/v1'}:
        raise RuntimeError('Expected project official DeepSeek endpoint; endpoint change requires review')
    return base, key

def preflight():
    base, key = credentials()
    rec = {'utc': datetime.now(timezone.utc).isoformat(), 'endpoint': base,
           'requested_model': CFG['model']}
    try:
        with httpx.Client(timeout=40, trust_env=False) as c:
            r = c.get(base + '/models', headers={'Authorization': 'Bearer ' + key})
        rec['http_status'] = r.status_code
        if r.status_code == 200:
            rec['model_ids'] = [x.get('id') for x in r.json().get('data', [])]
            rec['requested_model_available'] = CFG['model'] in rec['model_ids']
        else:
            rec['error_code'] = r.json().get('error', {}).get('code', 'http_error')
    except Exception as e:
        rec['error_type'] = type(e).__name__
    dump(OUT / 'api_preflight.json', rec)
    print(json.dumps(rec), flush=True)

def prepare():
    states_path = ROOT / 'outputs/revision_20260921/survey/task_states.jsonl'
    rows = [r for r in read_jsonl(states_path) if r.get('profile') == 'survey_options']
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r['city']][str(r['respondent_id'])].append(r)
    rng = np.random.default_rng(CFG['selection_seed'])
    selected, selection = [], {}
    for city in sorted(by):
        strata = defaultdict(list)
        for rid, tasks in sorted(by[city].items()):
            if len(tasks) != 10:
                raise ValueError(f'Expected complete 10-task group: {city}')
            p = tasks[0]['state']['persona']
            strata[(p['age_group'], p['car_ownership'])].append(rid)
        keys = sorted(strata)
        for ids in strata.values():
            rng.shuffle(ids)
        ids = []
        while len(ids) < min(CFG['per_city'], len(by[city])):
            for s in keys:
                if strata[s] and len(ids) < CFG['per_city']:
                    ids.append(strata[s].pop())
        selection[city] = ids
        for index, rid in enumerate(ids):
            for row in sorted(by[city][rid], key=lambda r: str(r['task'])):
                state = UniversalTravelerState.model_validate(row['state'])
                if CFG['profile'] == 'ownership':
                    for alt in state.alternatives:
                        alt.available = (state.persona.car_ownership and state.persona.driving_license) if alt.mode == 'car' else state.persona.bike_ownership if alt.mode == 'bike' else True
                # The model's numerical input does not consume this opaque identifier.
                state.persona.persona_id = f'participant_{index+1:03d}' if CFG.get('neutral_identifiers') else f'{city}_participant_{index+1:03d}'
                if hasattr(state.trip, 'trip_id'):
                    state.trip.trip_id = f'task_{row["task"]}'
                if CFG.get('neutral_identifiers'):
                    # These five fields are not encoded by any evaluated Student.
                    # Hold them neutral before the first call, rather than leaking
                    # city/task names or giving the Teacher extra place semantics.
                    state.trip.trip_id = 'query'
                    state.context.context_id = 'scenario'
                    state.trip.origin_type = 'unspecified'
                    state.trip.destination_type = 'unspecified'
                payload = state.model_dump(mode='json')
                selected.append({'city': city, 'respondent_id': rid, 'task': row['task'],
                                 'state_id': row['state_id'], 'selection_order': index,
                                 'state': payload, 'state_sha256': sha(payload)})
    selected_path = OUT / 'selected_states.jsonl'
    text = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in selected)
    if selected_path.exists() and selected_path.read_text(encoding='utf-8') != text:
        raise RuntimeError('Refusing to replace frozen Teacher selection')
    selected_path.write_text(text, encoding='utf-8')
    dump(OUT / 'config.json', CFG)
    dump(OUT / 'selection.json', {'respondents': selection,
         'state_count': len(selected), 'planned_calls': len(selected)*CFG['repeats'],
         'source_sha256': hashlib.sha256(states_path.read_bytes()).hexdigest(),
         'outcomes_read_during_selection': False, 'selection_config': CFG['selection']})
    (OUT / 'system_prompt.txt').write_text(SYSTEM_PROMPT, encoding='utf-8')
    (OUT / 'user_prompt_template.txt').write_text(build_user_prompt('{state_json}'), encoding='utf-8')
    print(json.dumps({'states': len(selected), 'calls':len(selected)*CFG['repeats']}), flush=True)

LOCK = threading.Lock()
def append(path, rec):
    with LOCK, path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + '\n')

def call_one(row, repeat, base, key):
    state = UniversalTravelerState.model_validate(row['state'])
    system, user = SYSTEM_PROMPT, build_user_prompt(state.model_dump_json(indent=2))
    payload = {'model': CFG['model'], 'messages': [{'role':'system','content':system},
                {'role':'user','content':user}], 'temperature':CFG['temperature'],
                'max_tokens':CFG['max_tokens']}
    parser, validator = TeacherResponseParser(), TeacherResponseValidator()
    rec_base = {k: row[k] for k in ('city','respondent_id','task','state_id','state_sha256')}
    rec_base.update(repeat=repeat)
    key_slot = 'primary'
    for attempt in range(CFG['max_attempts']):
        payload['max_tokens'] = CFG['retry_token_budgets'][attempt]
        rec = dict(rec_base, attempt=attempt, utc=datetime.now(timezone.utc).isoformat(),
                   request_sha256=sha(payload), max_tokens=payload['max_tokens'], credential_slot=key_slot,
                   prompt_version=CFG['prompt_version'])
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=CFG['timeout_seconds'], trust_env=False) as c:
                r = c.post(base + '/chat/completions', json=payload,
                           headers={'Authorization': 'Bearer '+key})
            rec['http_status'] = r.status_code
            if r.status_code != 200:
                rec['valid'] = False
                rec['failure_type'] = 'http_' + str(r.status_code)
            else:
                data = r.json()
                choice = data['choices'][0]
                raw = choice.get('message', {}).get('content') or ''
                rec.update(raw_content=raw, usage=data.get('usage'),
                           model=data.get('model'), completion_id=data.get('id'),
                           created=data.get('created'), finish_reason=choice.get('finish_reason'),
                           system_fingerprint=data.get('system_fingerprint'))
                action = parser.parse(raw)
                valid = validator.validate(state, action)
                rec.update(valid=valid.valid, validation_reason=valid.reason,
                           action=action.model_dump(mode='json'))
        except Exception as e:
            # Exception text can contain request details; save only a safe class.
            rec.update(valid=False, failure_type=type(e).__name__)
        rec['elapsed_seconds'] = time.perf_counter() - start
        append(OUT / 'calls.jsonl', rec)
        if rec.get('http_status') == 429:
            # Rate-limit failures are retained and retried after backoff, never
            # mistaken for model outputs or a reason to switch credentials.
            time.sleep(20 * (attempt + 1))
            continue
        if rec.get('http_status') == 402 and os.environ.get('DEEPSEEK_API_KEY_FALLBACK') and key_slot == 'primary':
            key = os.environ['DEEPSEEK_API_KEY_FALLBACK']
            key_slot = 'fallback'
            continue
        if rec['valid'] or rec.get('http_status') in (401, 402, 403):
            return rec
    return rec

def run(pilot=False):
    base, key = credentials()
    rows = read_jsonl(OUT / 'selected_states.jsonl')
    if pilot:
        rows = [r for r in rows if r['selection_order'] < 2]
    completed = set()
    calls_path = OUT / 'calls.jsonl'
    if calls_path.exists():
        completed = {(r['city'],r['respondent_id'],r['task'],r['repeat']) for r in read_jsonl(calls_path) if r.get('valid')}
    jobs = [(r,k) for r in rows for k in range(CFG['repeats']) if (r['city'],r['respondent_id'],r['task'],k) not in completed]
    print(json.dumps({'stage':'pilot' if pilot else 'main', 'pending_calls': len(jobs)}), flush=True)
    bad = 0
    with cf.ThreadPoolExecutor(max_workers=CFG['workers']) as pool:
        futures = [pool.submit(call_one,r,k,base,key) for r,k in jobs]
        for n,f in enumerate(cf.as_completed(futures),1):
            result = f.result()
            bad += not result['valid']
            if n % 10 == 0 or n == len(jobs):
                print(json.dumps({'finished':n,'total':len(jobs),'invalid':bad}),flush=True)
    summarize()

def summarize():
    calls = read_jsonl(OUT / 'calls.jsonl')
    valid = {(r['city'],r['respondent_id'],r['task'],r['repeat']):r for r in calls if r.get('valid')}
    by = defaultdict(list)
    for r in valid.values():
        by[(r['city'],r['respondent_id'],str(r['task']))].append(r)
    aggregates=[]
    modes=['car','pt','bike','walk']
    for sid, rs in sorted(by.items()):
        arr=np.array([[r['action']['mode_probabilities'].get(m,0) for m in modes] for r in rs])
        a={k:rs[0][k] for k in ('city','respondent_id','task','state_id','state_sha256')}
        a.update(repeats=len(rs), modes=modes, mean_probabilities=arr.mean(0).tolist(),
                 sd_probabilities=(arr.std(0,ddof=1) if len(arr)>1 else np.zeros(4)).tolist(),
                 mean_departure=np.mean([r['action']['departure_time_shift_min'] for r in rs]).item())
        aggregates.append(a)
    (OUT / 'aggregates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in aggregates),encoding='utf-8')
    usage=defaultdict(int)
    for r in calls:
        for k,v in (r.get('usage') or {}).items():
            if isinstance(v,int): usage[k]+=v
    peak=(usage['prompt_cache_miss_tokens']*1.32+usage['prompt_cache_hit_tokens']*.044+usage['completion_tokens']*3.96)/1e6
    if not usage['prompt_cache_miss_tokens'] and not usage['prompt_cache_hit_tokens']:
        peak=(usage['prompt_tokens']*1.32+usage['completion_tokens']*3.96)/1e6
    summary={'attempts':len(calls),'valid_repeat_targets':len(valid),'states':len(by),
             'complete_states':sum(len(rs)==CFG['repeats'] for rs in by.values()),
             'returned_models':sorted({r['model'] for r in calls if r.get('model')}),
             'usage':dict(usage),'estimated_cost_usd_offpeak_to_peak':[peak/2,peak],
             'cost_is_estimate_not_billed_amount':True,
             'failure_counts':dict(__import__('collections').Counter(r.get('failure_type',r.get('validation_reason','unknown')) for r in calls if not r.get('valid'))),
             'duplicate_completion_ids': len(valid)-len({r.get('completion_id') for r in valid.values()})}
    dump(OUT/'run_summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['preflight','prepare','pilot','run','summarize'])
    ap.add_argument('--condition',choices=['survey_options','ownership'],default='survey_options')
    ap.add_argument('--prompt',choices=['generic','accessibility'],default='generic')
    ap.add_argument('--workers',type=int,default=48)
    ap.add_argument('--per-city',type=int,default=24)
    ap.add_argument('--neutral-identifiers',action='store_true')
    args=ap.parse_args()
    CFG['workers']=args.workers;CFG['per_city']=args.per_city
    if args.prompt == 'accessibility':
        OUT=ROOT/'outputs/revision_20260921/teacher_accessibility_neutral'
        SYSTEM_PROMPT=S8_SYSTEM_PROMPT
        CFG['prompt_version']=TEACHER_S8_PROMPT_VERSION
        CFG['neutral_identifiers']=True
    elif args.neutral_identifiers:
        OUT=ROOT/'outputs/revision_20260921/teacher_neutral'
        CFG['neutral_identifiers']=True
    if args.condition == 'ownership':
        OUT=OUT.with_name(OUT.name+'_ownership')
        CFG['profile']='ownership'
    if CFG.get('neutral_identifiers'):
        CFG['unused_metadata_policy']='Neutral persona ID; constant query/scenario IDs; origin/destination types unspecified. All Student-encoded fields unchanged.'
    OUT.mkdir(parents=True,exist_ok=True)
    # Configuration revisions do not overwrite calls; request hashes and token caps
    # are retained per attempt. The initial pilot used 24 workers/180s/8192 tokens.
    dump(OUT / 'config.json', CFG)
    if args.stage=='preflight': preflight()
    elif args.stage=='prepare': prepare()
    elif args.stage=='summarize': summarize()
    else: run(pilot=args.stage=='pilot')
