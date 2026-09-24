"""Recode frozen questionnaire states without reading answers in state construction."""
from pathlib import Path
import copy, csv, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from survey_alignment_lib import load_bundle, map_state

OUT = ROOT/'outputs/revision_20260921/survey'
SG = ROOT/'outputs/survey_alignment_20260909/bundle'
SH = ROOT/'outputs/shanghai_sp_v1'
ORDER = ['B0','W1','D1','WD1','F1','P1','R1','A_WALK','A_WAIT','A_TRANSFER']
MODES = ['car','pt','bike','walk']

def rows(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def readj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def writej(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writejl(p,v): Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in v),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_id(state): return hashlib.sha256(json.dumps(state,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def normalise_state(state):
    state=copy.deepcopy(state)
    state['persona']['persona_id']='survey_persona'
    state['trip']['trip_id']='survey_trip'
    return state

def primary_states():
    load_bundle(SG)
    sgcfg=readj(SG/'mapping_config.json')
    sgcfg['tasks']={int(k):v for k,v in sgcfg['tasks'].items()}
    # This function only uses pre-existing demographics and task configurations.
    for person in rows(SG/'respondents.jsonl'):
        for task in range(1,11):
            st=normalise_state(map_state(person,task,sgcfg,'primary'))
            yield dict(city='Singapore',respondent_id=person['respondent_id'],task=str(task),
                       profile='survey_options',state_id=state_id(st),state=st)
    for row in rows(SH/'states_p042.jsonl'):
        st=normalise_state(row['state'])
        for alt in st['alternatives']: alt['available']=True
        yield dict(city='Shanghai',respondent_id=row['respondent_id'],task=row['card_id'],
                   profile='survey_options',state_id=state_id(st),state=st)

def labels():
    for r in rows(SG/'human_labels.jsonl'):
        yield dict(city='Singapore',respondent_id=r['respondent_id'],task=str(r['task']),
                   choice_status='selected',chosen_mode=r['choice'])
    ids={r['respondent_id'] for r in rows(SH/'states_p042.jsonl')}
    with (SH/'sp_responses.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['respondent_id'] in ids:
                yield dict(city='Shanghai',respondent_id=r['respondent_id'],task=r['card_id'],
                           choice_status=r['choice_status'],chosen_mode=r['chosen_mode'])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    states=list(primary_states())
    writejl(OUT/'task_states.jsonl',states)
    labs=list(labels())
    writejl(OUT/'human_labels.jsonl',labs)
    dem={}
    for r in states:
        dem[(r['city'],r['respondent_id'])]=dict(city=r['city'],respondent_id=r['respondent_id'],persona=r['state']['persona'])
    writejl(OUT/'respondents.jsonl',dem.values())
    conflicts=[]
    lut={(r['city'],r['respondent_id'],r['task']):r for r in states}
    for r in labs:
        p=lut[r['city'],r['respondent_id'],r['task']]['state']['persona']
        available=dict(car=p['car_ownership'] and p['driving_license'],bike=p['bike_ownership'],pt=True,walk=True)
        if r['choice_status']=='selected' and not available[r['chosen_mode']]:
            conflicts.append(dict(r,ownership_available=available))
    writejl(OUT/'ownership_support_conflicts.jsonl',conflicts)
    cfg=readj(SG/'mapping_config.json')
    sh_contrasts=dict(rain=({'B0':-1,'W1':1},['pt']),delay=({'B0':-1,'D1':1},['pt']),
        rain_delay_interaction=({'B0':1,'W1':-1,'D1':-1,'WD1':1},['pt']),fare=({'B0':-1,'F1':1},['pt']),
        parking=({'B0':-1,'P1':1},['car']),road=({'B0':-1,'R1':1},['car']),
        walk_vs_wait=({'A_WALK':1,'A_WAIT':-1},['pt']),transfer_vs_wait=({'A_TRANSFER':1,'A_WAIT':-1},['pt']))
    contrasts={'Singapore':{k:dict(weights={str(v['base']):-1,str(v['intervention']):1},modes=v['outcome_modes']) for k,v in cfg['pairs'].items()},
               'Shanghai':{k:dict(weights=w,modes=m) for k,(w,m) in sh_contrasts.items()}}
    writej(OUT/'contrasts.json',contrasts)
    summary={city:dict(respondents=sum(k[0]==city for k in dem),states=sum(r['city']==city for r in states),
          selected=sum(r['city']==city and r['choice_status']=='selected' for r in labs),
          unscored_nonmode=sum(r['city']==city and r['choice_status']!='selected' for r in labs),
          ownership_support_conflicts=sum(r['city']==city for r in conflicts)) for city in ['Singapore','Shanghai']}
    writej(OUT/'mapping_audit.json',dict(city_summary=summary,mode_order=MODES,epsilon_nll=1e-8,
        primary='All four displayed survey options available. Car represents the original driving/taxi composite; private-car utility remains a construct mismatch.',
        raw_unchanged=True,feature_sources={
          'Singapore':dict(measured_persona=['age_group','employment','driving_license','car_access','bike_access','habitual_mode','income_group'],
                          questionnaire=['distance','mode_total_times','drive_cost','taxi_cost','pt_fare','pt_access','transfers','weather','delay','roadworks'],
                          defaults=cfg['defaults'],profiles=cfg['profiles']),
          'Shanghai':dict(persona_mapping=readj(ROOT/'docs/plans/shanghai_survey_v3/field_mapping.json'),
                          unmeasured=['household_size=2 (sensitivity 1,4)','All numeric travel times, distance, departure and PT components in cards.json were reference imputations, not displayed online','baseline fares and car costs; qualitative intervention magnitudes','coverage_ratio=0.8'],
                          displayed='Qualitative scenarios only; explicit intervention values were PT fare=6 yuan, parking fee=30 yuan, and transfers=1. All four mode options displayed, driving/taxi composite retained.',
                          evidence='Archived HTML and parsed deployed questionnaire; both decorative header image assets visually inspected and contain no task attributes.',
                          network='Excluded from primary states; tested separately with assigned OD')} ,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [SG/'manifest.json',SG/'mapping_config.json',SH/'states_p042.jsonl',SH/'sp_responses.csv',ROOT/'docs/plans/shanghai_survey_v3/cards.json',ROOT/'docs/plans/shanghai_survey_v3/field/上海出行方式调查_问卷.md',ROOT/'docs/plans/shanghai_survey_v3/field/上海出行方式调查_原始页面.html']},
        outputs={p.name:sha(p) for p in [OUT/'task_states.jsonl',OUT/'human_labels.jsonl',OUT/'respondents.jsonl',OUT/'contrasts.json']}))
    print(json.dumps(summary))

if __name__=='__main__': main()
