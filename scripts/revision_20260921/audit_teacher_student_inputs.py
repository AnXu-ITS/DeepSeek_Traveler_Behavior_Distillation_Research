"""Independent before-call audit using every actual fitted Student extractor.

Reads no human answers and performs no API calls. Compares the neutral Teacher
states to the original ownership evaluation inputs and saved predictions.
"""
from collections import Counter
import datetime,hashlib,json
import numpy as np
import torch
from prepare_survey import ROOT,OUT,ORDER,rows,readj,writej,sha
from evaluate_survey import MODELS,load_model,predict,model_path
from traveler_distillation.schemas.state import UniversalTravelerState

DEST=ROOT/'outputs/revision_20260921/teacher_accessibility_neutral_ownership'
ALLOWED={'.persona.persona_id','.trip.trip_id','.context.context_id','.trip.origin_type','.trip.destination_type'}

def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return [x for k in sorted(set(a)|set(b)) for x in diff(a.get(k),b.get(k),path+'.'+k)]
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return [x for i,(u,v) in enumerate(zip(a,b)) for x in diff(u,v,path+f'[{i}]')]
    return [] if a==b else [path]

def key(r):return r['city'],r['respondent_id'],str(r['task'])
def owned(st):
    st=json.loads(json.dumps(st));p=st['persona']
    for a in st['alternatives']:
        a['available']=bool(p['car_ownership'] and p['driving_license']) if a['mode']=='car' else bool(p['bike_ownership']) if a['mode']=='bike' else True
    return st

def main():
    torch.set_num_threads(2)
    selected=rows(DEST/'selected_states.jsonl');primary=rows(OUT/'task_states.jsonl')
    lookup={key(r):r for r in primary};assert len(selected)==480 and len({key(r) for r in selected})==480
    counts=Counter((r['city'],r['respondent_id']) for r in selected)
    assert len(counts)==48 and set(counts.values())=={10}
    assert Counter(r['city'] for r in selected)=={'Singapore':240,'Shanghai':240}
    original=[owned(lookup[key(r)]['state']) for r in selected];neutral=[r['state'] for r in selected]
    changes=Counter()
    for a,b in zip(original,neutral):
        d=diff(a,b);assert set(d)<=ALLOWED;changes.update(d)
        assert b['trip']['trip_id']=='query' and b['context']['context_id']=='scenario'
        assert b['trip']['origin_type']==b['trip']['destination_type']=='unspecified'
        assert 'Singapore' not in json.dumps(b) and 'Shanghai' not in json.dumps(b)
    A=[UniversalTravelerState.model_validate(s) for s in original];B=[UniversalTravelerState.model_validate(s) for s in neutral]
    indices={}
    for city in ['Singapore','Shanghai']:
        ids=sorted({r['respondent_id'] for r in primary if r['city']==city});tasks=[str(i) for i in range(1,11)] if city=='Singapore' else ORDER
        indices[city]={ (rid,t):(i,j) for i,rid in enumerate(ids) for j,t in enumerate(tasks)}
    results=[]
    for name in MODELS:
        model=load_model(name);ext=model.ext if name=='mnl_s' else model.extractor
        ea=[ext.encode(s) for s in A];eb=[ext.encode(s) for s in B]
        for a,b in zip(ea,eb):
            assert set(a)==set(b)
            assert all(np.array_equal(np.asarray(a[k]),np.asarray(b[k])) for k in a)
        design_error=None
        if name=='mnl_s':
            xa,ma=model.design(original);xb,mb=model.design(neutral)
            assert np.array_equal(xa,xb) and np.array_equal(ma,mb);design_error=float(np.max(abs(xa-xb)))
        P,D=predict(model,name,B)
        saved={c:np.load(OUT/f'predictions/{c.lower()}/ownership/{name}.npz') for c in ['Singapore','Shanghai']}
        oldp=[];oldd=[]
        for r in selected:
            i,j=indices[r['city']][r['respondent_id'],str(r['task'])]
            oldp.append(saved[r['city']]['probabilities'][i,j]);oldd.append(saved[r['city']]['departure'][i,j])
        p_error=float(np.max(abs(P-np.array(oldp))));d_error=float(np.max(abs(D-np.array(oldd))))
        assert p_error<2e-6 and d_error<1e-4
        features_sha=hashlib.sha256(json.dumps(ea,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        results.append(dict(model=name,n_states=480,all_feature_arrays_exactly_equal=True,feature_components=list(ea[0]),feature_sha256=features_sha,
            mnl_full_design_max_abs_difference=design_error,stored_original_ownership_probability_max_abs_difference=p_error,stored_original_ownership_departure_max_abs_difference=d_error,
            checkpoint_sha256=sha(model_path(name))))
        print(name,'480 exact feature matches; max probability delta',p_error,flush=True)
    calls=DEST/'calls.jsonl';attempts=len(rows(calls)) if calls.exists() else 0
    files=[DEST/'selected_states.jsonl',DEST/'before_call_metadata_audit.json',DEST/'system_prompt.txt',DEST/'user_prompt_template.txt',OUT/'task_states.jsonl',ROOT/'src/traveler_distillation/accessibility/accessibility_features.py',ROOT/'src/traveler_distillation/student/features.py',ROOT/'cvpr_workspace/analysis/statistics/reviewer_closure.py']
    result=dict(status='passed',audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_teacher_attempts_at_audit=attempts,before_main_teacher_calls=attempts==0,n_states=480,n_respondents=48,states_per_city=240,n_fitted_models=len(MODELS),n_state_model_feature_comparisons=480*len(MODELS),
        raw_state_differences=dict(changes),allowed_neutral_metadata=sorted(ALLOWED),all_student_consumed_information_unchanged=True,
        interpretation='The neutral Teacher payload preserves all inputs consumed by each frozen Student extractor and by the MNL-S design. Language tokens and standardized numeric/category encodings are distinct interfaces; this audit does not equate those representations or identify pure historical distillation error.',
        answer_use='No questionnaire answers or label grids read; cohort is the original fixed selection. Saved prediction arrays are used only for inference parity.',models=results,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},code_sha256=sha(__file__))
    writej(DEST/'independent_student_feature_audit.json',result)
    print(json.dumps(dict(status='passed',models=len(MODELS),state_model_comparisons=480*len(MODELS),main_teacher_attempts_at_audit=attempts)))

if __name__=='__main__':main()
