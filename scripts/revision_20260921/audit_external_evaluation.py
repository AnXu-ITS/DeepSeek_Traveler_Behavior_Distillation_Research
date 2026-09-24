"""Independent read-only identity/denominator checks for other revision branches."""
from pathlib import Path
import sys,collections,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from departure_baselines import ROOT,OUT,read,rows,write,sha,MODES
import numpy as np
from traveler_distillation.accessibility.accessibility_features import S8FeatureExtractor
from traveler_distillation.schemas.state import UniversalTravelerState
import torch

def main():
    sur=ROOT/'outputs/revision_20260921/survey';tea=ROOT/'outputs/revision_20260921/teacher'
    primary=rows(sur/'task_states.jsonl');lookup={(r['city'],str(r['respondent_id']),str(r['task'])):r for r in primary}
    selected=rows(tea/'selected_states.jsonl')
    assert len(lookup)==len(primary)
    ck=torch.load(ROOT/'releases/s9_supply_aware_v2/checkpoint/model.pt',weights_only=False,map_location='cpu')
    ext=S8FeatureExtractor.from_state_dict(ck['extractor_state'])
    mismatches=[]
    for r in selected:
        source=lookup[r['city'],str(r['respondent_id']),str(r['task'])]
        assert r['state_id']==source['state_id']
        a=ext.encode(UniversalTravelerState.model_validate(r['state']));b=ext.encode(UniversalTravelerState.model_validate(source['state']))
        if a!=b:mismatches.append((r['city'],r['respondent_id'],r['task']))
    assert not mismatches
    raw_calls=rows(tea/'calls.jsonl');valid=[r for r in raw_calls if r.get('valid')]
    keys=[(r['city'],str(r['respondent_id']),str(r['task']),r['repeat']) for r in valid]
    unique=len(set(keys));probs=np.array([[r['action']['mode_probabilities'].get(m,0) for m in MODES] for r in valid])
    support={}
    for city in ['Singapore','Shanghai']:
        grid=read(sur/f'grid_{city.lower()}.json');Y=np.array(grid['labels']);v=Y>=0;ids=grid['respondents'];tasks=grid['tasks'];y=Y[v]
        support[city]={}
        for profile in ['survey_options','ownership','car_relaxed']:
            z=np.load(sur/f'predictions/{city.lower()}/{profile}/s9.npz');p=z['probabilities'];mask=z['availability'];score=read(sur/f'predictions/{city.lower()}/{profile}/s9.json')
            pp=p[v];truep=pp[np.arange(len(y)),y];structural=(~mask[v][np.arange(len(y)),y]).sum()
            acc=float((p.argmax(2)[v]==y).mean());brier=float(np.square(pp-np.eye(4)[y]).sum(1).mean())
            assert abs(acc-score['accuracy'])<1e-12 and abs(brier-score['brier'])<1e-12
            assert structural==score['structural_zero_labels'];assert score['n_choices']==v.sum()
            support[city][profile]=dict(n_respondents=len(ids),n_choices=int(v.sum()),n_unscored=int((~v).sum()),accuracy=acc,brier=brier,structural_masked_labels=int(structural),raw_NLL_infinite=bool((truep==0).any()),min_true_probability=float(truep.min()))
        for r in [r for r in selected if r['city']==city]:
            assert str(r['respondent_id']) in ids and str(r['task']) in tasks
    findings=[
       'Selected Teacher states match primary Student feature tensors exactly for all 480 tasks; changed opaque person/trip identifiers do not enter the extractor.',
       'Teacher analysis uses the same respondent/task grid and the same complete-case contrast rows for Human, Teacher and Student; bootstrap draws are shared across all models and all decomposition terms within each contrast.',
       'Choice scoring includes explicitly selected mask-conflicting labels; only absent/undecided labels are excluded. Primary survey_options and ownership sensitivity use the same answers.',
       'Teacher age-by-car-ownership round-robin sample is a balanced-strata diagnostic sample, not a probability-weighted estimate of either city population. Its Human response must come from that same selected subset; current analysis does so.',
       'Shanghai reference numeric encoding matches Teacher to Student numerically, but the archived human instrument provides qualitative scenarios and a few explicit numeric interventions, not the whole reference table; exact numerical Teacher-Human task equivalence must not be claimed.',
       'SG survey_options makes all offered modes available; the older ownership-conditioned rule removes named alternatives. These compare option/support mappings, not a causal effect of car/bike ownership. The composite driving/taxi utility remains unresolved.',
       'analyze_teacher.py currently reports clipped NLL and count of exact zero-probability labels but omits an explicit raw-NLL Infinity field. Add or state this when any true_zero_labels>0; epsilon=1e-8 must be documented.',
       'TeacherResponseValidator allows probability sums within 1e-3 of one. Current valid calls are checked below; normalize before scoring only if future valid calls actually deviate materially, retaining raw outputs.',
       'Teacher analysis refuses incomplete selected groups and requires all three repeats for every selected task. Current calls may still be running; do not interpret absence of final Teacher analysis as completed evidence.'
    ]
    write(OUT/'independent_survey_teacher_audit.json',dict(status='source_and_available_outputs_checked',selected_tasks=len(selected),feature_identity_mismatches=mismatches,valid_call_records=len(valid),unique_valid_call_keys=unique,duplicate_valid_repeat_keys=len(valid)-unique,max_valid_teacher_probability_sum_error=float(np.abs(probs.sum(1)-1).max()) if len(valid) else None,choice_denominator_recomputation=support,findings=findings,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [sur/'task_states.jsonl',sur/'human_labels.jsonl',tea/'selected_states.jsonl',ROOT/'scripts/revision_20260921/analyze_teacher.py']}))
    print(json.dumps(dict(feature_identity_tasks=len(selected),valid_calls=len(valid),prob_sum_error=float(abs(probs.sum(1)-1).max()),support=support),indent=2))
if __name__=='__main__':main()
