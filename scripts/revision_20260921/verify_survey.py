"""Numerical and source-provenance checks for the revision survey evidence."""
import csv,json,sys
from pathlib import Path
import numpy as np
from prepare_survey import ROOT,OUT,SG,SH,MODES,readj,rows,writej,sha
from survey_alignment_lib import read_survey,load_config

def main():
    checks={}
    workbook=next((ROOT/'新加披调查问卷').glob('*.xlsx'))
    questionnaire=ROOT/'新加披调查问卷/新加坡城市出行选择研究_调查问卷.md'
    orig=readj(SG/'audit.json')
    assert sha(workbook)==orig['source_hashes']['workbook'];checks['sg_raw_workbook_frozen_hash']=True
    load_config(ROOT/'configs/survey_alignment.yaml',questionnaire);checks['sg_task_transcription_matches_displayed_instrument']=True
    people,labels,audit=read_survey(workbook)
    assert people==rows(SG/'respondents.jsonl') and labels==rows(SG/'human_labels.jsonl');checks['sg_raw_demographics_and_answers_reparsed_identically']=True
    for name in ['respondents_p041.csv','respondents_p042.csv','respondents_p044.csv','sp_responses.csv','input_coverage.csv']:
        assert (SH/name).read_bytes()==(OUT/'raw_conversion_audit'/name).read_bytes()
    checks['sh_raw_workbook_reconversion_identical_five_exports']=True
    # Verify every saved prediction has valid probabilities, and zero only on masked modes.
    nfiles=0
    for path in (OUT/'predictions').glob('*/*/*.npz'):
        z=np.load(path);p=z['probabilities'];mask=z['availability'];dep=z['departure']
        assert p.shape==mask.shape and p.shape[1:]==(10,4)
        assert np.isfinite(p).all() and np.isfinite(dep).all() and (p>=0).all() and abs(p.sum(2)-1).max()<2e-6
        assert (p[~mask]==0).all()
        sc=readj(path.with_suffix('.json'))
        grid=readj(OUT/f'grid_{path.parts[-3]}.json');y=np.array(grid['labels']);valid=y>=0
        assert sc['n_choices']==valid.sum()
        assert abs(sc['accuracy']-(p.argmax(2)[valid]==y[valid]).mean())<1e-12
        counts=np.asarray(sc['confusion_matrix']);assert counts.sum()==valid.sum()
        tp=p[valid][np.arange(valid.sum()),y[valid]]
        assert abs(sc['nll_clipped']-np.mean(-np.log(np.maximum(tp,1e-8))))<1e-12
        assert (sc['nll_raw']=='Infinity')==bool((tp==0).any())
        assert sc['structural_zero_labels']==int((~mask[valid][np.arange(valid.sum()),y[valid]]).sum())
        if 'n_undecided' in sc:
            sc['n_unscored']=sc.pop('n_undecided');sc['unscored_policy']='Raw none-suitable response has no chosen mode; explicit masked mode choices remain scored'
            writej(path.with_suffix('.json'),sc)
        nfiles+=1
    checks['all_model_profile_probability_and_metric_checks']=nfiles
    # Directly compare historical production predictions to new ownership inference.
    grid=readj(OUT/'grid_shanghai.json');ids=grid['respondents'];tasks=grid['tasks']
    source=rows(SH/'predictions/s9_p042.jsonl');by={(r['respondent_id'],r['card_id']):r for r in source}
    old=np.array([[[by[r,t]['probabilities'][m] for m in MODES] for t in tasks] for r in ids])
    new=np.load(OUT/'predictions/shanghai/ownership/s9.npz')['probabilities']
    err=float(abs(old-new).max());assert err<2e-6;checks['original_fixed_s9_probability_parity_max_abs']=err
    net=ROOT/'outputs/shanghai_survey_matsim_321x10_v1'
    by={(r['respondent_id'],t):r for t in tasks for r in rows(net/t/'predictions.jsonl')}
    old=np.array([[[by[r,t]['probabilities'].get(m,0.) for m in MODES] for t in tasks] for r in ids])
    new=np.load(OUT/'od_sensitivity/ownership/original_00.npz')['probabilities']
    err=float(abs(old-new).max());assert err<2e-6;checks['original_network_s9_probability_parity_max_abs']=err
    for p in [OUT/'bootstrap_singapore.npz',OUT/'bootstrap_shanghai.npz']:
        z=np.load(p);assert z['indices'].shape==(10000,len(z['respondents']))
    checks['shared_full_respondent_bootstrap_grids']=True
    # Independently reconstruct a complete-case response interval directly from
    # saved index resamples, without Scorer's count-matrix implementation.
    y=np.array(grid['labels']);P=np.load(OUT/'predictions/shanghai/survey_options/s9.npz')['probabilities']
    j0=tasks.index('A_WAIT');j1=tasks.index('A_WALK');keep=(y[:,j0]>=0)&(y[:,j1]>=0)
    h=(y[:,j1]==1).astype(float)-(y[:,j0]==1);v=P[:,j1,1]-P[:,j0,1]-h
    draws=np.load(OUT/'bootstrap_shanghai.npz')['indices'];picked_keep=keep[draws]
    boots=np.where(picked_keep,v[draws],0).sum(1)/picked_keep.sum(1)
    recorded=readj(OUT/'predictions/shanghai/survey_options/s9.json')['responses']['walk_vs_wait']['bias_interval']
    assert np.allclose(np.quantile(boots,[.025,.975]),recorded['ci95'],rtol=0,atol=1e-12)
    assert np.allclose(np.quantile(boots,[.05/16,1-.05/16]),recorded['ci_family'],rtol=0,atol=1e-12)
    checks['complete_case_bootstrap_independently_reconstructed_from_indices']=True
    original_rows={r['respondent_id']:r for r in rows(OUT/'od_sensitivity/original_00_baseline.jsonl')}
    for path in (OUT/'od_sensitivity').glob('permutation_*_baseline.jsonl'):
        rr=rows(path);assert sorted(r['donor_id'] for r in rr)==sorted(ids)
        for r in rr:
            src=original_rows[r['donor_id']]
            assert r['alternatives']==src['alternatives']
            assert all(r[k]==src[k] for k in ['departure_min','origin_node','destination_node'])
    checks['all_twenty_permutations_preserve_complete_supply_vector_od_and_departure']=True
    runs=list(csv.DictReader((OUT/'od_sensitivity/runs.csv').open(encoding='utf-8-sig')))
    assert len(runs)==82 and all(sum(r['kind']==kind and r['profile']==profile for r in runs)==20 for kind in ['permutation','reassignment'] for profile in ['survey_options','ownership'])
    checks['od_twenty_reassignments_and_twenty_full_vector_permutations_both_masks']=True
    original=readj(OUT/'od_sensitivity/ownership/original_00.json')
    assert original['historical_support_scoring']['n']==3206
    assert round(original['historical_support_scoring']['network_accuracy']*100,2)==83.31
    checks['historical_3206_accuracy_reproduced']=True
    writej(OUT/'verification.json',dict(all_passed=True,checks=checks,raw_input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [workbook,questionnaire,ROOT/'上海调查问卷/384387017_按文本_上海出行方式调查_341_341.xlsx']},scripts={p.name:sha(p) for p in Path(__file__).parent.glob('*survey.py')}))
    writej(OUT/'sealed_manifest.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='sealed_manifest.json' and p.suffix!='.log'},
        scripts={name:sha(Path(__file__).parent/name) for name in ['prepare_survey.py','evaluate_survey.py','od_sensitivity.py','summarize_survey.py','verify_survey.py']}))
    print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
