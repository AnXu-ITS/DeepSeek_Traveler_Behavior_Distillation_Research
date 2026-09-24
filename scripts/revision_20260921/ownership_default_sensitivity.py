"""Answer-independent default sensitivities on the original model support.

Historical all-options profiles remain unchanged. Each numerical/categorical
default profile is reproduced exactly, then the original ownership mask is
applied. Comparisons use fixed models and common respondent bootstrap draws.
"""
from pathlib import Path
from collections import defaultdict
import json,sys
import numpy as np
import torch
from prepare_survey import ROOT,OUT,SG,readj,rows,writej,sha
from evaluate_survey import (MODELS,VARIANTS,SEEDS,model_path,load_model,predict,
                            profile_states,make_grid,Scorer,csvout)
from traveler_distillation.schemas.state import UniversalTravelerState

DEST=OUT/'ownership_default_sensitivity'
FAMILIES=['s9','mnl_s']+VARIANTS

def owned(states):
    for row in states:
        st=row['state'];p=st['persona']
        for a in st['alternatives']:
            a['available']=(p['car_ownership'] and p['driving_license']) if a['mode']=='car' else p['bike_ownership'] if a['mode']=='bike' else True
    return states

def deltas(scorer,P,B):
    y=scorer.y;v=y>=0;Y=scorer.Y
    avg=lambda x:np.where(v,x,0.).sum(1)/v.sum(1)
    true=lambda p:p[np.arange(len(y))[:,None],np.arange(10)[None,:],np.maximum(y,0)]
    a=avg((P.argmax(2)==y).astype(float)-(B.argmax(2)==y))
    b=avg(((P-Y)**2).sum(2)-((B-Y)**2).sum(2))
    n=avg(-np.log(np.maximum(true(P),1e-8))+np.log(np.maximum(true(B),1e-8)))
    values={'accuracy':a,'brier':b,'nll_clipped':n,'pt_probability':avg(P[:,:,1]-B[:,:,1])}
    out={m:scorer.interval(x) for m,x in values.items()}
    for name,tt,w,m,keep,h in scorer.conditions:
        d=((P-B)[:,tt,:][:,:,m].sum(2)*w).sum(1)
        out['response_'+name]=scorer.interval(d,keep)
    return out,values

def main():
    DEST.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    primary=rows(OUT/'task_states.jsonl');labs={(r['city'],r['respondent_id'],r['task']):r for r in rows(OUT/'human_labels.jsonl')}
    config=readj(SG/'mapping_config.json')
    profiles={'Singapore':list(config['profiles'])[1:],
              'Shanghai':['household_1','household_4','coverage_0.5','coverage_1.0','times_0.8','times_1.2','pt_wait_more','pt_walk_more']}
    sources={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'task_states.jsonl',OUT/'human_labels.jsonl',OUT/'mapping_audit.json',OUT/'task_field_sources.csv',SG/'mapping_config.json']}
    sources.update({str(model_path(n).relative_to(ROOT)):sha(model_path(n)) for n in MODELS})
    models={n:load_model(n) for n in MODELS};all_scores=[];all_delta=[];families=[];family_resp={};checks=[]
    for city,plist in profiles.items():
        crows=[r for r in primary if r['city']==city]
        ids,tasks,_,y=make_grid(city,crows,labs);sc=Scorer(city,ids,tasks,y)
        base={n:np.load(OUT/f'predictions/{city.lower()}/ownership/{n}.npz')['probabilities'] for n in MODELS}
        family_resp[city]={}
        for profile in plist:
            old=profile_states(city,profile,crows)
            # No human labels enter profile construction; existing default changes
            # remain byte-for-byte the same except the subsequently applied mask.
            numeric_sha=sha(OUT/f'predictions/{city.lower()}/{profile}/s9.npz')
            altered=owned(old);_,_,ordered,y2=make_grid(city,altered,labs);assert np.array_equal(y,y2)
            states=[UniversalTravelerState.model_validate(r['state']) for r in ordered]
            mask=np.array([[a.available for a in s.alternatives] for s in states]).reshape(len(ids),10,4)
            assert np.array_equal(mask,np.load(OUT/f'predictions/{city.lower()}/ownership/s9.npz')['availability'])
            target=DEST/city.lower()/profile;target.mkdir(parents=True,exist_ok=True)
            state_hash=__import__('hashlib').sha256(json.dumps([r['state'] for r in ordered],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            pp={};metric_units={};scores={}
            for name in MODELS:
                npz=target/f'{name}.npz';result=target/f'{name}.json'
                if npz.exists() and result.exists():
                    data=np.load(npz);P=data['probabilities'];D=data['departure'];score=readj(result)
                    assert score['state_sha256']==state_hash and score['checkpoint_sha256']==sha(model_path(name))
                else:
                    P,D=predict(models[name],name,states);P=P.reshape(len(ids),10,4);D=D.reshape(len(ids),10)
                    score=sc.score(P,mask,ci=True)
                    score.update(city=city,profile=profile,model=name,support='ownership',state_sha256=state_hash,checkpoint_sha256=sha(model_path(name)))
                    np.savez_compressed(npz,probabilities=P,departure=D,availability=mask)
                    writej(result,score)
                assert np.all(P[~mask]==0);pp[name]=P;scores[name]=score
                ds,units=deltas(sc,P,base[name]);metric_units[name]=units
                all_scores.append(dict(city=city,profile=profile,model=name,accuracy=score['accuracy'],balanced_accuracy=score['balanced_accuracy'],nll_clipped=score['nll_clipped'],nll_raw=score['nll_raw'],brier=score['brier'],pt_bias=score['probability_share_bias'][1],mean_absolute_response_bias=score['mean_absolute_response_bias'],structural_zero_labels=score['structural_zero_labels']))
                for metric,z in ds.items():all_delta.append(dict(city=city,profile=profile,model=name,metric=metric,**z))
            family_resp[city][profile]={}
            for family in FAMILIES:
                names=[family] if family in ['s9','mnl_s'] else [f'{family}_seed{s}' for s in SEEDS]
                row=dict(city=city,profile=profile,model=family,n_training_seeds=len(names))
                for metric in ['accuracy','balanced_accuracy','nll_clipped','brier','mean_absolute_response_bias']:
                    arr=np.array([scores[n][metric] for n in names]);row[metric+'_mean']=float(arr.mean());row[metric+'_sd']=float(arr.std(ddof=1)) if len(names)>1 else 0.
                arr=np.array([scores[n]['probability_share_bias'][1] for n in names]);row['pt_bias_mean']=float(arr.mean());row['pt_bias_sd']=float(arr.std(ddof=1)) if len(names)>1 else 0.
                families.append(row)
                meanP=np.mean([pp[n] for n in names],axis=0);meanB=np.mean([base[n] for n in names],axis=0)
                response=sc.score(meanP,mask,ci=True)['responses'];dd,_=deltas(sc,meanP,meanB)
                # Choice deltas average per-seed respondent metrics, never classify
                # the ensemble; linear probability response commutes with averaging.
                for metric in ['accuracy','brier','nll_clipped','pt_probability']:
                    dd[metric]=sc.interval(np.mean([metric_units[n][metric] for n in names],axis=0))
                for contrast,r in response.items():
                    r['training_seed_sd']=float(np.std([scores[n]['responses'][contrast]['model'] for n in names],ddof=1)) if len(names)>1 else 0.
                family_resp[city][profile][family]=dict(responses=response,paired_change_from_original_support_baseline=dd,n_training_seeds=len(names),aggregation='Arithmetic mean of per-seed choice metrics; response from seed-mean probabilities. Common respondent bootstrap conditional on fitted seeds.')
            checks.append(dict(city=city,profile=profile,n_states=len(states),state_sha256=state_hash,mask_equal_original_ownership=True,numerical_profile_source='Historical all-options profile construction; only support changed',historical_all_options_s9_npz_sha256=numeric_sha,n_models=len(MODELS)))
            print(f'{city} {profile}: {len(MODELS)} models complete',flush=True)
    csvout(DEST/'model_scores.csv',all_scores);csvout(DEST/'paired_default_changes.csv',all_delta);csvout(DEST/'seed_summary.csv',families)
    writej(DEST/'family_responses.json',family_resp)
    writej(DEST/'verification.json',dict(status='passed',n_evaluations=len(all_scores),profiles=checks,sources_preserved=all(sha(ROOT/p)==h for p,h in sources.items()),source_sha256=sources,
        inference='Original unchanged models; all requested profiles, no outcome-dependent choice of defaults',uncertainty='10000 common respondent draws, seed917, full task panels; nominal95 and within-city response-family Bonferroni intervals; fixed training seeds',
        multiplicity='No additional adjustment across sensitivity profiles or model families; no confirmatory superiority claims',code_sha256=sha(__file__)))
    # This new record explicitly supersedes prose recommendations, never raw rows.
    writej(OUT/'authoritative_mapping_revision.json',dict(status='authoritative interpretation; historical exports retained unchanged',supersedes=[dict(path='mapping_audit.json',field='primary'),dict(path='task_field_sources.csv',field='available source annotation')],
        primary='Original ownership-conditioned model contract; all explicit human mode labels scored, including 8 Singapore and 2 Shanghai support conflicts; no taxi-inclusive behavioral-validity claim',
        task_states_profile='task_states.jsonl intentionally retains historical all-displayed-options state export. Ownership availability is a deterministic derived mask, not a modification of questionnaire options or raw answers.',
        ownership={'car':'car_ownership and driving_license','bike':'bike_ownership','pt':True,'walk':True},
        all_options='All four displayed options is an alternative support-expansion diagnostic; displayed option does not establish private-car or bicycle access.',
        defaults='ownership_default_sensitivity contains the same fixed numerical/categorical profiles evaluated on original support; old predictions/<city>/<default> remain all-options sensitivity, not primary-support evidence.',
        fields='authoritative_task_field_sources.csv retains displayed/default field provenance and adds evaluation support explicitly',
        human_exclusions='Shanghai two raw none-suitable responses (unable) excluded only from four-mode scoring and contrasts requiring that label. No explicit masked label removed.',
        claim_scope='Original support satisfies source availability rules only; this does not prove survey numeric features/personas are in distribution.',source_sha256=sources))
    import csv
    fieldrows=list(csv.DictReader((OUT/'task_field_sources.csv').open(encoding='utf-8-sig')))
    for r in fieldrows:
        r['record_profile']='displayed-options reference encoding, not primary availability'
        r['primary_support']='car ownership AND licence; bicycle access; PT and walk always available'
        if r['field']=='available':
            r['source']='All four questionnaire options are displayed. This stored value describes the historical all-options diagnostic, not genuine respondent availability.'
            r['primary_encoded_value']='car_ownership AND driving_license' if r['mode']=='car' else 'bike_ownership' if r['mode']=='bike' else 'True'
        else:r['primary_encoded_value']=r['encoded_value']
    csvout(OUT/'authoritative_task_field_sources.csv',fieldrows)
    report=['# Ownership-conditioned default sensitivity','',
       'All original default profiles were rerun on the original ownership support. No profile was selected from human outcomes; the original all-options profiles and all raw answers remain unchanged. Every profile includes S9, MNL-S, and three seeds of each of the four controlled neural objectives.','',
       'The point metrics use all explicit labels, retaining support conflicts. Paired default-minus-baseline intervals use common respondent resamples (10,000 draws, seed 917). Neural family choice metrics average individual-seed metrics; response estimates average seed probabilities before respondent resampling. These intervals condition on fitted seeds. Profile comparisons are sensitivity analyses, with no claim of simultaneous confidence across profiles.','',
       '| City | Family | Baseline accuracy, % | Sensitivity accuracy range, % | Baseline PT bias, pp | Sensitivity PT-bias range, pp |',
       '|---|---|---:|---:|---:|---:|']
    orig=list(csv.DictReader((OUT/'seed_summary.csv').open(encoding='utf-8-sig')))
    for city in profiles:
        for family in FAMILIES:
            rr=[r for r in families if r['city']==city and r['model']==family];b=next(r for r in orig if r['city']==city and r['profile']=='ownership' and r['model']==family)
            report.append(f"| {city} | {family} | {100*float(b['accuracy_mean']):.2f} | {100*min(r['accuracy_mean'] for r in rr):.2f} to {100*max(r['accuracy_mean'] for r in rr):.2f} | {100*float(b['pt_bias_mean']):.2f} | {100*min(r['pt_bias_mean'] for r in rr):.2f} to {100*max(r['pt_bias_mean'] for r in rr):.2f} |")
    response_ranges=[]
    baseline_families=readj(OUT/'family_responses.json')
    for city,profs in family_resp.items():
        for family in FAMILIES:
            for contrast,baseline in baseline_families[city]['ownership'][family]['responses'].items():
                rr=[(p,v[family]['responses'][contrast]) for p,v in profs.items()]
                low=min(rr,key=lambda z:z[1]['bias']);high=max(rr,key=lambda z:z[1]['bias'])
                response_ranges.append(dict(city=city,model=family,contrast=contrast,n=baseline['n'],baseline_bias_pp=100*baseline['bias'],min_bias_pp=100*low[1]['bias'],min_profile=low[0],max_bias_pp=100*high[1]['bias'],max_profile=high[0],min_model_response_pp=100*min(r['model'] for p,r in rr),max_model_response_pp=100*max(r['model'] for p,r in rr),n_profiles=len(rr)))
    csvout(DEST/'family_response_ranges.csv',response_ranges)
    report+=['','These ranges pool only prespecified, one-profile-at-a-time variants. They neither identify the true unmeasured attributes nor justify choosing the variant with the highest accuracy. Response directions and paired changes must be inspected by contrast in family_responses.json; a stable accuracy is not a guarantee of response stability.',
      '', 'For S9, the Singapore poorer-access response remains attenuated (bias +9.36 to +11.44 pp), while its delay bias ranges from -4.62 to -0.74 pp. Shanghai delay remains exaggerated (bias -16.53 to -10.25 pp), fare response remains reversed (+6.65 to +7.72 pp bias), and walking-versus-waiting remains reversed (-13.73 to -10.62 pp bias). These statements describe point-estimate signs across the tested profiles, not simultaneous significance or validation of the missing values.']
    (DEST/'OWNERSHIP_DEFAULT_SENSITIVITY.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    writej(DEST/'manifest.json',dict(files={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file() and p.name!='manifest.json'},n_models=len(MODELS),n_evaluations=len(all_scores),code_sha256=sha(__file__)))
    print(json.dumps(dict(status='complete',n_evaluations=len(all_scores),sources_preserved=True)),flush=True)

if __name__=='__main__':main()
