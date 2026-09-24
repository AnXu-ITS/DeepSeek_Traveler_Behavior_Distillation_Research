"""Audit admissibility of choice-set expansion; preserve every previous result."""
from pathlib import Path
import copy,csv,json,sys,collections
import numpy as np
import torch
from prepare_survey import ROOT,OUT,MODES,readj,rows,writej,sha
from evaluate_survey import MODELS,load_model,profile_states,csvout
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.student.features import GLOBAL_CAT,GLOBAL_NUM
from traveler_distillation.student.dataset import collate_batch

DEST=OUT/'validity_diagnosis'

def support_summary(records):
    by=collections.Counter();bad_car=bad_bike=0
    for r in records:
        s=r['state'];p=s['persona'];a={x['mode']:x['available'] for x in s['alternatives']}
        by[(bool(p['car_ownership']),bool(p['driving_license']),bool(p['bike_ownership']),bool(a['car']),bool(a['bike']))]+=1
        bad_car+=bool(a['car'])!=bool(p['car_ownership'] and p['driving_license'])
        bad_bike+=bool(a['bike'])!=bool(p['bike_ownership'])
    return dict(n_states=len(records),n_personas=len({r['state']['persona']['persona_id'] for r in records}),
        car_ownership_rule_violations=bad_car,bike_ownership_rule_violations=bad_bike,
        available_car_with_no_ownership_or_licence=sum(n for (o,l,b,c,ba),n in by.items() if c and not(o and l)),
        available_bike_without_ownership=sum(n for (o,l,b,c,ba),n in by.items() if ba and not b),
        combinations=[dict(car_ownership=o,licence=l,bike_ownership=b,car_available=c,bike_available=ba,n=n) for (o,l,b,c,ba),n in sorted(by.items())])

def tensor_features(ext,states):
    fs=[ext.encode(s) for s in states]
    return collate_batch([{k:torch.tensor(f[src],dtype=dtype) for k,src,dtype in [('global_cat','global_cat',torch.long),('global_num','global_num',torch.float32),('alt_mode_idx','alt_mode_idx',torch.long),('alt_num','alt_num',torch.float32),('alt_mask','alt_available',torch.float32)]} for f in fs])

def main():
    DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    manifest=readj(ROOT/'outputs/matched_response_v1/bundle/manifest.json')
    sources={};srcrows={}
    for source in ['legacy','joint','accessibility']:
        p=Path(manifest['source_paths'][source]);srcrows[source]=rows(p)
        assert sha(p)==manifest['source_hashes'][source]
        sources[source]=support_summary(srcrows[source])
    p=Path(manifest['source_paths']['mechanism']);quad=rows(p)
    srcrows['mechanism']=[dict(state=v['state']) for r in quad for v in r['members'].values()]
    assert sha(p)==manifest['source_hashes']['mechanism'];sources['mechanism']=support_summary(srcrows['mechanism'])
    bundles={}
    for split in ['train','val','test']:
        rr=rows(ROOT/f'outputs/matched_response_v1/bundle/{split}_endpoints.jsonl')
        bundles[split]=support_summary(rr)
        bundles[split]['sources']={src:support_summary([r for r in rr if r['source']==src]) for src in sorted({r['source'] for r in rr})}
    writej(DEST/'training_support.json',dict(original_label_sources=sources,controlled_bundle=bundles,
        interpretation='Original source counts include train/validation/test and repeat context states; controlled train counts are the frozen 1839 endpoints. Zero violations in every original source proves the same availability invariant throughout S9 replay sources and inherited generative supervision, without counting these states as independent travelers.'))
    primary=rows(OUT/'task_states.jsonl');group_rows=[];model_rows=[];feature_audit={};teacher_audit={};direct={}
    s9=load_model('s9')
    # Direct architecture diagnostic uses identical feature tensors and only changes mask.
    for city in ['Singapore','Shanghai']:
        grid=readj(OUT/f'grid_{city.lower()}.json');ids=grid['respondents'];tasks=grid['tasks'];y=np.array(grid['labels']);valid=y>=0
        lookup={(r['respondent_id'],r['task']):r for r in primary if r['city']==city}
        order=[lookup[r,t] for r in ids for t in tasks]
        ss=[UniversalTravelerState.model_validate(r['state']) for r in order]
        oldmask=np.array([[s.persona.car_ownership and s.persona.driving_license,True,s.persona.bike_ownership,True] for s in ss]).reshape(len(ids),10,4)
        owner=np.array([ss[i*10].persona.car_ownership for i in range(len(ids))]);lic=np.array([ss[i*10].persona.driving_license for i in range(len(ids))]);bike=np.array([ss[i*10].persona.bike_ownership for i in range(len(ids))]);car=owner&lic
        anyopen=~(car&bike)
        b=tensor_features(s9.extractor,ss);bm={k:v.clone() for k,v in b.items()};bm['alt_mask']=torch.tensor(oldmask.reshape(-1,4),dtype=torch.float32)
        with torch.no_grad():a=s9.model(b);c=s9.model(bm)
        err=float((a['utilities']-c['utilities']).abs().max());assert err==0
        # Analytic masked-softmax gradients for an originally valid PT target.
        u=a['utilities'].detach().clone().requires_grad_(True);p=torch.softmax(u.masked_fill(bm['alt_mask']<.5,float('-inf')),1);loss=-torch.log(p[:,1]).mean();loss.backward()
        masked_gradient=float(u.grad[bm['alt_mask']<.5].abs().max());assert masked_gradient==0
        direct[city]=dict(n_states=len(ss),utility_max_abs_change=err,masked_choice_loss_logit_gradient_max_abs=masked_gradient,
            departure_mean_abs_change=float((a['departure_time_shift_min']-c['departure_time_shift_min']).abs().mean()),
            note='Utilities unchanged exactly; selection probability is renormalized. Parameters are shared and trained, but unavailable mode-persona combinations have no direct choice supervision. Departure changes because available-alt pooling changes.')
        # Store only two diagnostically meaningful arrays, one per city.
        full=np.load(OUT/f'predictions/{city.lower()}/survey_options/s9.npz')['probabilities'];own=np.load(OUT/f'predictions/{city.lower()}/ownership/s9.npz')['probabilities']
        newmass=(full*~oldmask).sum(2)
        np.savez_compressed(DEST/f'{city.lower()}_support_expansion.npz',utilities=a['utilities'].numpy().reshape(len(ids),10,4),probability_full=full,probability_ownership=own,
            old_availability=oldmask,newly_enabled_probability_mass=newmass,car_ownership=owner,licence=lic,bike_ownership=bike,respondents=np.array(ids),tasks=np.array(tasks))
        for name in MODELS:
            f=np.load(OUT/f'predictions/{city.lower()}/survey_options/{name}.npz')['probabilities'];o=np.load(OUT/f'predictions/{city.lower()}/ownership/{name}.npz')['probabilities']
            conditional=f*oldmask;conditional/=conditional.sum(2,keepdims=True)
            renormerror=float(abs(conditional-o).max())
            if name!='mnl_s':assert renormerror<1e-6
            mass=(f*~oldmask).sum(2);winner=f.argmax(2);newwinner=~oldmask[np.arange(len(ids))[:,None],np.arange(10)[None,:],winner]
            model_rows.append(dict(city=city,model=name,n_respondents=len(ids),n_any_newly_enabled=int(anyopen.sum()),n_new_car=int((~car).sum()),n_new_bike=int((~bike).sum()),
                mean_newly_enabled_mass=float(mass.mean()),mean_newly_enabled_mass_affected=float(mass[anyopen].mean()),newly_enabled_winner_fraction=float(newwinner.mean()),
                accuracy_full=float((winner[valid]==y[valid]).mean()),accuracy_ownership=float((o.argmax(2)[valid]==y[valid]).mean()),conditional_renormalization_max_abs_error=renormerror,
                newly_enabled_winner_error_fraction=float((newwinner&valid&(winner!=y)).sum()/valid.sum())))
            if name=='s9':
                for c0,b0 in [(False,False),(False,True),(True,False),(True,True)]:
                    keep=(car==c0)&(bike==b0)
                    if not keep.any():continue
                    v=valid[keep];yp=y[keep];ff=f[keep];oo=o[keep];ww=newwinner[keep]
                    group_rows.append(dict(city=city,private_car_available=c0,bicycle_available=b0,n_respondents=int(keep.sum()),n_choices=int(v.sum()),
                        n_human_choice_outside_ownership=int((~oldmask[keep][np.arange(keep.sum())[:,None],np.arange(10)[None,:],np.maximum(yp,0)]&v).sum()),
                        mean_new_mass=float(mass[keep].mean()),new_winner_rate=float(ww[v].mean()),accuracy_full=float((ff.argmax(2)[v]==yp[v]).mean()),accuracy_ownership=float((oo.argmax(2)[v]==yp[v]).mean()),
                        brier_full=float(((ff-np.eye(4)[np.maximum(yp,0)])**2).sum(2)[v].mean()),brier_ownership=float(((oo-np.eye(4)[np.maximum(yp,0)])**2).sum(2)[v].mean()),
                        human_pt=float((yp[v]==1).mean()),full_pt=float(ff[:,:,1][v].mean()),ownership_pt=float(oo[:,:,1][v].mean())))
        unknown=collections.Counter()
        for s in ss:
            for name in GLOBAL_CAT:
                if str(s9.extractor._get_cat(s,name)) not in s9.extractor.cat_vocabs[name]:unknown[name+':'+str(s9.extractor._get_cat(s,name))]+=1
        components=[]
        for s in ss:
            pt=next(a for a in s.alternatives if a.mode=='pt')
            components.append(pt.travel_time_min-sum(getattr(pt,k) for k in ['access_time_min','egress_time_min','wait_time_min','in_vehicle_time_min','transfer_time_min']))
        feature_audit[city]=dict(unknown_category_endpoint_counts=dict(unknown),max_abs_pt_total_minus_component_sum=float(np.max(np.abs(components))),
            nonzero_pt_component_total_counts=int((abs(np.array(components))>1e-5).sum()),
            money_unit='SGD card values' if city=='Singapore' else 'CNY card/reference values; no conversion in current adapter',
            currency_feature_present=False,city_feature_present=False,
            numeric_profiles='Existing answer-independent reference sensitivities retained; no new monetary conversion fit to choices.')
        # Current Teacher outputs are read only; describe support violation mass.
        tr=[r for r in rows(ROOT/'outputs/revision_20260921/teacher/aggregates.jsonl') if r['city']==city and r.get('repeats',0)>=3]
        vals=[]
        for r in tr:
            st=lookup[r['respondent_id'],r['task']]['state'];per=st['persona'];m=np.array([per['car_ownership'] and per['driving_license'],True,per['bike_ownership'],True]);pp=np.array(r['mean_probabilities'])
            vals.append(dict(pt=float(pp[1]),newmass=float(pp[~m].sum())))
        teacher_audit[city]=dict(n_complete_states=len(vals),n_complete_respondents=len({r['respondent_id'] for r in tr}),mean_pt=float(np.mean([x['pt'] for x in vals])) if vals else None,
            mean_newly_enabled_mass=float(np.mean([x['newmass'] for x in vals])) if vals else None,
            caveat='Partial current cohort snapshot, not a completed inferential comparison. Current generic teacher_v0.1 prompt omits S8 accessibility semantics, currency unit and explicit driving/taxi composite instruction.')
    csvout(DEST/'model_support_expansion.csv',model_rows);csvout(DEST/'s9_support_groups.csv',group_rows)
    # Actual primary versus source PT numeric ranges (mode-specific, not mixed-mode z-scores).
    rr=rows(ROOT/'outputs/matched_response_v1/bundle/train_endpoints.jsonl');ranges=[]
    for mode in MODES:
        for field in ['travel_time_min','monetary_cost','access_time_min','coverage_ratio']:
            vals=np.array([next(a for a in r['state']['alternatives'] if a['mode']==mode).get(field,0) for r in rr if next(a for a in r['state']['alternatives'] if a['mode']==mode)['available']])
            ranges.append(dict(mode=mode,field=field,training_n=len(vals),min=float(vals.min()),p05=float(np.quantile(vals,.05)),median=float(np.median(vals)),p95=float(np.quantile(vals,.95)),max=float(vals.max())))
    csvout(DEST/'training_mode_feature_ranges.csv',ranges)
    prompt=Path(ROOT/'outputs/revision_20260921/teacher/system_prompt.txt').read_text(encoding='utf-8')
    from traveler_distillation.teacher.prompts_s8 import S8_SYSTEM_PROMPT
    prompt_audit=dict(matches_accessibility_S8_prompt=prompt.strip()==S8_SYSTEM_PROMPT.strip(),has_accessibility_paragraph='PUBLIC TRANSPORT (pt) ALTERNATIVE' in prompt,
        current_declared_version=readj(ROOT/'outputs/revision_20260921/teacher/config.json')['prompt_version'],S9_accessibility_labeling_version='teacher_s8_accessibility_v0.1',
        code_provenance='scripts/label_s8_teacher.py uses S8_SYSTEM_PROMPT and build_s8_user_prompt',
        interpretation='Current teacher predictions are a generic-prompt diagnostic, not an exact repeat of the S9 accessibility Teacher. Same-generic-prompt ownership comparison isolates mask; S8-prompt comparison must be separately versioned.')
    writej(DEST/'diagnostics.json',dict(architecture=direct,feature_audit=feature_audit,current_teacher_snapshot=teacher_audit,prompt_provenance=prompt_audit,
        sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/traveler_distillation/student/model.py',ROOT/'src/traveler_distillation/student/features.py',ROOT/'src/traveler_distillation/generators/alternative_generator.py',ROOT/'src/traveler_distillation/accessibility/accessibility_dataset.py',ROOT/'outputs/revision_20260921/teacher/system_prompt.txt']},code_sha256=sha(__file__)))
    print(json.dumps(dict(training=bundles['train'],s9=[r for r in model_rows if r['model']=='s9'],groups=group_rows,features=feature_audit,direct=direct,teacher=teacher_audit),indent=2))

if __name__=='__main__':main()
