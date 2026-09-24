"""Executed source, loss mathematics, six-persona and Teacher-repeat analyses."""
from pathlib import Path
import sys,json,collections,itertools,copy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from departure_baselines import ROOT,OUT,BUNDLE,OLD,SEEDS,MAIN_VARIANTS,MODES,read,rows,write,write_rows,csvout,sha
import numpy as np
import torch
from traveler_distillation.student.losses import elasticity_direction_loss,elasticity_magnitude_loss,kl_divergence
from traveler_distillation.matched_response.experiment import VARIANTS
from traveler_distillation.matched_response.metrics import paired_cluster_difference

def objective_audit():
    records=[]; schedules={}; budgets={};inits={};protocols=[]
    for seed in SEEDS:
        for variant in MAIN_VARIANTS:
            folder=OLD/f'train/{variant}_seed{seed}';r=read(folder/'run.json');st=read(folder/'status.json');history=rows(folder/'history.jsonl')
            schedules[variant,seed]=[x['schedule_hash'] for x in history];budgets[variant,seed]=r['training_budget'];inits[variant,seed]=r['initial_model_hash'];protocols.append(r['protocol_id'])
            records.append(dict(variant=variant,seed=seed,initialization='random',endpoint='soft KL + CE' if VARIANTS[variant]['ce'] else 'soft KL',response=variant if variant not in ['soft_kl','ce_kl'] else 'none',departure='Huber(delta=1), weight=1',auxiliary_losses='none',response_mask='union',endpoint_normalization='sum of endpoint losses per unit; mean over batch units',response_normalization='signed/magnitude: available modes; direction: informative available modes',optimizer_steps=st['optimizer_steps'],epochs=st['epochs'],checkpoint='minimum validation macro-source KL',best_epoch=st['best_epoch']))
    checks=dict(common_protocol=len(set(protocols))==1,same_initialization_within_seed=all(len({inits[v,s] for v in MAIN_VARIANTS})==1 for s in SEEDS),same_schedule_within_seed=all(all(schedules[v,s]==schedules['soft_kl',s] for v in MAIN_VARIANTS) for s in SEEDS),same_endpoint_exposure=all(b==budgets['soft_kl',42] for b in budgets.values()),all_6360_updates=all(r['optimizer_steps']==6360 for r in records),paired_methods_overlay_soft_KL=all(VARIANTS[v]['ce']==0 for v in ['signed_l1','direction_magnitude']))
    assert all(checks.values()),checks
    csvout(OUT/'controlled_configurations.csv',records)
    write(OUT/'objective_audit.json',dict(checks=checks,correction_retraining_required=False,reason='Actual implementation and archived runs use the intended common endpoints, masks, schedules and KL objective; only the intended loss coefficients differ.',s9_historical_configuration=read(ROOT/'releases/s9_supply_aware_v2/config/training_s9.json'),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/traveler_distillation/matched_response/experiment.py',ROOT/'src/traveler_distillation/student/losses.py',ROOT/'configs/matched_response.yaml']}))

def mathematics():
    torch.set_num_threads(2);torch.manual_seed(917)
    examples=[]
    for target in [[.6,.25,.15],[.31,.29,.23,.17],[.85,.1,.05]]:
        t=torch.tensor(target,dtype=torch.float64); z=torch.randn(len(t),dtype=torch.float64,requires_grad=True)
        opt=torch.optim.LBFGS([z],lr=1.,max_iter=200,tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad();p=z.softmax(-1);loss=-(p[t.argmax()].log())+(t*(t.log()-p.log())).sum();loss.backward();return loss
        opt.step(closure);p=z.softmax(-1).detach();ideal=(t+torch.nn.functional.one_hot(t.argmax(),len(t)))/2
        err=float((p-ideal).abs().max());assert err<1e-6
        examples.append(dict(teacher=target,numerical_optimum=p.tolist(),analytic_optimum=ideal.tolist(),max_absolute_error=err))
    scalar=[]
    for sval in [-.199,-.15,-.1,-.05,-.001]:
        s=torch.tensor([sval],dtype=torch.float64,requires_grad=True);t=torch.tensor([.2],dtype=torch.float64)
        loss=torch.clamp(-s,min=0)+(t-s.abs()).abs();loss.sum().backward()
        scalar.append(dict(teacher=.2,student=sval,loss=float(loss.detach()),gradient=float(s.grad)))
    assert max(abs(r['loss']-.2) for r in scalar)<1e-12 and max(abs(r['gradient']) for r in scalar)==0
    legal=[]
    # Baseline [.4,.3,.3], Teacher intervention [.6,.2,.2], valid student intervention
    # [.3,.35,.35] moves all modes opposite; direction/magnitude cancel gradients.
    for teacher,student in [([.2,-.1,-.1],[-.1,.05,.05]),([.2,-.2,0.],[-.1,.1,0.])]:
        t=torch.tensor([teacher],dtype=torch.float64);s=torch.tensor([student],dtype=torch.float64,requires_grad=True);m=torch.ones_like(t)
        ld=elasticity_direction_loss(t,s,m);lm=elasticity_magnitude_loss(t,s,m);(ld+lm).sum().backward()
        legal.append(dict(teacher_delta=teacher,student_delta=student,baseline_probability=[.4,.3,.3],teacher_probability=(np.array([.4,.3,.3])+teacher).tolist(),student_probability=(np.array([.4,.3,.3])+student).tolist(),informative_modes=int((t.abs()>1e-6).sum()),available_modes=3,direction=float(ld.detach()),magnitude=float(lm.detach()),gradient=s.grad[0].tolist()))
    assert max(abs(x) for x in legal[0]['gradient'])<1e-12
    pair_rows=rows(OLD/'test/soft_kl_seed42/pairs.jsonl'); pred={r['id']:r for r in rows(OLD/'test/soft_kl_seed42/predictions.jsonl')}
    counts=collections.Counter()
    for r in pair_rows:
        a,b=pred[r['base']],pred[r['cf']];mask=np.array(a['mask'])|np.array(b['mask']);td=np.array(r['teacher_delta']);counts['all_pairs']+=1
        counts['same_direction_and_magnitude_denominator']+=int(((abs(td)>1e-6)&mask).sum()==mask.sum())
    empirical=[]
    for seed in SEEDS:
        ps={v:rows(OLD/f'test/{v}_seed{seed}/predictions.jsonl') for v in ['soft_kl','ce_kl']}
        pairs={v:rows(OLD/f'test/{v}_seed{seed}/pairs.jsonl') for v in ['soft_kl','ce_kl']}
        for v in ps:
            entropy=lambda a:float(-(np.asarray(a)*np.log(np.clip(a,1e-12,1))).sum())
            same=[r for r in pairs[v] if np.argmax(pred[r['base']]['teacher'])==np.argmax(pred[r['cf']]['teacher'])]
            empirical.append(dict(seed=seed,model=v,mean_max_probability=float(np.mean([max(r['student']) for r in ps[v]])),mean_entropy=float(np.mean([entropy(r['student']) for r in ps[v]])),teacher_mean_entropy=float(np.mean([entropy(r['teacher']) for r in ps[v]])),same_teacher_argmax_pairs=len(same),same_argmax_response_magnitude=float(np.mean([np.abs(r['student_delta']).sum() for r in same])),same_argmax_teacher_response_magnitude=float(np.mean([np.abs(r['teacher_delta']).sum() for r in same])),same_argmax_response_gap=float(np.mean([r['response_gap'] for r in same]))))
    csvout(OUT/'ce_sharpening_empirical.csv',empirical)
    write(OUT/'loss_mathematics.json',dict(CE_KL_optimum=examples,scalar_flat_interval=scalar,legal_probability_examples=legal,real_test_mask_counts=dict(counts),interpretation='CE+KL free-state optimum sharpens at Teacher argmax; unchanged argmax endpoints have exactly half the ideal probability difference. Actual networks also share parameters, optimize departure and can add pair losses, so half-response is not asserted for trained networks. Direction uses informative-mode denominator and magnitude uses available-mode denominator: exact cancellation requires equal effective denominator/weights. When quiet modes exist, these denominators differ and the opposite-direction interval need not be flat. Scalar and probability-vector tests use interior points to avoid nondifferentiable boundaries.'))

def sensitivity():
    predictions={(v,s):rows(OLD/f'test/{v}_seed{s}/predictions.jsonl') for v,s in itertools.product(MAIN_VARIANTS,SEEDS)}
    pairs={(v,s):rows(OLD/f'test/{v}_seed{s}/pairs.jsonl') for v,s in itertools.product(MAIN_VARIANTS,SEEDS)}
    ints={(v,s):rows(OLD/f'test/{v}_seed{s}/interactions.jsonl') for v,s in itertools.product(MAIN_VARIANTS,SEEDS)}
    breakdown=[];leave=[];paired=[]
    for (v,s),rs in pairs.items():
        for field in ['persona','source','contrast_type','bucket']:
            for group in sorted({r.get(field,'unspecified') for r in rs}):
                sub=[r for r in rs if r.get(field,'unspecified')==group]
                breakdown.append(dict(model=v,seed=s,field=field,group=group,n_pairs=len(sub),response_gap=float(np.mean([r['response_gap'] for r in sub]))))
        if v=='soft_kl':continue
        ref={r['id']:r for r in pairs['soft_kl',s]}
        for excluded in sorted({r['persona'] for r in rs}):
            ss=[r for r in rs if r['persona']!=excluded]
            delta=np.mean([r['response_gap']-ref[r['id']]['response_gap'] for r in ss])
            leave.append(dict(model=v,seed=s,excluded_persona=excluded,n_pairs=len(ss),response_gap_difference=float(delta)))
        for key,pool in [('kl',predictions),('departure_mae',predictions),('response_gap',pairs),('interaction_gap',ints)]:
            dif=paired_cluster_difference(pool[v,s],pool['soft_kl',s],key,10000,917)
            paired.append(dict(model=v,seed=s,metric=key,difference=dif['mean'],ci_low=dif['ci'][0],ci_high=dif['ci'][1],n_units=dif['n'],n_personas=dif['n_personas']))
    csvout(OUT/'breakdown_response.csv',breakdown);csvout(OUT/'leave_one_persona_out.csv',leave);csvout(OUT/'paired_seed_differences.csv',paired)
    summary={}
    for v in MAIN_VARIANTS[1:]:
        ls=[r for r in leave if r['model']==v];seedmean=[np.mean([r['response_gap_difference'] for r in ls if r['excluded_persona']==g]) for g in sorted({r['excluded_persona'] for r in ls})]
        summary[v]=dict(seed_mean_leave_one_out_difference_range=[float(min(seedmean)),float(max(seedmean))],seed_mean_improves_all_six_removals=bool(max(seedmean)<0),individual_seed_removal_improved=sum(r['response_gap_difference']<0 for r in ls),individual_seed_removal_n=len(ls))
    write(OUT/'robustness_summary.json',summary)

def teacher_repeats():
    test=rows(BUNDLE/'test_endpoints.jsonl');pairs=rows(BUNDLE/'test_pairs.jsonl')
    paths={'legacy':('data/student_v0_3_s3/aggregated_teacher_dataset.jsonl','data/student_v0_3_s3/repeat_records.jsonl'),'joint':('data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl','data/student_s5_joint/repeat_records.jsonl'),'accessibility':('data/singapore_accessibility/states_with_teacher.jsonl','data/singapore_accessibility/repeat_records.jsonl')}
    records={};matched={};counts=[];sourcehash={}
    for source,(ap,rp) in paths.items():
        aggregate={r['sample_id']:r for r in rows(ROOT/ap)};raw={r['completion_id']:r for r in rows(ROOT/rp) if r.get('action') and r.get('completion_id')}
        for path in [ap,rp]:sourcehash[path]=sha(ROOT/path)
        for e in [e for e in test if e['source']==source]:
            a=aggregate[e['original_id']];ids=a['aggregation_metadata']['source_completion_ids'];rs=[raw[i] for i in ids if i in raw]
            if len(rs)!=a['aggregation_metadata']['k']:continue
            mat=np.array([[r['action']['mode_probabilities'].get(m,0.) for m in MODES] for r in rs],float)
            mat/=mat.sum(1,keepdims=True);d=np.array([r['action']['departure_time_shift_min'] for r in rs])
            error=float(abs(mat.mean(0)-e['teacher_probs']).max())
            if error>2e-6:raise ValueError((e['id'],error))
            records[e['id']]=mat;matched[e['id']]=e
            counts.append(dict(id=e['id'],source=source,k=len(rs),max_aggregate_reproduction_error=error,probability_sd_mean=float(mat.std(0,ddof=1).mean()),departure_sd=float(d.std(ddof=1))))
    retained=[r for r in pairs if r['base'] in records and r['cf'] in records]
    ids=sorted(records);idx={e:i for i,e in enumerate(ids)};base=np.array([idx[r['base']] for r in retained]);cf=np.array([idx[r['cf']] for r in retained]);ms=np.array([[a['available'] for a in matched[e]['state']['alternatives']] for e in ids],bool)
    mask=ms[base]|ms[cf];denom=mask.sum(1)
    pred={}
    for v,s in itertools.product(MAIN_VARIANTS,SEEDS):
        by={r['id']:r for r in rows(OLD/f'test/{v}_seed{s}/predictions.jsonl')};p=np.array([by[e]['student'] for e in ids]);pred[v,s]=p[cf]-p[base]
    mnl={r['id']:r for r in rows(ROOT/'outputs/reviewer_closure_20260920/baseline/predictions.jsonl')};p=np.array([mnl[e]['student'] for e in ids]);pred['mnl_s',0]=p[cf]-p[base]
    def evaluate(target):
        td=target[cf]-target[base]
        return {k:float(((np.abs(td-sd)*mask).sum(1)/denom).mean()) for k,sd in pred.items()}
    schemes={}
    schemes['original_mean']=evaluate(np.array([records[e].mean(0) for e in ids]))
    med=np.array([np.median(records[e],axis=0) for e in ids]);med/=med.sum(1,keepdims=True);schemes['coordinate_median_renormalized']=evaluate(med)
    for i in range(3):schemes[f'leave_repeat_{i}_out']=evaluate(np.array([np.delete(records[e],i,axis=0).mean(0) for e in ids]))
    table=[]
    for name,values in schemes.items():
        for (v,s),val in values.items():table.append(dict(aggregation=name,model=v,seed=s,response_gap=val,n_pairs=len(retained)))
    csvout(OUT/'teacher_aggregation_sensitivity.csv',table);csvout(OUT/'teacher_repeat_state_variability.csv',counts)
    rng=np.random.default_rng(917);draws=collections.defaultdict(list)
    for b in range(1000):
        target=np.array([r[rng.integers(len(r),size=len(r))].mean(0) for r in [records[e] for e in ids]])
        val=evaluate(target);ref=np.mean([val['soft_kl',s] for s in SEEDS])
        for v in MAIN_VARIANTS[1:]:draws[v].append(np.mean([val[v,s] for s in SEEDS])-ref)
        draws['mnl_s'].append(val['mnl_s',0]-ref)
    result={v:dict(mean=float(np.mean(x)),target_resampling_95_interval=np.quantile(x,[.025,.975]).tolist(),fraction_lower_than_soft_kl=float(np.mean(np.array(x)<0))) for v,x in draws.items()}
    kcounts=collections.Counter(r['k'] for r in counts)
    variability={str(k):dict(n=sum(r['k']==k for r in counts),mean_probability_sd=float(np.mean([r['probability_sd_mean'] for r in counts if r['k']==k])),mean_departure_sd=float(np.mean([r['departure_sd'] for r in counts if r['k']==k]))) for k in kcounts}
    report=dict(test_endpoints=len(test),repeated_endpoints=len(records),repeated_pairs=len(retained),original_pairs=len(pairs),test_k_counts=dict(kcounts),variability_by_k=variability,max_target_reproduction_error=max(r['max_aggregate_reproduction_error'] for r in counts),target_resampling=result,replicates=1000,seed=917,scope='Conditioned on fixed trained models and existing repeats. Independently resample calls within each endpoint, preserve shared endpoints across all response pairs and models. These are target-aggregation sensitivity intervals, not human or population confidence intervals. Mechanism raw repeated actions are not included because their repeat records are absent from the checked source files; no replacement targets are invented.',source_sha256=sourcehash)
    write(OUT/'teacher_repeat_sensitivity.json',report)

def main():
    objective_audit();mathematics();sensitivity();teacher_repeats()
    print('Completed objective, numerical-loss, persona and Teacher-repeat analyses.',flush=True)
if __name__=='__main__':main()
