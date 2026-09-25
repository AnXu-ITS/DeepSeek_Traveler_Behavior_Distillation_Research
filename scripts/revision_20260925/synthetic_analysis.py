"""Same-state new-Teacher comparisons and prospective formal sample sizing."""
import collections,copy,math
from synthetic_teacher import *
import torch
from traveler_distillation.matched_response.metrics import encode,predict
from traveler_distillation.matched_response.experiment import TravelerStudentS8,S8FeatureExtractor
from traveler_distillation.matched_response.data import endpoint

def evaluate(cohort='pilot'):
    status=read_json(DEST/cohort/'status.json');assert status['complete'],'No incomplete-person or partial-pilot analysis'
    calls={}
    for p in sorted((DEST/cohort).glob('attempt_*.json')):
        r=read_json(p)
        if r.get('valid'):calls.setdefault((r['state_id'],r['repeat']),r)
    states=[s for s in rows(DEST/'states.jsonl') if s['cohort']==cohort and (s['id'],0) in calls];eps=[]
    for s in states:
        actions=[calls[s['id'],k]['action'] for k in range(3)];modes=['car','pt','bike','walk'];probs={m:np.mean([a['mode_probabilities'].get(m,0.) for a in actions]) for m in modes}
        eps.append(endpoint(s['id'],'synthetic',cohort,s['state'],probs,np.mean([a['departure_time_shift_min'] for a in actions]),bucket=s['card'],original_id=s['id']))
    assert all(sum(e['persona']==p for e in eps)==12 for p in {e['persona'] for e in eps})
    torch.set_num_threads(2);unit=[];allpred=[]
    checkpoint_sets=[('full',OUT),('delay_holdout',OUT.parent/'delay_family/train')]
    for family,folder in checkpoint_sets:
        variants=[('soft_kl',0),('signed_l1',1)]
        if family=='full':variants.append(('direction_magnitude',1))
        for v,w in variants:
            for seed in [42,2026,7]:
                for sel in ['static','response']:
                    path=folder/f'{v}_w{w}_seed{seed}'/f'best_{sel}.pt';ck=torch.load(path,map_location='cpu',weights_only=False);ext=S8FeatureExtractor.from_state_dict(ck['extractor_state']);model=TravelerStudentS8(ck['config'],ext.spec);model.load_state_dict(ck['model_state'])
                    pred=predict(model,eps,{e['id']:encode(e,ext) for e in eps},'cpu',128);by={p['id']:p for p in pred}
                    for p in pred:
                        allpred.append(dict(family=family,variant=v,seed=seed,selection=sel,checkpoint_sha256=file_hash(path),**p))
                        if p['bucket']=='baseline':continue
                        b=by[p['persona']+':baseline'];mask=np.array(p['mask'])|np.array(b['mask']);dt=np.array(p['teacher'])-b['teacher'];ds=np.array(p['student'])-b['student']
                        unit.append(dict(family=family,variant=v,seed=seed,selection=sel,persona=p['persona'],card=p['bucket'],response_gap=float(abs(dt-ds)[mask].mean())))
    # Descriptive baseline, selected only on historical validation data.
    from mnl_selection import LinearUtility,predictions
    mnl=LinearUtility();X,M=mnl.design([e['state'] for e in eps])
    for sel,choice in read_json(OUT.parent/'mnl_selection/selection.json').items():
        path=OUT.parent/f'mnl_selection/candidate_{choice["l2"]:g}.npz'
        ck=np.load(path);mnl.coef=ck['coef'];mnl.departure=ck['departure']
        P,D=mnl.predict_arrays(X,M);pr=predictions(eps,P,D);by={r['id']:r for r in pr}
        for r in pr:
            allpred.append(dict(family='full',variant='mnl',seed=None,selection=sel,checkpoint_sha256=file_hash(path),**r))
            if r['bucket']=='baseline':continue
            b=by[r['persona']+':baseline'];mask=np.array(r['mask'])|np.array(b['mask'])
            dt=np.array(r['teacher'])-b['teacher'];ds=np.array(r['student'])-b['student']
            unit.append(dict(family='full',variant='mnl',seed=None,selection=sel,persona=r['persona'],card=r['bucket'],response_gap=float(abs(dt-ds)[mask].mean())))
    write_rows(DEST/cohort/'predictions.jsonl',allpred);write_rows(DEST/cohort/'pair_metrics.jsonl',unit)
    primary=[]
    for pid in sorted({r['persona'] for r in unit}):
        sub=[r for r in unit if r['persona']==pid and r['family']=='full' and r['selection']=='static'];diff=np.mean([r['response_gap'] for r in sub if r['variant']=='signed_l1'])-np.mean([r['response_gap'] for r in sub if r['variant']=='soft_kl']);primary.append(dict(persona=pid,signed_minus_softkl=float(diff)))
    x=np.array([r['signed_minus_softkl'] for r in primary]);rng=np.random.default_rng(20260925);boot=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    result=dict(cohort=cohort,n=len(x),primary=primary,mean=float(x.mean()),sd=float(x.std(ddof=1)),ci95=np.quantile(boot,[.025,.975]).tolist(),claim='Cross-model agreement against Go Flash reference; not original Pro distillation error or human validity')
    if cohort=='pilot':
        n=max(30,math.ceil((1.96*x.std(ddof=1)/.01)**2));result.update(formal_n_required=n,formal_n_budget_max=120,precision_halfwidth=.01,formal_authorized_to_run=n<=120,pilot_excluded_from_formal=True)
    write_json(DEST/cohort/'analysis.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    import sys
    evaluate(sys.argv[1] if len(sys.argv)>1 else 'pilot')
