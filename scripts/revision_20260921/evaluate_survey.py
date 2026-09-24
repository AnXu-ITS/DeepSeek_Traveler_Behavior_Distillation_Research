"""Uniform answer-blind questionnaire inference and respondent-cluster evaluation.

Every measured choice is scored, including labels outside an ownership mask.
An unavailable true class has infinite raw NLL; clipped NLL is a separate diagnostic.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse, copy, csv, json, sys
import numpy as np
import torch
from prepare_survey import ROOT,OUT,SG,SH,ORDER,MODES,readj,rows,writej,writejl,sha,normalise_state
sys.path[:0]=[str(ROOT/'cvpr_workspace/analysis/statistics')]
from reviewer_closure import LinearUtility
from reference_pipeline.student_adapter import StudentAdapter
from traveler_distillation.schemas.state import UniversalTravelerState
from survey_alignment_lib import map_state

VARIANTS=['soft_kl','ce_kl','signed_l1','direction_magnitude']
SEEDS=[42,2026,7]
MODELS=['s9','mnl_s']+[f'{v}_seed{s}' for v in VARIANTS for s in SEEDS]

def model_path(name):
    if name=='s9': return ROOT/'releases/s9_supply_aware_v2/checkpoint/model.pt'
    if name=='mnl_s': return ROOT/'outputs/reviewer_closure_20260920/baseline/model.npz'
    return ROOT/f'outputs/matched_response_v1/train/{name}/best.pt'

def load_model(name):
    return LinearUtility().load() if name=='mnl_s' else StudentAdapter(model_path(name),'cpu')

def predict(model,name,states):
    decisions=model.decisions(states) if name=='mnl_s' else model.predict(states,batch_size=512)
    P=np.array([[d['mode_probabilities'].get(m,0.) for m in MODES] for d in decisions])
    D=np.array([d['departure_time_shift_min'] for d in decisions])
    assert np.isfinite(P).all() and (P>=0).all() and np.max(abs(P.sum(1)-1))<2e-6
    return P,D

def profile_states(city,profile,primary):
    result=copy.deepcopy(primary)
    if city=='Singapore' and profile not in ['survey_options','ownership','car_relaxed']:
        cfg=readj(SG/'mapping_config.json'); cfg['tasks']={int(k):v for k,v in cfg['tasks'].items()}
        people={p['respondent_id']:p for p in rows(SG/'respondents.jsonl')}
        for r in result: r['state']=normalise_state(map_state(people[r['respondent_id']],int(r['task']),cfg,profile))
    for r in result:
        st=r['state']; p=st['persona']
        for a in st['alternatives']:
            if profile=='ownership':
                a['available']=p['car_ownership'] and p['driving_license'] if a['mode']=='car' else p['bike_ownership'] if a['mode']=='bike' else True
            if profile=='car_relaxed': a['available']=p['bike_ownership'] if a['mode']=='bike' else True
        if city=='Shanghai':
            if profile.startswith('household_'): p['household_size']=int(profile.split('_')[1])
            if profile.startswith('coverage_'):
                for a in st['alternatives']:
                    if a['mode']=='pt': a['coverage_ratio']=float(profile.split('_')[1])
            if profile.startswith('times_'):
                factor=float(profile.split('_')[1])
                st['context']['transit_delay_min']*=factor
                for a in st['alternatives']:
                    for field in ['travel_time_min','access_time_min','egress_time_min','wait_time_min','in_vehicle_time_min','transfer_time_min','reliability_delay_min']:
                        a[field]*=factor
            if profile in ['pt_wait_more','pt_walk_more']:
                for a in st['alternatives']:
                    if a['mode']=='pt':
                        # Unknown decomposition, fixed total and original intervention deltas.
                        delta=min(3.,a['in_vehicle_time_min'])
                        a['in_vehicle_time_min']-=delta
                        a['wait_time_min' if profile=='pt_wait_more' else 'access_time_min']+=delta
        r['profile']=profile
    return result

def make_grid(city,records,labs):
    ids=sorted({r['respondent_id'] for r in records}); tasks=[str(i) for i in range(1,11)] if city=='Singapore' else ORDER
    lookup={(r['respondent_id'],r['task']):r for r in records}
    ordered=[lookup[r,t] for r in ids for t in tasks]
    y=np.array([[MODES.index(labs[city,r,t]['chosen_mode']) if labs[city,r,t]['choice_status']=='selected' else -1 for t in tasks] for r in ids])
    return ids,tasks,ordered,y

class Scorer:
    def __init__(self,city,ids,tasks,y,resamples=10000):
        self.city=city;self.ids=ids;self.tasks=tasks;self.y=y;self.valid=y>=0
        self.Y=np.eye(4)[np.maximum(y,0)];self.Y[y<0]=0
        self.spec=readj(OUT/'contrasts.json')[city];self.family=len(self.spec)
        path=OUT/f'bootstrap_{city.lower()}.npz'
        rng=np.random.default_rng(917)
        self.idx=rng.integers(len(ids),size=(resamples,len(ids)),dtype=np.int32)
        # Common resamples are stored once and reused across every model/profile.
        if not path.exists(): np.savez_compressed(path,indices=self.idx,respondents=np.array(ids))
        self.W=np.stack([np.bincount(a,minlength=len(ids)) for a in self.idx]).astype(float)
        self.conditions=[]
        for name,spec in self.spec.items():
            taskidx=[tasks.index(t) for t in spec['weights']]
            weights=np.array(list(spec['weights'].values()))
            modes=[MODES.index(m) for m in spec['modes']]
            keep=self.valid[:,taskidx].all(1)
            h=(self.Y[:,taskidx,:][:,:,modes].sum(2)*weights).sum(1)
            self.conditions.append((name,taskidx,weights,modes,keep,h))

    def interval(self,x,keep=None):
        x=np.asarray(x,float)
        keep=np.isfinite(x) if keep is None else keep & np.isfinite(x)
        den=self.W@keep.astype(float)
        boot=(self.W@np.where(keep,x,0.))/den
        return dict(mean=float(x[keep].mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                    ci_family=np.quantile(boot,[.05/(2*self.family),1-.05/(2*self.family)]).tolist(),n=int(keep.sum()))

    def score(self,P,mask=None,ci=True):
        v=self.valid; pp=P[v]; yy=self.y[v]; Y=self.Y[v]
        chosen=pp.argmax(1); acc=(P.argmax(2)==self.y)
        truep=pp[np.arange(len(yy)),yy]
        cm=np.zeros((4,4),dtype=int); np.add.at(cm,(yy,chosen),1)
        recalls=np.divide(cm.diagonal(),cm.sum(1),out=np.full(4,np.nan),where=cm.sum(1)>0)
        clipped=-np.log(np.maximum(truep,1e-8)); brier=((P-self.Y)**2).sum(2)
        structural=int((~mask[v][np.arange(len(yy)),yy]).sum()) if mask is not None else 0
        result=dict(n_respondents=len(self.ids),n_choices=int(v.sum()),n_unscored=int((~v).sum()),unscored_policy='No mode label for raw none-suitable response; explicit masked choices remain scored',
            accuracy=float(acc[v].mean()),balanced_accuracy=float(np.nanmean(recalls)),
            recall={m:float(r) if np.isfinite(r) else None for m,r in zip(MODES,recalls)},confusion_matrix=cm.tolist(),
            majority_class=MODES[int(np.bincount(yy,minlength=4).argmax())],majority_accuracy=float(np.bincount(yy,minlength=4).max()/len(yy)),
            human_shares=Y.mean(0).tolist(),mean_probabilities=pp.mean(0).tolist(),probability_share_bias=(pp.mean(0)-Y.mean(0)).tolist(),
            nll_raw='Infinity' if (truep==0).any() else float(-np.log(truep).mean()),nll_clipped=float(clipped.mean()),
            epsilon=1e-8,structural_zero_labels=structural,total_zero_probability_labels=int((truep==0).sum()),
            epsilon_clipped_labels=int((truep<1e-8).sum()),brier=float(brier[v].mean()))
        if ci:
            # Equal-respondent means, separately named from choice-weighted point metrics.
            means=lambda x: np.sum(np.where(v,x,0.),axis=1)/v.sum(1)
            result['respondent_intervals']={name:self.interval(means(x)) for name,x in dict(accuracy=acc,brier=brier,pt_share_bias=P[:,:,1]-self.Y[:,:,1]).items()}
        result['responses']={}
        for name,tt,w,m,keep,h in self.conditions:
            pred=(P[:,tt,:][:,:,m].sum(2)*w).sum(1)
            one=dict(n=int(keep.sum()),human=float(h[keep].mean()),model=float(pred[keep].mean()),bias=float((pred-h)[keep].mean()),weights=self.spec[name]['weights'],modes=self.spec[name]['modes'])
            if ci: one.update(human_interval=self.interval(h,keep),model_interval=self.interval(pred,keep),bias_interval=self.interval(pred-h,keep))
            result['responses'][name]=one
        result['mean_absolute_response_bias']=float(np.mean([abs(r['bias']) for r in result['responses'].values()]))
        result['per_task']={task:dict(n=int(v[:,i].sum()),accuracy=float(acc[v[:,i],i].mean()),brier=float(brier[v[:,i],i].mean()),
            human_share=self.Y[v[:,i],i].mean(0).tolist(),model_share=P[v[:,i],i].mean(0).tolist()) for i,task in enumerate(self.tasks)}
        return result

def csvout(path,data):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=MODELS);ap.add_argument('--primary-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2)
    primary=rows(OUT/'task_states.jsonl');labs={(r['city'],r['respondent_id'],r['task']):r for r in rows(OUT/'human_labels.jsonl')}
    cfg=readj(SG/'mapping_config.json')
    models={n:load_model(n) for n in args.models}
    allscores=[];allresponses=[]
    for city in ['Singapore','Shanghai']:
        citystates=[r for r in primary if r['city']==city]
        ids,tasks,_,y=make_grid(city,citystates,labs)
        scorer=Scorer(city,ids,tasks,y)
        profiles=['survey_options','ownership','car_relaxed']
        if not args.primary_only: profiles+=list(cfg['profiles'])[1:] if city=='Singapore' else ['household_1','household_4','coverage_0.5','coverage_1.0','times_0.8','times_1.2','pt_wait_more','pt_walk_more']
        writej(OUT/f'grid_{city.lower()}.json',dict(respondents=ids,tasks=tasks,labels=y.tolist(),mode_order=MODES))
        for profile in profiles:
            records=profile_states(city,profile,citystates)
            _,_,ordered,_=make_grid(city,records,labs)
            states=[UniversalTravelerState.model_validate(r['state']) for r in ordered]
            mask=np.array([[a.available for a in s.alternatives] for s in states]).reshape(len(ids),10,4)
            for name in args.models:
                d=OUT/'predictions'/city.lower()/profile;d.mkdir(parents=True,exist_ok=True)
                predpath=d/f'{name}.npz'; scorepath=d/f'{name}.json'
                if predpath.exists():
                    z=np.load(predpath);P=z['probabilities'];D=z['departure']
                else:
                    P,D=predict(models[name],name,states);P=P.reshape(len(ids),10,4);D=D.reshape(len(ids),10)
                    np.savez_compressed(predpath,probabilities=P,departure=D,availability=mask)
                if scorepath.exists(): score=readj(scorepath)
                else:
                    score=scorer.score(P,mask,ci=profile in ['survey_options','ownership','car_relaxed'])
                    score.update(city=city,profile=profile,model=name,checkpoint_sha256=sha(model_path(name)),mode_order=MODES,bootstrap=dict(seed=917,resamples=10000,unit='respondent, full task set',family=scorer.family,shared_across_models=True))
                    # Source features define groups; never group respondents using their chosen answer.
                    score['subgroups']={}
                    for field in ['age_group','income_group','habitual_mode','car_ownership','bike_ownership']:
                        values=[states[i*10].persona.model_dump()[field] for i in range(len(ids))]
                        score['subgroups'][field]={}
                        for val in sorted(set(values),key=str):
                            keep=np.array([x==val for x in values]);valid=y[keep]>=0; yp=y[keep];one=np.eye(4)[np.maximum(yp,0)];p=P[keep]
                            score['subgroups'][field][str(val)]=dict(n_respondents=int(keep.sum()),n_choices=int(valid.sum()),accuracy=float((p.argmax(2)[valid]==yp[valid]).mean()),brier=float(((p-one)**2).sum(2)[valid].mean()),pt_bias=float((p[:,:,1]-one[:,:,1])[valid].mean()))
                    writej(scorepath,score)
                allscores.append({k:score[k] for k in ['city','profile','model','n_respondents','n_choices','accuracy','balanced_accuracy','majority_accuracy','nll_raw','nll_clipped','structural_zero_labels','brier','mean_absolute_response_bias']}|dict(pt_bias=score['probability_share_bias'][1]))
                for contrast,r in score['responses'].items():
                    allresponses.append(dict(city=city,profile=profile,model=name,contrast=contrast,n=r['n'],human=r['human'],predicted=r['model'],bias=r['bias'],
                        ci95_low=r.get('bias_interval',{}).get('ci95',[None,None])[0],ci95_high=r.get('bias_interval',{}).get('ci95',[None,None])[1],
                        family_low=r.get('bias_interval',{}).get('ci_family',[None,None])[0],family_high=r.get('bias_interval',{}).get('ci_family',[None,None])[1]))
                print(city,profile,name,'accuracy',round(score['accuracy'],4),'Brier',round(score['brier'],4),'response_bias',round(score['mean_absolute_response_bias'],4),flush=True)
    csvout(OUT/'model_scores.csv',allscores);csvout(OUT/'model_responses.csv',allresponses)
    writej(OUT/'evaluation_manifest.json',dict(models={n:sha(model_path(n)) for n in args.models},task_states_sha256=sha(OUT/'task_states.jsonl'),human_labels_sha256=sha(OUT/'human_labels.jsonl'),code_sha256=sha(__file__),n_model_profile_city_results=len(allscores)))

if __name__=='__main__':main()
