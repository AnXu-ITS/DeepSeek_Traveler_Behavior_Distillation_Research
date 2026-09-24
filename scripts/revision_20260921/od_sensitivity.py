"""Full-vector Shanghai supply sensitivity with response-independent OD assignments.

Twenty fresh assignments use the original nearest-6-km rule and candidate pool;
twenty permutations exchange the complete original supply vector and departure.
All ten tasks of a respondent share a baseline. No choices enter assignment.
"""
from pathlib import Path
import argparse,copy,hashlib,json,random,sys,time
import numpy as np
import torch
from prepare_survey import ROOT,OUT,ORDER,MODES,readj,rows,writej,writejl,sha
from evaluate_survey import profile_states,make_grid,Scorer,load_model,predict,csvout
sys.path.insert(0,str(ROOT/'scripts/shanghai'))
from sp_survey_matsim import SUPPLY,network_card_state,assign_od
from reference_pipeline.matsim_adapter import build_supply_view
from traveler_distillation.accessibility.gtfs_accessibility import SupplyIndex,plan_accessibility
from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives
from traveler_distillation.schemas.state import UniversalTravelerState

DEST=OUT/'od_sensitivity'
NET=ROOT/'outputs/shanghai_survey_matsim_321x10_v1'

class OriginalRule:
    def __init__(self,nodes,xy,target_km=6.):
        self.nodes=nodes;self.xy=xy;self.target=target_km
        pts=np.array([xy[n['node']] for n in nodes]);self.candidates=[]
        for i in range(len(nodes)):
            dist=np.linalg.norm(pts-pts[i],axis=1);j=int(np.argmin(abs(dist-target_km*1000)))
            self.candidates.append((abs(float(dist[j])-target_km*1000),i,j,float(dist[j])/1000))
    def assign(self,rid,repeat=None):
        # Repeat=None reproduces the historical hash namespace exactly.
        key=rid if repeat is None else f'{rid}:revision_repeat:{repeat}'
        seed=int(hashlib.sha256(('shanghai-sp-od-20260919:'+key).encode()).hexdigest()[:16],16)
        rng=random.Random(seed)
        best=min([self.candidates[i] for i in rng.sample(range(len(self.nodes)),min(150,len(self.nodes)))],key=lambda t:t[0])
        err,i,j,actual=best
        assert i!=j and err<=100
        return dict(respondent_id=rid,origin_node=self.nodes[i]['node'],destination_node=self.nodes[j]['node'],target_km=self.target,straight_line_km=actual,error_m=err,observed_od=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeats',type=int,default=20);args=ap.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    prim=[r for r in rows(OUT/'task_states.jsonl') if r['city']=='Shanghai']
    labs={(r['city'],r['respondent_id'],r['task']):r for r in rows(OUT/'human_labels.jsonl')}
    ids,tasks,_,y=make_grid('Shanghai',prim,labs);scorer=Scorer('Shanghai',ids,tasks,y)
    template={(r['respondent_id'],r['task']):UniversalTravelerState.model_validate(r['state']) for r in prim}
    oldrows=rows(NET/'network_baseline_states.jsonl');oldbase={r['respondent_id']:UniversalTravelerState.model_validate(r['state']) for r in oldrows}
    oldod={r['respondent_id']:r for r in readj(NET/'od_assignments.json')}
    manifest=DEST/'protocol.json'
    if not manifest.exists(): writej(manifest,dict(repeats=args.repeats,model='frozen S9',profiles=['survey_options','ownership'],
        rule='Original nearest-6-km among 150 random candidate origins. Fresh hash salt per respondent/repeat, no outcome/prediction/feasibility screening.',
        permutation='Complete original baseline alternative vector plus departure, origin and destination exchanged between respondents. Recipient demographics and all original task transformations retained.',
        tasks_per_respondent=10,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [NET/'network_baseline_states.jsonl',NET/'od_assignments.json',ROOT/'scripts/shanghai/sp_survey_matsim.py']},
        raw_questionnaires_modified=False,interpretation='Input-replacement robustness conditional on assigned rather than observed OD, unobserved numerical task attributes, and retained synthetic timetable.'))
    print('Loading unchanged Shanghai supply',flush=True)
    view=build_supply_view(SUPPLY['network'],SUPPLY['activity_nodes']);idx=SupplyIndex(SUPPLY)
    rule=OriginalRule(view[3],view[1])
    for rid in ids[:5]: assert rule.assign(rid)==assign_od(rid,view[3],view[1])
    for rid in ids: assert rule.assign(rid)['origin_node']==oldod[rid]['origin_node'] and rule.assign(rid)['destination_node']==oldod[rid]['destination_node']
    print('Original OD rule exactly reproduced for all respondents',flush=True)
    cache={}
    for rid in ids:
        o=oldod[rid];st=oldbase[rid];key=(o['origin_node'],o['destination_node'],st.trip.desired_departure_min)
        cache[key]=([a.model_dump(mode='json') for a in st.alternatives],None)
    model=load_model('s9');summary=[];responses=[];groups=[]
    basepred={prof:np.load(OUT/f'predictions/shanghai/{prof}/s9.npz')['probabilities'] for prof in ['survey_options','ownership']}
    valid=y>=0; one=np.eye(4)[np.maximum(y,0)]
    for kind,repeat in [('original',0)]+[('permutation',i) for i in range(args.repeats)]+[('reassignment',i) for i in range(args.repeats)]:
        run=f'{kind}_{repeat:02d}';path=DEST/f'{run}_baseline.jsonl'
        if path.exists(): baserows=rows(path)
        elif kind=='original':
            baserows=[dict(respondent_id=rid,donor_id=rid,departure_min=oldbase[rid].trip.desired_departure_min,alternatives=[a.model_dump(mode='json') for a in oldbase[rid].alternatives],**{k:v for k,v in oldod[rid].items() if k!='respondent_id'}) for rid in ids]
            writejl(path,baserows)
        elif kind=='permutation':
            donors=np.random.default_rng(2026092100+repeat).permutation(ids)
            baserows=[]
            for rid,donor in zip(ids,donors):
                st=oldbase[donor];o=oldod[donor]
                baserows.append(dict(respondent_id=rid,donor_id=str(donor),departure_min=st.trip.desired_departure_min,alternatives=[a.model_dump(mode='json') for a in st.alternatives],**{k:v for k,v in o.items() if k!='respondent_id'}))
            writejl(path,baserows)
        else:
            baserows=[]
            for i,rid in enumerate(ids):
                o=rule.assign(rid,repeat);t=template[rid,'B0'];dep=t.trip.desired_departure_min
                key=(o['origin_node'],o['destination_node'],dep)
                if key not in cache:
                    acc=plan_accessibility(idx,o['origin_node'],o['destination_node'],dep*60.)
                    alts=build_real_alternatives(t.persona,t.trip,t.context,idx,o['origin_node'],o['destination_node'],acc)
                    cache[key]=([a.model_dump(mode='json') for a in alts],acc)
                alts,acc=cache[key]
                baserows.append(dict(**o,departure_min=dep,alternatives=alts,accessibility=acc))
            writejl(path,baserows)
        by={r['respondent_id']:r for r in baserows}
        for profile in ['survey_options','ownership']:
            d=DEST/profile;d.mkdir(exist_ok=True)
            predpath=d/f'{run}.npz';scorepath=d/f'{run}.json'
            records=profile_states('Shanghai',profile,prim)
            tpls={(r['respondent_id'],r['task']):UniversalTravelerState.model_validate(r['state']) for r in records}
            states=[]
            for rid in ids:
                b=by[rid];t=tpls[rid,'B0']
                base=t.model_copy(deep=True);base.alternatives=[type(t.alternatives[0]).model_validate(a) for a in b['alternatives']]
                # Moving full supply carries its planning time. Historical rows all use 480 min.
                base.trip.desired_departure_min=b['departure_min']
                for task in tasks:
                    tt=tpls[rid,task].model_copy(deep=True);tt.trip.desired_departure_min=b['departure_min']
                    states.append(network_card_state(base,tt,t))
            mask=np.array([[a.available for a in s.alternatives] for s in states]).reshape(len(ids),10,4)
            if predpath.exists(): P=np.load(predpath)['probabilities']
            else:
                P,D=predict(model,'s9',states);P=P.reshape(len(ids),10,4)
                np.savez_compressed(predpath,probabilities=P,departure=D.reshape(len(ids),10),availability=mask)
            if scorepath.exists(): score=readj(scorepath)
            else:
                score=scorer.score(P,mask,ci=kind=='original')
                pc=basepred[profile];correct=P.argmax(2)==y;oldcorrect=pc.argmax(2)==y
                brier=((P-one)**2).sum(2);oldbrier=((pc-one)**2).sum(2)
                mean=lambda x:np.where(valid,x,0).sum(1)/valid.sum(1)
                score['paired_accuracy_gain']=scorer.interval(mean(correct.astype(float)-oldcorrect)) if kind=='original' else dict(mean=float(mean(correct.astype(float)-oldcorrect).mean()))
                score['paired_brier_change']=scorer.interval(mean(brier-oldbrier)) if kind=='original' else dict(mean=float(mean(brier-oldbrier).mean()))
                eligible=valid & mask[np.arange(len(ids))[:,None],np.arange(10)[None,:],np.maximum(y,0)]
                score['historical_support_scoring']=dict(n=int(eligible.sum()),network_accuracy=float(correct[eligible].mean()),card_accuracy=float(oldcorrect[eligible].mean()))
                score.update(kind=kind,repeat=repeat,profile=profile,unique_od=len({(r['origin_node'],r['destination_node']) for r in baserows}),baseline_file_sha256=sha(path))
                writej(scorepath,score)
                for field in ['age_group','income_group','habitual_mode','car_ownership']:
                    values=[template[rid,'B0'].persona.model_dump()[field] for rid in ids]
                    for val in sorted(set(values),key=str):
                        keep=np.array([v==val for v in values])
                        groups.append(dict(kind=kind,repeat=repeat,profile=profile,field=field,value=val,n=int(keep.sum()),accuracy_gain=float(mean(correct.astype(float)-oldcorrect)[keep].mean()),brier_change=float(mean(brier-oldbrier)[keep].mean())))
                for j,task in enumerate(tasks):
                    keep=valid[:,j];groups.append(dict(kind=kind,repeat=repeat,profile=profile,field='task',value=task,n=int(keep.sum()),accuracy_gain=float((correct.astype(float)-oldcorrect)[keep,j].mean()),brier_change=float((brier-oldbrier)[keep,j].mean())))
            summary.append(dict(kind=kind,repeat=repeat,profile=profile,accuracy=score['accuracy'],brier=score['brier'],pt_bias=score['probability_share_bias'][1],response_bias=score['mean_absolute_response_bias'],accuracy_gain=score['paired_accuracy_gain']['mean'],brier_change=score['paired_brier_change']['mean'],unique_od=score['unique_od']))
            responses.extend(dict(kind=kind,repeat=repeat,profile=profile,contrast=c,human=r['human'],model=r['model'],bias=r['bias'],n=r['n']) for c,r in score['responses'].items())
        print(run,'done; cached OD',len(cache),'acc',round(summary[-2]['accuracy'],4),round(summary[-1]['accuracy'],4),flush=True)
    csvout(DEST/'runs.csv',summary);csvout(DEST/'responses.csv',responses)
    if groups:csvout(DEST/'subgroup_gains.csv',groups)
    distribution={}
    for profile in ['survey_options','ownership']:
        origin=next(r for r in summary if r['profile']==profile and r['kind']=='original')
        distribution[profile]={}
        for kind in ['permutation','reassignment']:
            rs=[r for r in summary if r['profile']==profile and r['kind']==kind]
            distribution[profile][kind]={}
            for metric in ['accuracy','brier','pt_bias','response_bias','accuracy_gain','brier_change']:
                vals=np.array([r[metric] for r in rs])
                distribution[profile][kind][metric]=dict(original=origin[metric],mean=float(vals.mean()),sd=float(vals.std(ddof=1)),min=float(vals.min()),max=float(vals.max()),quantile_025=float(np.quantile(vals,.025)),quantile_975=float(np.quantile(vals,.975)),original_empirical_percentile=float((vals<=origin[metric]).mean()),n=len(vals))
    writej(DEST/'distribution_summary.json',distribution)

if __name__=='__main__':main()
