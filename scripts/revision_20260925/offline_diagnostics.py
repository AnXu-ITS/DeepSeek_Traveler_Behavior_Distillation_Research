"""Summarize retained gradients and coherent MNL access perturbations, no fits."""
import copy,collections
import numpy as np
from controlled import ROOT,OUT,SEEDS,GRID,name,read_json,write_json,write_rows,file_hash,BUNDLE
from offline_audits import rows
from mnl_selection import LinearUtility

def main():
    gradient=[]
    for v,w in GRID:
        for seed in SEEDS:
            folder=OUT/name(v,w,seed);run=read_json(folder/'run.json')
            wd=run['config']['objective']['departure_weight']
            for g in read_json(folder/'gradient_diagnostics.json'):
                gradient.append(dict(variant=v,weight=w,seed=seed,epoch=g['epoch'],kl_norm=g['kl']['gradient_norm'],departure_norm=g['departure']['gradient_norm'],weighted_departure_norm=wd*g['departure']['gradient_norm'],weighted_response_norm=g['weighted_response_gradient_norm'],response_to_kl=g['weighted_response_gradient_norm']/g['kl']['gradient_norm'],cosine=g['kl_response_cosine']))
    summary=[]
    for v,w in GRID:
        for ep in [0,1,30,60,120]:
            rs=[r for r in gradient if r['variant']==v and r['weight']==w and r['epoch']==ep]
            summary.append(dict(variant=v,weight=w,epoch=ep,n_seeds=3,**{k:float(np.mean([r[k] for r in rs])) for k in ['kl_norm','weighted_departure_norm','weighted_response_norm','response_to_kl','cosine']}))
    write_rows(OUT.parent/'audit/gradient_units.jsonl',gradient);write_json(OUT.parent/'audit/gradient_summary.json',summary)
    model=LinearUtility();es=rows(BUNDLE/'test_endpoints.jsonl');X,M=model.design([e['state'] for e in es]);unit=[]
    for selector,choice in read_json(OUT.parent/'mnl_selection/selection.json').items():
        path=OUT.parent/f'mnl_selection/candidate_{choice["l2"]:g}.npz';ck=np.load(path);model.coef=ck['coef'];model.departure=ck['departure'];P,D=model.predict_arrays(X,M)
        for kind in ['cost_plus1','time_plus5','access_and_total_plus5']:
            changed=copy.deepcopy(es)
            for e in changed:
                pt=next(a for a in e['state']['alternatives'] if a['mode']=='pt')
                if kind=='cost_plus1':pt['monetary_cost']+=1
                else:
                    pt['travel_time_min']+=5
                    if kind=='access_and_total_plus5':pt['access_time_min']+=5
            XX,MM=model.design([e['state'] for e in changed]);PP,_=model.predict_arrays(XX,MM)
            for e,p,p2,x,xx in zip(es,P,PP,X,XX):
                pt=next(a for a in e['state']['alternatives'] if a['mode']=='pt')
                unit.append(dict(selection=selector,perturbation=kind,id=e['id'],persona=e['persona'],source=e['source'],pt_feasible=pt.get('pt_feasible'),pt_probability_change=float(p2[1]-p[1]),pt_minus_walk_utility_change=float((xx-x)@(model.coef[:,1]-model.coef[:,3])),checkpoint_sha256=file_hash(path)))
    summaries=[]
    for selector in ['static','response']:
        for kind in ['cost_plus1','time_plus5','access_and_total_plus5']:
            for subset in ['all','service_feasible']:
                rs=[r for r in unit if r['selection']==selector and r['perturbation']==kind and (subset=='all' or r['pt_feasible']==1)]
                if not rs:continue
                vals=np.array([r['pt_probability_change'] for r in rs])
                summaries.append(dict(selection=selector,perturbation=kind,subset=subset,n_endpoints=len(rs),n_personas=len({r['persona'] for r in rs}),mean_pt_probability_change=float(vals.mean()),fraction_pt_increases=float((vals>1e-8).mean()),minimum=float(vals.min()),maximum=float(vals.max())))
    write_rows(OUT.parent/'mnl_selection/extended_sensitivity_units.jsonl',unit)
    write_json(OUT.parent/'mnl_selection/extended_sensitivity_summary.json',dict(rows=summaries,interpretation='Finite perturbations of an unconstrained feature-based utility. Access adds five minutes to both access and total PT time; pure time and cost are encoding diagnostics. Mixed coefficient signs do not identify economic values of time or causal preferences. All endpoints are retained, with feasible-service subset separately labeled.'))
    print('135 gradient records and '+str(len(unit))+' MNL finite perturbations summarized')

if __name__=='__main__':main()
