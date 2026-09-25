"""MNL regularization crossed with the same two validation selectors."""
from pathlib import Path
import collections,copy,sys
from controlled import *
from offline_audits import RAW,rows
sys.path[:0]=[str(RAW/'cvpr_workspace/analysis/statistics'),str(RAW/'scripts/shanghai')]
from reviewer_closure import LinearUtility
from scipy.optimize import minimize
from scipy.special import logsumexp
DEST=OUT.parent/'mnl_selection'

def predictions(es,P,D):
    pred=[]
    for e,p,dep in zip(es,P,D):
        t=np.array(e['teacher_probs']);inf=e.get('accessibility_class')=='E_infeasible'
        pred.append(dict(**{k:e[k] for k in ['id','persona','trip','input_hash','source','bucket','split']},teacher=t.tolist(),student=p.tolist(),mask=[a['available'] for a in e['state']['alternatives']],teacher_departure=e['teacher_departure'],student_departure=float(dep),kl=float(sum(x*np.log(x/max(y,1e-8)) for x,y in zip(t,p) if x>0)),probability_l1=float(abs(t-p).sum()),mode_accuracy=float(t.argmax()==p.argmax()),departure_mae=float(abs(dep-e['teacher_departure'])),pt_mae=float(abs(t[1]-p[1])),fvr=float(p.argmax()==1) if inf else None,infeasible_pt_mass=float(p[1]) if inf else None))
    return pred

def main():
    DEST.mkdir(parents=True,exist_ok=True);model=LinearUtility();tr,_=load_split(BUNDLE,'train');va,_=load_split(BUNDLE,'val')
    X,M=model.design([e['state'] for e in tr['endpoints']]);Y=np.array([e['teacher_probs'] for e in tr['endpoints']]);XV,MV=model.design([e['state'] for e in va['endpoints']])
    count=collections.Counter(i for u in tr['units'] for i in u['ids']);w=np.array([count[e['id']] for e in tr['endpoints']],float);w/=w.sum();d=X.shape[1]
    # Retain historical departure coefficients: this experiment selects choice,
    # and no departure selection result is used to select the choice model.
    departure=np.load(RAW/'outputs/reviewer_closure_20260920/baseline/model.npz')['departure'];model.departure=departure
    candidates=[];best={}
    for lam in [.0001,.001,.01,.1]:
        def fun(flat):
            W=flat.reshape(d,4);z=X@W;z[~M]=-1e30;lp=z-logsumexp(z,axis=1)[:,None];pen=W.copy();pen[0]=0
            return float(-np.sum(w[:,None]*Y*lp)+.5*lam*np.square(pen).sum()),(X.T@(w[:,None]*(np.exp(lp)-Y))+lam*pen).ravel()
        fit=minimize(fun,np.zeros(d*4),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-12,gtol=1e-7,maxls=50,maxcor=30))
        model.coef=fit.x.reshape(d,4);P,D=model.predict_arrays(XV,MV);sc=scores(predictions(va['endpoints'],P,D),va['pairs'])
        rec=dict(l2=lam,success=bool(fit.success),iterations=int(fit.nit),gradient_max=float(abs(fit.jac).max()),message=str(fit.message),validation=sc);candidates.append(rec)
        np.savez(DEST/f'candidate_{lam:g}.npz',coef=model.coef,departure=departure)
        write_json(DEST/'candidates.json',candidates)
        for sel,v in sc.items():
            if fit.success and (sel not in best or v<best[sel]['score']):best[sel]=dict(score=v,l2=lam)
        print(rec,flush=True)
    assert set(best)=={'static','response'};write_json(DEST/'selection.json',best)
    test,_=load_split(BUNDLE,'test');XT,MT=model.design([e['state'] for e in test['endpoints']]);sens=[]
    for sel,b in best.items():
        z=np.load(DEST/f'candidate_{b["l2"]:g}.npz');model.coef=z['coef'];P,D=model.predict_arrays(XT,MT);pr=predictions(test['endpoints'],P,D);pa=pair_scores(pr,test['pairs'],[.0001,.001,.01],'union');ints=interaction_scores(pr,test['interactions'])
        folder=DEST/sel;folder.mkdir(exist_ok=True)
        write_rows(folder/'predictions.jsonl',pr);write_rows(folder/'pairs.jsonl',pa);write_json(folder/'metrics.json',summarize(pr,pa,ints))
        # Identified utility differences relative to walk; raw coefficients are
        # encoding-dependent and are not interpreted as money/time tradeoffs.
        np.savez(folder/'utility_contrasts_walk_reference.npz',coef=model.coef-model.coef[:,3:4])
        for field,increment in [('monetary_cost',1.),('travel_time_min',5.)]:
            changed=copy.deepcopy(test['endpoints'])
            for e in changed:
                pt=next(a for a in e['state']['alternatives'] if a['mode']=='pt');pt[field]+=increment
            XX,MM=model.design([e['state'] for e in changed]);PP,_=model.predict_arrays(XX,MM)
            for e,p,p2,xx,x in zip(test['endpoints'],P,PP,XX,XT):
                dv=(xx-x)@(model.coef[:,1]-model.coef[:,3])
                sens.append(dict(selection=sel,id=e['id'],persona=e['persona'],field=field,increment=increment,pt_probability_change=float(p2[1]-p[1]),pt_minus_walk_utility_change=float(dv),interpretation='finite encoding perturbation; conditional association, no causal or value-of-time claim'))
    write_rows(DEST/'finite_sensitivity.jsonl',sens)

if __name__=='__main__':main()
