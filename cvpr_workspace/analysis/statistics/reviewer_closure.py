"""Review closure, immutable inputs and explicitly conditional inference.
Run protocol before statistics/baseline/fvr. No teacher queries or S9 training.
"""
from pathlib import Path
import sys,json,csv,hashlib,collections,itertools,datetime,platform,os
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/shanghai')]
OUT=Path(os.environ.get('AIT_CLOSURE_OUTPUT',str(ROOT/'outputs/reviewer_closure_20260920'))).resolve()
OUT.relative_to(ROOT/'outputs')  # Output must stay in the project's experiment tree.
BUNDLE=ROOT/'outputs/matched_response_v1/bundle'
NET=ROOT/'outputs/shanghai_survey_matsim_321x10_v1'
MODES=['car','pt','bike','walk']
VARIANTS=['soft_kl','ce_kl','signed_l1','direction_magnitude']
SEEDS=[42,2026,7]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def jl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writejl(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in x),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def csvout(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(x,B=10000,seed=917,family=None):
 x=np.asarray(x,float);assert len(x)>0 and np.isfinite(x).all()
 z=x[np.random.default_rng(seed).integers(len(x),size=(B,len(x)))].mean(1)
 result=dict(n=len(x),mean=float(x.mean()),ci95=np.quantile(z,[.025,.975]).tolist())
 if family is not None:result[f'ci_family{family}']=np.quantile(z,[.05/(2*family),1-.05/(2*family)]).tolist()
 return result
CONTRASTS={
 'rain':({'W1':1,'B0':-1},'pt'),'delay':({'D1':1,'B0':-1},'pt'),
 'rain_delay_interaction':({'WD1':1,'W1':-1,'D1':-1,'B0':1},'pt'),
 'fare':({'F1':1,'B0':-1},'pt'),'parking':({'P1':1,'B0':-1},'car'),
 'road':({'R1':1,'B0':-1},'car'),'walk_vs_wait':({'A_WALK':1,'A_WAIT':-1},'pt'),
 'transfer_vs_wait':({'A_TRANSFER':1,'A_WAIT':-1},'pt')}
def protocol():
 OUT.mkdir(exist_ok=True)
 if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists; do not replace after observing results')
 files=list(BUNDLE.glob('*'))+list((ROOT/'outputs/shanghai_sp_v1/predictions').glob('*.jsonl'))
 files+=list(NET.glob('*/predictions.jsonl'))+list(NET.glob('*/states.jsonl'))
 files += [ROOT/'releases/s9_supply_aware_v2/checkpoint/model.pt',ROOT/'releases/s7_w3_generic_core_v1/checkpoint/model.pt',ROOT/'scripts/shanghai/sp_survey_score.py',ROOT/'scripts/shanghai/sp_survey_compare.py']
 write(OUT/'protocol.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 purpose='Reviewer gaps, designed after old results were inspected; new analyses are exploratory, not prospectively preregistered original claims',
 stages=['survey missingness repair and eight contrasts','12-run complete matched tables','same-feature soft-target MNL','frozen S7/S9 exact FVR','Shanghai execution policy and departure sensitivity','historical evidence ledger'],
 bootstrap='10000 respondent resamples; complete paired people per contrast; 95% marginal and 99.375% Bonferroni intervals for eight contrasts; seeds fixed',
 missing='Never impute a missing human choice. Primary offered-choice rule retained. No outlier exclusion. Every model/seed retained.',
 baseline=dict(model='alternative-specific linear multinomial logit on all common extractor categorical one-hots, numeric global features and all four alternatives numeric features; no hidden layers',
 train='Exposure-frequency-weighted soft cross entropy on same train endpoints; ridge departure auxiliary fitted separately',
 candidates=[0.0001,0.001,0.01,0.1],selection='minimum validation macro-source KL; no test tuning; all candidates saved',
 optimizer='double precision scipy L-BFGS-B, up to 2000 iterations; deterministic convex objective',
 distinction='Same supplied information, endpoint pool and relative supervision exposures; different architecture and converged full-batch optimization, not identical SGD update budget'),
 execution=dict(cards=['B0','D1','A_TRANSFER'],sampling_seeds=[42,2026,7],
 policies=['argmax_soft','argmax_planner_mask','sample_soft','sample_planner_mask'],
 mask='Preflight each offered mode at model-adjusted departure for outbound route only; mask modes whose routed mode differs, renormalize; return separately audited',
 coupling='SHA256 respondent+seed uniform, same across models/cards/masks',models=['s9','supply_mnl'],
 interpretation='Supply/execution stress test using assumed OD and fixed synthetic timetable, not new human validation or physical policy intervention'),
 exclusions=['No new cities or wholesale S9 retraining','No paid Teacher labels: causal attribution to Teacher/city is outside retained claims','No D/E necessity claim, hence no mandatory component ablation'],
 environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform()),
 source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
 for n in ('sp_survey_score.py','sp_survey_compare.py'):
  (OUT/('original_'+n)).write_bytes((ROOT/'scripts/shanghai'/n).read_bytes())
 print('Protocol frozen',flush=True)
def valid(r):return r['choice_status']=='selected' and r['probabilities'].get(r['chosen_mode'],0)>0
def survey_one(rows):
 rows=[r for r in rows if valid(r)]; by={(r['respondent_id'],r['card_id']):r for r in rows}
 assert len(by)==len(rows)
 cards={};paired=[]
 for c in sorted({r['card_id'] for r in rows}):
  rs=[r for r in rows if r['card_id']==c]
  cards[c]=dict(n=len(rs),pt_gap=ci([r['probabilities'].get('pt',0)-float(r['chosen_mode']=='pt') for r in rs]),accuracy=float(np.mean([max(r['probabilities'],key=r['probabilities'].get)==r['chosen_mode'] for r in rs])))
 contrasts={}
 for name,(weights,mode) in CONTRASTS.items():
  ids=sorted(set.intersection(*[{r for r,c0 in by if c0==c} for c in weights]))
  H=[];P=[]
  for rid in ids:
   h=sum(w*float(by[rid,c]['chosen_mode']==mode) for c,w in weights.items())
   p=sum(w*by[rid,c]['probabilities'].get(mode,0) for c,w in weights.items())
   H.append(h);P.append(p);paired.append(dict(respondent_id=rid,contrast=name,human=h,model=p,difference=p-h))
  contrasts[name]=dict(mode=mode,weights=weights,human=ci(H,family=8),model=ci(P,family=8),model_minus_human=ci(np.array(P)-H,family=8))
 return dict(n_primary=len(rows),cards=cards,contrasts=contrasts),paired
def statistics():
 allrows=[]
 for path in sorted((ROOT/'outputs/shanghai_sp_v1/predictions').glob('*.jsonl')):
  result,paired=survey_one(jl(path));write(OUT/'survey'/f'{path.stem}.json',result)
  for name,r in result['contrasts'].items():allrows.append(dict(run=path.stem,contrast=name,n=r['human']['n'],human=r['human']['mean'],model=r['model']['mean'],difference=r['model_minus_human']['mean'],ci_low=r['model_minus_human']['ci95'][0],ci_high=r['model_minus_human']['ci95'][1],family_low=r['model_minus_human']['ci_family8'][0],family_high=r['model_minus_human']['ci_family8'][1]))
  if path.stem=='s9_p042':writejl(OUT/'survey/fixed_s9_contrast_units.jsonl',paired)
 result,paired=survey_one([r for p in NET.glob('*/predictions.jsonl') for r in jl(p)])
 write(OUT/'survey/network_s9.json',result);writejl(OUT/'survey/network_s9_contrast_units.jsonl',paired)
 for name,r in result['contrasts'].items():allrows.append(dict(run='network_s9',contrast=name,n=r['human']['n'],human=r['human']['mean'],model=r['model']['mean'],difference=r['model_minus_human']['mean'],ci_low=r['model_minus_human']['ci95'][0],ci_high=r['model_minus_human']['ci95'][1],family_low=r['model_minus_human']['ci_family8'][0],family_high=r['model_minus_human']['ci_family8'][1]))
 csvout(OUT/'survey/contrasts.csv',allrows)
 full=[]
 for v,s in itertools.product(VARIANTS,SEEDS):
  path=ROOT/f'outputs/matched_response_v1/test/{v}_seed{s}';m=read(path/'metrics.json')['metrics'];pred=jl(path/'predictions.jsonl')
  row=dict(variant=v,seed=s,**{k:m['state'][k] for k in ['n','personas','macro_source_kl','kl','probability_l1','mode_accuracy','fvr','infeasible_pt_mass','departure_mae','departure_p90']},departure_median=float(np.median([r['departure_mae'] for r in pred])),response_gap=m['response']['response_gap'],interaction_error=m['interaction']['error'])
  for eps,t in m['response']['thresholds'].items():
   for k,val in t.items():row[f'{k}@{eps}']=val
  full.append(row)
 csvout(OUT/'matched/all_seeds.csv',full)
 write(OUT/'matched/means_and_training_seed_sd.json',{v:{k:dict(mean=float(np.mean([r[k] for r in full if r['variant']==v])),sd=float(np.std([r[k] for r in full if r['variant']==v],ddof=1))) for k in full[0] if k not in ('variant','seed')} for v in VARIANTS})
 print('Survey 39 fixed encodings + network, 320 contrasts; 12-seed full tables complete',flush=True)

class LinearUtility:
 def __init__(self):
  from traveler_distillation.accessibility.accessibility_features import S8FeatureExtractor
  self.ext=S8FeatureExtractor().from_state_dict(read(BUNDLE/'extractor.json')['state'])
  self.spec=read(BUNDLE/'extractor.json')['feature_spec'] if 'feature_spec' in read(BUNDLE/'extractor.json') else read(BUNDLE/'extractor.json')['spec']
 def design(self,states):
  from traveler_distillation.schemas.state import UniversalTravelerState
  fs=[self.ext.encode(UniversalTravelerState.model_validate(s)) for s in states];xs=[];masks=[]
  sizes=list(self.spec['cat_vocab_sizes'].values())
  for f in fs:
   xs.append(np.concatenate([[1.],*[np.eye(size)[int(k)] for size,k in zip(sizes,f['global_cat'])],f['global_num'],np.array(f['alt_num']).ravel(),np.array(f['alt_available'],float)]))
   masks.append(f['alt_available'])
  return np.array(xs,dtype=float),np.array(masks,dtype=bool)
 def predict_arrays(self,X,M):
  from scipy.special import softmax
  logits=X@self.coef;logits[~M]=-1e30
  return softmax(logits,axis=1),X@self.departure
 def load(self):
  z=np.load(OUT/'baseline/model.npz');self.coef=z['coef'];self.departure=z['departure'];return self
 def decisions(self,states):
  X,M=self.design([s.model_dump(mode='json') for s in states]);P,D=self.predict_arrays(X,M)
  return [dict(mode=MODES[int(p.argmax())],mode_probabilities={m:float(p[j]) for j,m in enumerate(MODES) if mask[j]},departure_time_shift_min=float(d)) for p,d,mask in zip(P,D,M)]
def baseline():
 from scipy.optimize import minimize
 from scipy.special import logsumexp
 from traveler_distillation.matched_response.metrics import pair_scores,interaction_scores,summarize,paired_cluster_difference
 model=LinearUtility();train=jl(BUNDLE/'train_endpoints.jsonl');val=jl(BUNDLE/'val_endpoints.jsonl')
 X,M=model.design([e['state'] for e in train]);Y=np.array([e['teacher_probs'] for e in train]);assert np.all(Y[~M]==0)
 counts=collections.Counter(i for u in jl(BUNDLE/'train_units.jsonl') for i in u['ids']);w=np.array([counts[e['id']] for e in train],float);assert (w>0).all();w/=w.sum()
 XV,MV=model.design([e['state'] for e in val]);YV=np.array([e['teacher_probs'] for e in val]);d=X.shape[1];logYV=np.log(np.clip(YV,1e-300,1))
 candidates=[];best=None
 for lam in read(OUT/'protocol.json')['baseline']['candidates']:
  def fun(flat):
   W=flat.reshape(d,4);z=X@W;z[~M]=-1e30;logp=z-logsumexp(z,axis=1)[:,None];P=np.exp(logp)
   penalty=W.copy();penalty[0]=0
   loss=-np.sum(w[:,None]*Y*logp)+.5*lam*np.square(penalty).sum()
   grad=X.T@(w[:,None]*(P-Y))+lam*penalty
   return float(loss),grad.ravel()
  fit=minimize(fun,np.zeros(d*4),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-12,gtol=1e-7,maxls=50,maxcor=30))
  model.coef=fit.x.reshape(d,4);model.departure=np.zeros(d);PV,_=model.predict_arrays(XV,MV)
  kl=np.sum(YV*(logYV-np.log(np.clip(PV,1e-8,1))),1)
  macro=float(np.mean([kl[[i for i,e in enumerate(val) if e['source']==source]].mean() for source in sorted({e['source'] for e in val})]))
  record=dict(l2=lam,val_macro_kl=macro,success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),train_objective=float(fit.fun),gradient_max=float(np.abs(fit.jac).max()))
  candidates.append(record);write(OUT/'baseline/candidates.json',candidates)
  if fit.success and (best is None or macro<best[0]):best=(macro,lam,model.coef.copy())
  print('MNL candidate',record,flush=True)
 if best is None:raise RuntimeError('No converged baseline candidate')
 model.coef=best[2];lam=best[1]
 # Departure regularization is separately selected by validation MAE, never mode/test results.
 dep_candidates=[];depbest=None
 for dl in read(OUT/'protocol.json')['baseline']['candidates']:
  reg=np.eye(d)*dl;reg[0,0]=0
  coef=np.linalg.solve(X.T@(w[:,None]*X)+reg,X.T@(w*np.array([e['teacher_departure'] for e in train])))
  err=float(np.abs(XV@coef-np.array([e['teacher_departure'] for e in val])).mean());dep_candidates.append(dict(l2=dl,val_mae=err))
  if depbest is None or err<depbest[0]:depbest=(err,dl,coef)
 model.departure=depbest[2]
 write(OUT/'baseline/selection.json',dict(selected_l2=lam,val_macro_kl=best[0],departure_candidates=dep_candidates,selected_departure_l2=depbest[1],features=d,mode_parameters=d*4,train_endpoints=len(train),train_exposures=int(sum(counts.values())),test_used_for_selection=False))
 np.savez(OUT/'baseline/model.npz',coef=model.coef,departure=model.departure)
 # Test is first encoded/evaluated only after selection has been persisted.
 test=jl(BUNDLE/'test_endpoints.jsonl');XT,MT=model.design([e['state'] for e in test]);P,D=model.predict_arrays(XT,MT);pred=[]
 for e,p,dep in zip(test,P,D):
  t=np.array(e['teacher_probs']);inf=e.get('accessibility_class')=='E_infeasible'
  pred.append(dict(**{k:e[k] for k in ['id','persona','trip','input_hash','source','bucket','split']},teacher=t.tolist(),student=p.tolist(),mask=[a['available'] for a in e['state']['alternatives']],teacher_departure=e['teacher_departure'],student_departure=float(dep),kl=float(sum(x*np.log(x/max(y,1e-8)) for x,y in zip(t,p) if x>0)),probability_l1=float(abs(t-p).sum()),mode_accuracy=float(t.argmax()==p.argmax()),departure_mae=float(abs(dep-e['teacher_departure'])),pt_mae=float(abs(t[1]-p[1])),fvr=float(p.argmax()==1) if inf else None,infeasible_pt_mass=float(p[1]) if inf else None))
 pairs=pair_scores(pred,jl(BUNDLE/'test_pairs.jsonl'),[.0001,.001,.01],'union');inter=interaction_scores(pred,jl(BUNDLE/'test_interactions.jsonl'))
 writejl(OUT/'baseline/predictions.jsonl',pred);writejl(OUT/'baseline/pairs.jsonl',pairs);writejl(OUT/'baseline/interactions.jsonl',inter);write(OUT/'baseline/metrics.json',summarize(pred,pairs,inter))
 comparisons={}
 for v in VARIANTS:
  # Seed-mean per unit, followed by persona resampling, conditional on these three seeds.
  comparisons[v]={}
  for filename,key,idkey,ours in [('predictions.jsonl','kl','id',pred),('predictions.jsonl','departure_mae','id',pred),('pairs.jsonl','response_gap','id',pairs),('interactions.jsonl','interaction_gap','id',inter)]:
   runs=[jl(ROOT/f'outputs/matched_response_v1/test/{v}_seed{s}/{filename}') for s in SEEDS]
   by=[{r[idkey]:r for r in run} for run in runs];avg=[dict(r,**{key:float(np.mean([b[r[idkey]][key] for b in by]))}) for r in runs[0]]
   comparisons[v][key]=paired_cluster_difference(ours,avg,key,10000,917,id_key=idkey)
 write(OUT/'baseline/paired_mnl_minus_neural.json',comparisons)
 from traveler_distillation.schemas.state import UniversalTravelerState
 from sp_survey_matsim import ORDER,score
 allpred=[]
 for card in ORDER:
  states=[UniversalTravelerState.model_validate(r['state']) for r in jl(NET/card/'states.jsonl')];dec=model.decisions(states);original=jl(NET/card/'predictions.jsonl')
  rows=[dict(r,probabilities=a['mode_probabilities'],departure_time_shift_min=a['departure_time_shift_min']) for r,a in zip(original,dec)]
  writejl(OUT/f'baseline/survey/{card}.jsonl',rows);allpred+=rows
 write(OUT/'baseline/survey_scores.json',score(allpred));stats,units=survey_one(allpred);write(OUT/'baseline/survey_contrasts.json',stats)
 print('Baseline selected without test, test and Shanghai evaluations saved',flush=True)
def fvr():
 import eval_s8_accessibility as ev
 from traveler_distillation.student.features import FeatureExtractor
 from traveler_distillation.student.model import TravelerStudent
 from reference_pipeline.student_adapter import StudentAdapter
 from scipy.stats import binomtest
 ck=torch.load(ROOT/'releases/s7_w3_generic_core_v1/checkpoint/model.pt',map_location='cpu',weights_only=False)
 ext=FeatureExtractor().from_state_dict(ck['extractor_state']);m=TravelerStudent(ck['config'],ck['feature_spec']);m.load_state_dict(ck['model_state']);m.eval()
 s9=StudentAdapter(ROOT/'releases/s9_supply_aware_v2/checkpoint/model.pt',device='cpu')
 manifest=read(ROOT/'data/singapore_accessibility/split_manifest.json');people=set(manifest['persona_split']['test']);records={r['sample_id']:r for r in jl(ROOT/'data/singapore_accessibility/records.jsonl')}
 targets=[t for t in ev._load(ROOT/'data/singapore_accessibility/states_with_teacher.jsonl') if t.persona_group_id in people];assert len(targets)==51
 p9=s9.predict([t.state for t in targets]);rows=[]
 for t,b in zip(targets,p9):
  a=ev._predict(m,ext,t.state,'cpu');r=records[t.sample_id]
  rows.append(dict(id=t.sample_id,persona=t.persona_group_id,curve_group=r['curve_group'],od_index=r['od_index'],accessibility_class=r['accessibility_class'],s7=a,s9=b['mode_probabilities'],teacher=t.teacher_aggregate.mode_probabilities,s9_departure=b['departure_time_shift_min'],teacher_departure=t.teacher_aggregate.departure_time_shift_min))
 inf=[r for r in rows if r['accessibility_class']=='E_infeasible'];assert len(inf)==12
 for r in inf:
  r['s7_violation']=int(max(r['s7'],key=r['s7'].get)=='pt');r['s9_violation']=int(max(r['s9'],key=r['s9'].get)=='pt');r['delta']=r['s9_violation']-r['s7_violation']
 improved=sum(r['delta']==-1 for r in inf);worsened=sum(r['delta']==1 for r in inf)
 # Match old shared RNG: first 2000x51 MAE indices, then 2000x12 mass, then FVR.
 rng=np.random.default_rng(42);rng.integers(0,51,size=(2000,51));rng.integers(0,12,size=(2000,12));draw=rng.integers(0,12,size=(2000,12));delta=np.array([r['delta'] for r in inf]);z=delta[draw].mean(1)
 grouped={}
 for field in ['persona','curve_group']:
  gs=[[r['delta'] for r in inf if r[field]==g] for g in sorted({r[field] for r in inf})];sums=np.array([sum(g) for g in gs]);ns=np.array([len(g) for g in gs]);ix=np.random.default_rng(917).integers(len(gs),size=(10000,len(gs)));zz=sums[ix].sum(1)/ns[ix].sum(1)
  grouped[field]=dict(n_clusters=len(gs),mean=float(delta.mean()),ci95=np.quantile(zz,[.025,.975]).tolist())
 # Exact conditional iid-state bootstrap enumerates the multinomial count law.
 from scipy.stats import multinomial
 vals,counts=np.unique(delta,return_counts=True);exact=[]
 for cells in itertools.product(range(13),repeat=len(vals)):
  if sum(cells)==12:exact.append((float(np.dot(cells,vals)/12),float(multinomial.pmf(cells,12,counts/12))))
 exact.sort();cdf=np.cumsum([p for _,p in exact]);quant=[exact[min(int(np.searchsorted(cdf,q)),len(exact)-1)][0] for q in [.025,.975]]
 result=dict(n=12,s7_violations=sum(r['s7_violation'] for r in inf),s9_violations=sum(r['s9_violation'] for r in inf),improved=improved,worsened=worsened,exact_mcnemar_two_sided=float(binomtest(improved,improved+worsened,.5).pvalue) if improved+worsened else 1.,old_rng_bootstrap_ci=np.quantile(z,[.025,.975]).tolist(),exact_iid_bootstrap_ci=quant,cluster_sensitivity=grouped,warning='12 designed states and only six personas; McNemar assumes independent discordances, cluster intervals descriptive; old narrow CI is Monte Carlo-sensitive',s9_departure=dict(n=51,mae=float(np.mean([abs(r['s9_departure']-r['teacher_departure']) for r in rows])),median=float(np.median([abs(r['s9_departure']-r['teacher_departure']) for r in rows])),p90=float(np.quantile([abs(r['s9_departure']-r['teacher_departure']) for r in rows],.9))))
 writejl(OUT/'fvr/all_51_predictions.jsonl',rows);writejl(OUT/'fvr/paired_12.jsonl',inf);write(OUT/'fvr/analysis.json',result);print('FVR',result,flush=True)
if __name__=='__main__':
 torch.set_num_threads(4)
 globals()[sys.argv[1]]()
