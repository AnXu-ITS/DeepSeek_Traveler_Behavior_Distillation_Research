"""Fixed-exposure timing baselines and validation-MAE neural checkpoint selection.

All fitting uses the frozen matched-response training and validation pools.
Original checkpoints are read-only. Command: .venv/Scripts/python.exe
scripts/revision_20260921/departure_baselines.py train|analyze.
"""
from pathlib import Path
import sys, json, copy, time, csv, hashlib, collections, argparse
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'cvpr_workspace/analysis/statistics')]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from reviewer_closure import LinearUtility
from traveler_distillation.accessibility.accessibility_features import S8FeatureExtractor, TravelerStudentS8
from traveler_distillation.student.dataset import collate_batch
from traveler_distillation.matched_response.experiment import seed_all, epoch_order, loss_for_units, model_digest, VARIANTS
from traveler_distillation.matched_response.metrics import encode, predict, pair_scores, interaction_scores, summarize
from traveler_distillation.matched_response.data import load_split

BUNDLE = ROOT/'outputs/matched_response_v1/bundle'
OUT = ROOT/'outputs/revision_20260921/baselines'
OLD = ROOT/'outputs/matched_response_v1'
SEEDS = [42,2026,7]
MAIN_VARIANTS = ['soft_kl','ce_kl','signed_l1','direction_magnitude']
MODES = ['car','pt','bike','walk']
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_rows(p,rs):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in rs),encoding='utf-8')
def csvout(p,rs):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class IndependentDeparture(TravelerStudentS8):
    """Same feature encoders and timing head; no utility scorer or mode loss."""
    def __init__(self,cfg,spec):
        super().__init__(cfg,spec)
        del self.scorer
    def forward(self,b):
        cats=torch.cat([self.cat_embeddings[n](b['global_cat'][:,i]) for i,n in enumerate(self.cat_names)],-1)
        g=self.global_encoder(torch.cat([cats,b['global_num']],-1))
        a=self.alt_encoder(torch.cat([self.mode_embedding(b['alt_mode_idx']),b['alt_num']],-1))
        m=b['alt_mask'];pool=(a*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
        return 60*torch.tanh(self.departure_head(torch.cat([g,pool],-1)).squeeze(-1))

class BoundedLinear(nn.Module):
    def __init__(self,d):
        super().__init__();self.linear=nn.Linear(d-1,1)
    def forward(self,x): return 60*torch.tanh(self.linear(x[:,1:]).squeeze(-1))

class CombinationAdapter:
    """Same frozen MNL-S choice with a validation-selected timing branch."""
    def __init__(self,checkpoint):
        self.choice=LinearUtility().load()
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.kind=ck['kind'];self.ext=S8FeatureExtractor.from_state_dict(ck['extractor_state'])
        self.model=BoundedLinear(ck['design_dim']) if self.kind=='bounded_linear' else IndependentDeparture(ck['config'],ck['feature_spec'])
        self.model.load_state_dict(ck['model_state']);self.model.eval()
    @torch.no_grad()
    def predict(self,states):
        sd=[s.model_dump(mode='json') if hasattr(s,'model_dump') else s for s in states]
        X,M=self.choice.design(sd);P,_=self.choice.predict_arrays(X,M)
        if self.kind=='bounded_linear': D=self.model(torch.tensor(X,dtype=torch.float32)).tolist()
        else:
            es=[dict(state=s,teacher_probs=[0]*4,teacher_departure=0) for s in sd]
            D=self.model(collate_batch([encode(e,self.ext) for e in es])).tolist()
        return [dict(mode=MODES[int(p.argmax())],mode_probabilities={m:float(p[j]) for j,m in enumerate(MODES) if mask[j]},departure_time_shift_min=float(d)) for p,d,mask in zip(P,D,M)]
    decisions=predict

def extra_summary(pred,ps,ins):
    result=summarize(pred,ps,ins)
    result['state']['departure_median']=float(np.median([r['departure_mae'] for r in pred]))
    for source,v in result['state']['by_source'].items():
        v['departure_median']=float(np.median([r['departure_mae'] for r in pred if r['source']==source]))
    return result

def save_eval(folder,pred,split):
    ps=pair_scores(pred,split['pairs'],[.0001,.001,.01],'union')
    ins=interaction_scores(pred,split['interactions'])
    for name,rs in [('predictions',pred),('pairs',ps),('interactions',ins)]:write_rows(folder/(name+'.jsonl'),rs)
    report=extra_summary(pred,ps,ins);write(folder/'metrics.json',report);return report

def combined_predictions(data,departures):
    original=rows(ROOT/'outputs/reviewer_closure_20260920/baseline/predictions.jsonl')
    by={r['id']:r for r in original}
    return [dict(by[e['id']],student_departure=float(d),departure_mae=float(abs(d-e['teacher_departure']))) for e,d in zip(data['endpoints'],departures)]

def train(only=None):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    cfg=yaml.safe_load((ROOT/'configs/matched_response.yaml').read_text(encoding='utf-8'))
    train,manifest=load_split(BUNDLE,'train'); val,_=load_split(BUNDLE,'val')
    ext=S8FeatureExtractor.from_state_dict(read(BUNDLE/'extractor.json')['state'])
    enc={e['id']:encode(e,ext) for e in train['endpoints']}
    venc={e['id']:encode(e,ext) for e in val['endpoints']}
    lm=LinearUtility().load(); X,_=lm.design([e['state'] for e in train['endpoints']]);XV,_=lm.design([e['state'] for e in val['endpoints']])
    X=torch.tensor(X,dtype=torch.float32); XV=torch.tensor(XV,dtype=torch.float32)
    ix={e['id']:i for i,e in enumerate(train['endpoints'])}
    y=torch.tensor([e['teacher_departure'] for e in train['endpoints']],dtype=torch.float32)
    yv=np.array([e['teacher_departure'] for e in val['endpoints']])
    vb=collate_batch([venc[e['id']] for e in val['endpoints']])
    unit_ids=[[ix[e] for e in u['ids']] for u in train['units']]
    training=dict(epochs=120,batch_units=32,learning_rate=.000125,weight_decay=.0001,seeds=SEEDS,threads=2,
        departure_huber_delta=1.,selection='minimum unweighted validation endpoint departure MAE; earliest tie',
        train_endpoints=len(enc),validation_endpoints=len(venc),units_per_epoch=len(unit_ids),
        endpoint_exposures_per_epoch=sum(map(len,unit_ids)),initialization='independent random initialization; no inherited weights',
        bundle_id=manifest['bundle_id'],survey_used_for_training_or_selection=False)
    write(OUT/'training_protocol.json',dict(training=training,original_config=cfg,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),BUNDLE/'manifest.json',ROOT/'src/traveler_distillation/matched_response/experiment.py',ROOT/'outputs/reviewer_closure_20260920/baseline/model.npz']}))
    jobs=[(kind,s) for s in SEEDS for kind in ['bounded_linear','independent_neural']+MAIN_VARIANTS]
    if only:jobs=[(k,s) for k,s in jobs if k in only]
    for kind,seed in jobs:
        folder=OUT/'train'/f'{kind}_seed{seed}'
        if (folder/'status.json').exists() and read(folder/'status.json')['status']=='complete':continue
        folder.mkdir(parents=True,exist_ok=True);seed_all(seed)
        joint=kind in MAIN_VARIANTS
        m=TravelerStudentS8(cfg['student'],ext.spec) if joint else (IndependentDeparture(cfg['student'],ext.spec) if kind=='independent_neural' else BoundedLinear(X.shape[1]))
        opt=torch.optim.Adam(m.parameters(),lr=.000125,weight_decay=.0001)
        best=float('inf');bestkl=float('inf');hist=[];start=time.perf_counter();step=0
        initial=model_digest(m)
        write(folder/'status.json',dict(status='running',kind=kind,seed=seed))
        for epoch in range(120):
            order=epoch_order(len(train['units']),seed,epoch);m.train()
            for offset in range(0,len(order),32):
                ui=order[offset:offset+32];opt.zero_grad(set_to_none=True)
                if joint:loss,_=loss_for_units(m,[train['units'][i] for i in ui],enc,'cpu',kind,cfg['objective'])
                else:
                    ids=[i for u in ui for i in unit_ids[u]]
                    if kind=='bounded_linear':p=m(X[ids])
                    else:p=m(collate_batch([enc[train['endpoints'][i]['id']] for i in ids]))
                    loss=F.huber_loss(p,y[ids],delta=1.,reduction='sum')/len(ui)
                if not torch.isfinite(loss):raise FloatingPointError(kind)
                loss.backward();opt.step();step+=1
            m.eval()
            with torch.no_grad():
                if joint:
                    vp=predict(m,val['endpoints'],venc,'cpu',128);vm=summarize(vp,[],[])['state'];mae=vm['departure_mae'];kl=vm['macro_source_kl']
                else:
                    dv=m(XV) if kind=='bounded_linear' else m(vb)
                    mae=float(np.abs(dv.numpy()-yv).mean());kl=None
            hist.append(dict(epoch=epoch+1,validation_departure_mae=mae,validation_macro_kl=kl,optimizer_steps=step))
            ck=dict(model_state=copy.deepcopy(m.state_dict()),config=cfg['student'],feature_spec=ext.spec,extractor_state=ext.state_dict(),kind=kind,seed=seed,epoch=epoch+1,selection_score=mae,selection='validation_departure_mae',initial_model_hash=initial,design_dim=X.shape[1],bundle_id=manifest['bundle_id'])
            if mae<best:best=mae;torch.save(ck,folder/'best_departure.pt');bestep=epoch+1
            if joint and kl<bestkl:
                bestkl=kl;ck['selection']='validation_macro_source_kl';ck['selection_score']=kl;torch.save(ck,folder/'best_kl_replay.pt')
            if (epoch+1)%30==0:
                write_rows(folder/'history.jsonl',hist)
                print(kind,seed,'epoch',epoch+1,'best val MAE',round(best,4),flush=True)
        write_rows(folder/'history.jsonl',hist)
        status=dict(status='complete',kind=kind,seed=seed,epochs=120,optimizer_steps=step,best_epoch=bestep,best_validation_departure_mae=best,elapsed_s=time.perf_counter()-start,parameters=sum(p.numel() for p in m.parameters()),initial_model_hash=initial,checkpoint_sha256=sha(folder/'best_departure.pt'))
        if joint:
            original=torch.load(OLD/f'train/{kind}_seed{seed}/best.pt',weights_only=False,map_location='cpu')
            replay=torch.load(folder/'best_kl_replay.pt',weights_only=False,map_location='cpu')
            status['kl_replay']=dict(original_epoch=original['epoch'],replay_epoch=replay['epoch'],max_absolute_weight_difference=max(float((original['model_state'][k]-v).abs().max()) for k,v in replay['model_state'].items()))
        write(folder/'status.json',status);print('COMPLETE',kind,seed,round(status['elapsed_s'],2),flush=True)

def analyze():
    torch.set_num_threads(2)
    test,_=load_split(BUNDLE,'test');lm=LinearUtility().load();XT,MT=lm.design([e['state'] for e in test['endpoints']]);raw=XT@lm.departure
    ridge=dict(test_n=len(raw),outside_bounds_n=int((np.abs(raw)>60).sum()),outside_bounds_fraction=float((np.abs(raw)>60).mean()),minimum=float(raw.min()),maximum=float(raw.max()))
    for label,ds in [('original',raw),('clipped',np.clip(raw,-60,60))]:
        ridge[label]=save_eval(OUT/'ridge'/label,combined_predictions(test,ds),test)
    write(OUT/'ridge_diagnostic.json',ridge)
    out=[]
    for folder in sorted((OUT/'train').glob('*_seed*')):
        if not (folder/'status.json').exists() or read(folder/'status.json')['status']!='complete':continue
        st=read(folder/'status.json');ck=torch.load(folder/'best_departure.pt',map_location='cpu',weights_only=False);kind=st['kind']
        if kind in MAIN_VARIANTS:
            ext=S8FeatureExtractor.from_state_dict(ck['extractor_state']);m=TravelerStudentS8(ck['config'],ck['feature_spec']);m.load_state_dict(ck['model_state']);m.eval()
            enc={e['id']:encode(e,ext) for e in test['endpoints']};p=predict(m,test['endpoints'],enc,'cpu',128)
            batch=collate_batch([enc[e['id']] for e in test['endpoints']]);call=lambda:m(batch)
            params=st['parameters']
        else:
            adapter=CombinationAdapter(folder/'best_departure.pt');d=adapter.predict([e['state'] for e in test['endpoints']]);p=combined_predictions(test,[r['departure_time_shift_min'] for r in d]);params=444+st['parameters']
            if kind=='bounded_linear':arg=torch.tensor(XT,dtype=torch.float32)
            else:arg=collate_batch([encode(e,adapter.ext) for e in test['endpoints']])
            call=lambda: (lm.predict_arrays(XT,MT)[0],adapter.model(arg))
        report=save_eval(OUT/'test'/folder.name,p,test)
        with torch.no_grad():
            for _ in range(5):call()
            t=[]
            for _ in range(30):
                start=time.perf_counter();call();t.append(time.perf_counter()-start)
        timing=dict(n_endpoints=len(test['endpoints']),repetitions=30,warmups=5,median_ms=float(np.median(t)*1000),per_endpoint_us=float(np.median(t)*1e6/len(test['endpoints'])),scope='batched pre-encoded model computation; combination includes both MNL matrix softmax and timing branch; excludes encoding',threads=2)
        write(OUT/'test'/folder.name/'timing.json',timing)
        s=report['state'];out.append(dict(model=kind,seed=st['seed'],selection='validation_departure_mae',epoch=ck['epoch'],parameters=params,training_seconds=st['elapsed_s'],inference_us=timing['per_endpoint_us'],departure_mae=s['departure_mae'],departure_median=s['departure_median'],departure_p90=s['departure_p90'],macro_kl=s['macro_source_kl'],response_gap=report['response']['response_gap'],interaction_error=report['interaction']['error']))
    csvout(OUT/'departure_comparison.csv',out)
    means={k:{metric:dict(mean=float(np.mean([r[metric] for r in out if r['model']==k])),sd=float(np.std([r[metric] for r in out if r['model']==k],ddof=1))) for metric in ['departure_mae','departure_median','departure_p90','macro_kl','response_gap','interaction_error','inference_us']} for k in sorted({r['model'] for r in out})}
    write(OUT/'departure_summary.json',means);print(json.dumps(means,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','analyze']);parser.add_argument('--only',nargs='+');args=parser.parse_args()
    train(args.only) if args.command=='train' else analyze()
