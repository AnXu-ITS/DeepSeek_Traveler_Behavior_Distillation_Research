"""Prospectively locked weight/selection experiment, using frozen targets only.

Two checkpoint selectors observe the same training trajectory. Test evaluation is
a separate command and cannot affect either checkpoint. No historical files change.
"""
from pathlib import Path
import argparse, collections, copy, json, sys, time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
import yaml
from traveler_distillation.matched_response.experiment import (
    seed_all, model_digest, epoch_order, loss_for_units, _encoded, code_fingerprint,
    S8FeatureExtractor, TravelerStudentS8)
from traveler_distillation.matched_response.data import load_split, read_json, write_json, write_rows, file_hash, digest
from traveler_distillation.matched_response.metrics import predict, pair_scores, interaction_scores, summarize, paired_cluster_difference
BUNDLE=ROOT/'outputs/matched_response_v1/bundle'
OUT=ROOT/'outputs/revision_20260925/controlled'
SEEDS=[42,2026,7]
GRID=[('soft_kl',0.)]+[(v,w) for v in ['signed_l1','direction_magnitude'] for w in [.25,.5,1.,2.]]

def name(v,w,s): return f'{v}_w{w:g}_seed{s}'
def scores(pred,pairs):
    by={p['id']:p for p in pred}
    gaps=[]
    for r in pairs:
        a,b=by[r['base']],by[r['cf']]
        mask=np.asarray(a['mask'])|np.asarray(b['mask'])
        d=(np.asarray(b['teacher'])-a['teacher'])-(np.asarray(b['student'])-a['student'])
        gaps.append(float(abs(d[mask]).mean()))
    return dict(static=float(np.mean([np.mean([p['kl'] for p in pred if p['source']==s]) for s in sorted({p['source'] for p in pred})])),response=float(np.mean(gaps)))

def gradient_diagnostic(model,units,enc,variant,obj):
    # Eval mode avoids dropout and leaves the training RNG stream unchanged.
    model.eval(); params=list(model.parameters()); results={}; vectors={}
    for term in ['kl','departure','response']:
        conf=dict(obj,departure_weight=float(term=='departure'),response_weight=float(term=='response'))
        total,parts=loss_for_units(model,units,enc,'cpu',variant,conf)
        if term!='kl':
            kl,_=loss_for_units(model,units,enc,'cpu','soft_kl',dict(obj,departure_weight=0.,response_weight=0.))
            total=total-kl
        gs=torch.autograd.grad(total,params,allow_unused=True)
        g=torch.cat([(torch.zeros_like(p) if x is None else x).flatten() for p,x in zip(params,gs)])
        vectors[term]=g
        results[term]=dict(loss=float(total.detach()),gradient_norm=float(g.norm()))
    results['kl_response_cosine']=float(torch.nn.functional.cosine_similarity(vectors['kl'],vectors['response'],dim=0))
    results['weighted_response_gradient_norm']=results['response']['gradient_norm']*obj['response_weight']
    return results

def train(variant,weight,seed):
    out=OUT/name(variant,weight,seed)
    if (out/'status.json').exists() and read_json(out/'status.json')['status']=='complete':return
    if out.exists(): raise RuntimeError(f'Incomplete run needs explicit recovery: {out}')
    out.mkdir(parents=True)
    cfg=yaml.safe_load((ROOT/'configs/matched_response.yaml').read_text())
    cfg['objective']['response_weight']=weight
    tr,manifest=load_split(BUNDLE,'train');va,_=load_split(BUNDLE,'val')
    ext=S8FeatureExtractor.from_state_dict(read_json(BUNDLE/'extractor.json')['state'])
    te=_encoded(tr,ext);ve=_encoded(va,ext)
    torch.set_num_threads(cfg['training']['threads']);torch.use_deterministic_algorithms(True);seed_all(seed)
    model=TravelerStudentS8(cfg['student'],ext.spec)
    init=model_digest(model); opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'])
    protocol=read_json(OUT.parent/'protocol.json')
    write_json(out/'run.json',dict(variant=variant,weight=weight,seed=seed,initial_hash=init,config=cfg,protocol_sha256=file_hash(OUT.parent/'protocol.json'),bundle_id=manifest['bundle_id'],code_id=code_fingerprint(),driver_sha256=file_hash(__file__),test_used_for_selection=False))
    best=dict(static=float('inf'),response=float('inf'));history=[];grad=[];steps=0;start=time.monotonic()
    diagnostic=[tr['units'][i] for i in epoch_order(len(tr['units']),917,0)[:128]]
    grad.append(dict(epoch=0,**gradient_diagnostic(model,diagnostic,te,variant,cfg['objective'])))
    try:
        for ep in range(cfg['training']['epochs']):
            order=epoch_order(len(tr['units']),seed,ep); model.train();total=0.;parts_sum=collections.Counter()
            for offset in range(0,len(order),cfg['training']['batch_units']):
                units=[tr['units'][i] for i in order[offset:offset+cfg['training']['batch_units']]]
                opt.zero_grad(set_to_none=True)
                loss,parts=loss_for_units(model,units,te,'cpu',variant,cfg['objective'])
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                opt.step();steps+=1;total+=float(loss.detach())*len(units);parts_sum.update(parts)
            pred=predict(model,va['endpoints'],ve,'cpu',128);sc=scores(pred,va['pairs'])
            history.append(dict(epoch=ep+1,steps=steps,loss=total/len(order),parts=dict(parts_sum),validation=sc,schedule_hash=digest(order)))
            for sel,value in sc.items():
                if value<best[sel]:
                    best[sel]=value
                    torch.save(dict(model_state=model.state_dict(),config=cfg['student'],feature_spec=ext.spec,extractor_state=ext.state_dict(),epoch=ep+1,selection=sel,selection_score=value,variant=variant,weight=weight,seed=seed,initial_hash=init),out/f'best_{sel}.pt')
            if ep+1 in [1,30,60,120]:grad.append(dict(epoch=ep+1,**gradient_diagnostic(model,diagnostic,te,variant,cfg['objective'])))
            write_rows(out/'history.jsonl',history)
        write_json(out/'gradient_diagnostics.json',grad)
        write_json(out/'status.json',dict(status='complete',epochs=len(history),steps=steps,elapsed_s=time.monotonic()-start,best=best,checkpoint_sha256={s:file_hash(out/f'best_{s}.pt') for s in best}))
        print(name(variant,weight,seed),json.dumps(read_json(out/'status.json')),flush=True)
    except BaseException as exc:
        write_json(out/'status.json',dict(status='failed',completed_epochs=len(history),steps=steps,error=type(exc).__name__))
        raise

def evaluate():
    assert all((OUT/name(v,w,s)/'status.json').exists() and read_json(OUT/name(v,w,s)/'status.json')['status']=='complete' for v,w in GRID for s in SEEDS)
    test,_=load_split(BUNDLE,'test');result=[];pairsets={}
    torch.set_num_threads(4)
    for v,w in GRID:
        for s in SEEDS:
            folder=OUT/name(v,w,s)
            for sel in ['static','response']:
                ck=torch.load(folder/f'best_{sel}.pt',map_location='cpu',weights_only=False)
                ext=S8FeatureExtractor.from_state_dict(ck['extractor_state']);model=TravelerStudentS8(ck['config'],ext.spec);model.load_state_dict(ck['model_state'])
                pr=predict(model,test['endpoints'],_encoded(test,ext),'cpu',128)
                pairs=pair_scores(pr,test['pairs'],[.0001,.001,.01],'union');inter=interaction_scores(pr,test['interactions']);summary=summarize(pr,pairs,inter)
                dest=folder/f'test_{sel}';dest.mkdir(exist_ok=True)
                for filename,rs in [('predictions',pr),('pairs',pairs),('interactions',inter)]:write_rows(dest/f'{filename}.jsonl',rs)
                write_json(dest/'metrics.json',summary);pairsets[v,w,s,sel]=pairs
                result.append(dict(variant=v,weight=w,seed=s,selection=sel,epoch=ck['epoch'],static_kl=summary['state']['macro_source_kl'],response_gap=summary['response']['response_gap'],departure_mae=summary['state']['departure_mae'],unseen_combination=summary['by_bucket'].get('joint_unseen')))
    write_json(OUT/'test_summary.json',result)
    comparisons=[]
    for v,w in GRID[1:]:
        for sel in ['static','response']:
            lefts=[{r['id']:r for r in pairsets[v,w,s,sel]} for s in SEEDS]
            rights=[{r['id']:r for r in pairsets['soft_kl',0.,s,sel]} for s in SEEDS]
            avg=lambda sets:[dict(r,response_gap=float(np.mean([d[r['id']]['response_gap'] for d in sets]))) for r in sets[0].values()]
            comparisons.append(dict(variant=v,weight=w,selection=sel,versus='soft_kl',**paired_cluster_difference(avg(lefts),avg(rights),'response_gap',10000,20260925)))
    write_json(OUT/'paired_comparisons.json',comparisons)
    print('54 selected checkpoints evaluated; comparisons written',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['resource','train','evaluate']);a=p.parse_args()
    if a.command=='resource':train('soft_kl',0.,42)
    elif a.command=='train':
        for v,w in GRID:
            for s in SEEDS:train(v,w,s)
    else:evaluate()
