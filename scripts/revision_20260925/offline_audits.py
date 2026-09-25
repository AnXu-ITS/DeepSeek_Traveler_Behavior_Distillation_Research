"""Frozen-source split, disjoint-repeat and paired stagewise re-analyses."""
from pathlib import Path
import argparse,collections,itertools,json,sys,os
import numpy as np
from controlled import ROOT,BUNDLE,OUT,SEEDS,GRID,read_json,write_json,write_rows,file_hash
RAW=Path(os.environ.get('AIT_RAW_ROOT',str(ROOT.parents[2]/'蒸馏出行意图workbench'))).resolve()
DEST=OUT.parent/'audit'
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def save(name,x):DEST.mkdir(parents=True,exist_ok=True);write_json(DEST/name,x)

def leakage():
    splits={s:{k:rows(BUNDLE/f'{s}_{k}.jsonl') for k in ['endpoints','pairs','interactions','units']} for s in ['train','val','test']}
    excluded={r['id'] for r in read_json(BUNDLE/'manifest.json')['exclusions'] if 'unseen combination' in r['reason']}
    allheld=excluded|{e['id'] for e in splits['test']['endpoints'] if e['bucket']=='joint_unseen'}
    checks=[]
    for s,d in splits.items():
        ids={e['id'] for e in d['endpoints']}
        refs={i for u in d['units'] for i in u['ids']}|{i for p in d['pairs'] for i in [p['base'],p['cf']]}|{i for r in d['interactions'] for i in r['endpoints']}
        semantic=sum(e['state']['context']['fare_multiplier']>=1.5 and e['state']['context']['road_congestion']>=.6 for e in d['endpoints'])
        assert s=='test' or semantic==0
        checks.append(dict(split=s,n_endpoints=len(ids),n_pairs=len(d['pairs']),n_interactions=len(d['interactions']),all_references_local=refs<=ids,heldout_endpoints=len(ids&allheld),heldout_references=len(refs&allheld),excluded_references=len(refs&excluded),cross_source_semantic_fare_congestion_endpoints=semantic))
    overlap=[]
    for a,b in itertools.combinations(splits,2):
        for field in ['persona','input_hash']:
            shared={e[field] for e in splits[a]['endpoints']}&{e[field] for e in splits[b]['endpoints']}
            overlap.append(dict(left=a,right=b,field=field,n_shared=len(shared)))
    lineage=[]
    for p in (ROOT/'outputs/matched_response_v1/train').glob('*/run.json'):
        r=read_json(p);lineage.append(dict(run=p.parent.name,initialization=r['initialization'],bundle=r['bundle_id'],source_code=r['code_id']))
    assert all(r['all_references_local'] and not r['excluded_references'] and (r['split']=='test' or r['heldout_references']==0) for r in checks)
    assert not any(r['n_shared'] for r in overlap)
    held=[]
    for p in (ROOT/'outputs/matched_response_v1/test').glob('*/metrics.json'):
        r=read_json(p);m=r.get('metrics',r)
        held.append(dict(run=p.parent.name,**m['by_bucket']['joint_unseen']))
    save('combination_leakage.json',dict(checks=checks,overlap=overlap,lineage=lineage,heldout_results=held,auxiliary='Controlled loss_for_units uses endpoint KL, departure Huber and declared pair loss only. No auxiliary updater or pretrained checkpoint is used.',source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/'src/traveler_distillation/matched_response/data.py',ROOT/'src/traveler_distillation/matched_response/experiment.py',BUNDLE/'manifest.json']},scope='fare x congestion combination disjoint only; constituent axes occur in training; frozen S9 has different lineage and is not covered by this guarantee'))

def repeats():
    test=rows(BUNDLE/'test_endpoints.jsonl');pairs=rows(BUNDLE/'test_pairs.jsonl');record={};completion={};hashes={}
    paths={'legacy':('data/student_v0_3_s3/aggregated_teacher_dataset.jsonl','data/student_v0_3_s3/repeat_records.jsonl'),'joint':('data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl','data/student_s5_joint/repeat_records.jsonl'),'accessibility':('data/singapore_accessibility/states_with_teacher.jsonl','data/singapore_accessibility/repeat_records.jsonl')}
    for source,(ap,rp) in paths.items():
        agg={r['sample_id']:r for r in rows(RAW/ap)};raw={r['completion_id']:r for r in rows(RAW/rp) if r.get('action') and r.get('completion_id')}
        hashes[ap]=file_hash(RAW/ap);hashes[rp]=file_hash(RAW/rp)
        for e in [e for e in test if e['source']==source]:
            a=agg[e['original_id']];ids=a['aggregation_metadata']['source_completion_ids']
            if not all(i in raw for i in ids):continue
            assert len(ids)==len(set(ids))
            mat=np.array([[raw[i]['action']['mode_probabilities'].get(m,0.) for m in ['car','pt','bike','walk']] for i in ids]);mat/=mat.sum(1,keepdims=True)
            assert np.max(abs(mat.mean(0)-e['teacher_probs']))<2e-6
            record[e['id']]=mat;completion[e['id']]=ids
    pairs=[p for p in pairs if p['base'] in record and p['cf'] in record]
    by={e['id']:e for e in test};ids=sorted(record);index={x:i for i,x in enumerate(ids)};base=[index[p['base']] for p in pairs];cf=[index[p['cf']] for p in pairs]
    masks=np.array([[a['available'] for a in by[i]['state']['alternatives']] for i in ids]);mask=masks[base]|masks[cf]
    # Exhaust all ordered, disjoint equal-size allocations locally, cycle the
    # endpoint lists through 30 shared rotations. Rotations are correlated.
    allocations={}
    for k in sorted({len(x) for x in record.values()}):
        q=k//2;seq=[]
        for a in itertools.combinations(range(k),q):
            for b in itertools.combinations([i for i in range(k) if i not in a],q):seq.append((a,b))
        allocations[k]=seq
    models={}
    for p in (ROOT/'outputs/matched_response_v1/test').glob('*/predictions.jsonl'):
        pr={r['id']:r for r in rows(p)};mat=np.array([pr[i]['student'] for i in ids]);models[p.parent.name]=mat[cf]-mat[base]
    pr={r['id']:r for r in rows(RAW/'outputs/reviewer_closure_20260920/baseline/predictions.jsonl')};mat=np.array([pr[i]['student'] for i in ids]);models['mnl_s']=mat[cf]-mat[base]
    detail=[];partition=[]
    for rot in range(30):
        ta=[];tb=[]
        for eid in ids:
            mat=record[eid];a,b=allocations[len(mat)][rot%len(allocations[len(mat)])]
            assert not set(a)&set(b) and len(a)==len(b)
            ta.append(mat[list(a)].mean(0));tb.append(mat[list(b)].mean(0))
            partition.append(dict(rotation=rot,id=eid,subset_a=[completion[eid][i] for i in a],subset_b=[completion[eid][i] for i in b]))
        da=np.array(ta)[cf]-np.array(ta)[base];db=np.array(tb)[cf]-np.array(tb)[base]
        def gap(d):return (abs(d)*mask).sum(1)/mask.sum(1)
        selfgap=gap(da-db)
        for model,sd in models.items():
            ga=gap(sd-da);gb=gap(sd-db)
            for source in ['all']+sorted({p['source'] for p in pairs}):
                ix=[i for i,p in enumerate(pairs) if source=='all' or p['source']==source]
                detail.append(dict(rotation=rot,model=model,source=source,n_pairs=len(ix),teacher_self_gap=float(selfgap[ix].mean()),student_to_a=float(ga[ix].mean()),student_to_b=float(gb[ix].mean())))
    DEST.mkdir(exist_ok=True,parents=True);write_rows(DEST/'repeat_disjoint_partitions.jsonl',partition);write_rows(DEST/'repeat_disjoint_metrics.jsonl',detail)
    save('repeat_disjoint_summary.json',dict(n_endpoints=len(ids),n_pairs=len(pairs),repeat_counts=dict(collections.Counter(len(v) for v in record.values())),allocations={k:len(v) for k,v in allocations.items()},source_sha256=hashes,scope='Equal-size disjoint Teacher repeat sensitivity, shared endpoint partitions and model comparisons. Thirty correlated allocations are not 30 independent experiments or a precise noise ceiling; no mechanism repeat records available.',summaries=[dict(model=m,**{key:float(np.mean([r[key] for r in detail if r['model']==m and r['source']=='all'])) for key in ['teacher_self_gap','student_to_a','student_to_b']}) for m in models]))

def stagewise():
    runroot=RAW/'outputs/revision_20260921/execution/runs';group=collections.defaultdict(list);sources={}
    for p in runroot.glob('*/result.json'):
        r=read_json(p)
        if r.get('exit_code')==0:group[r['model'],r['policy'],r['card']].append(p.parent)
    result=[];unitrows=[]
    for (model,policy,card),folders in sorted(group.items()):
        if card!='C0_baseline':continue
        delays=group[model,policy,'C3_transit_delay'];lookup={read_json(p/'result.json')['sampling_seed']:p for p in delays};arrays=[]
        for folder in sorted(folders):
            seed=read_json(folder/'result.json')['sampling_seed'];other=lookup[seed]
            b=rows(folder/'person_ledger.jsonl');d=rows(other/'person_ledger.jsonl');assert [r['person_id'] for r in b]==[r['person_id'] for r in d]
            f=lambda r:np.array([r['original_probability']['pt'],r['filtered_probability']['pt'],r['assigned_mode']=='pt',r['routed_mode']=='pt',r['events']['outbound_pt_boardings']>0],float)
            arrays.append(np.array([f(y)-f(x) for x,y in zip(b,d)])*100)
            sources[str(folder.relative_to(RAW))]=file_hash(folder/'person_ledger.jsonl');sources[str(other.relative_to(RAW))]=file_hash(other/'person_ledger.jsonl')
        arr=np.array(arrays);avg=arr.mean(0);stages=['raw','feasibility','assignment','routing','boarding'];rng=np.random.default_rng(20260925)
        ix=rng.integers(len(b),size=(4000,len(b)));boot=avg[ix].mean(1)
        for j,stage in enumerate(stages):
            delta=avg[:,j] if j==0 else avg[:,j]-avg[:,j-1];bs=boot[:,j] if j==0 else boot[:,j]-boot[:,j-1]
            result.append(dict(model=model,policy=policy,stage=stage,n_initial=len(b),sampling_seeds=len(arrays),response_pp=float(avg[:,j].mean()),response_ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),increment_from_previous_stage_pp=float(delta.mean()),increment_ci95=np.quantile(bs,[.025,.975]).tolist(),sampling_seed_sd_pp=float(arr[:,:,j].mean(1).std(ddof=1)) if len(arr)>1 else None))
        for pid,row in zip([r['person_id'] for r in b],avg):unitrows.append(dict(model=model,policy=policy,person=pid,seed_averaged_response_pp=dict(zip(stages,row.tolist()))))
    save('stagewise_summary.json',dict(results=result,scope='Average paired assignment seeds within person, then resample whole paired persons; conditional on each checkpoint and fixed population, not field or training uncertainty.',source_sha256=sources));write_rows(DEST/'stagewise_person_units.jsonl',unitrows)

if __name__=='__main__':
    leakage();print('Leakage audit complete',flush=True)
    repeats();print('Disjoint repeat analysis complete',flush=True)
    stagewise();print('Stagewise paired analysis complete',flush=True)
