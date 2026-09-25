"""Four-arm Helsinki service-frequency experiment, frozen people/checkpoints.

Historic simulations are reused only for identical baseline/perceived-only arms,
models, policy and seeds. New supply is never written over historical inputs.
"""
from pathlib import Path
import argparse,collections,copy,concurrent.futures,functools,json,sys,time,xml.etree.ElementTree as ET
from controlled import ROOT,OUT,read_json,write_json,write_rows,file_hash,digest
from offline_audits import RAW,rows
sys.path[:0]=[str(RAW/'scripts/revision_20260921'),str(RAW),str(RAW/'src'),str(RAW/'scripts'),str(RAW/'scripts/shanghai')]
import helsinki_execution as old
from traveler_distillation.accessibility.gtfs_accessibility import plan_accessibility
from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives
from traveler_distillation.schemas.state import UniversalTravelerState
from departure_baselines import CombinationAdapter
import numpy as np
import torch
DEST=RAW/'outputs/revision_20260925/physical_supply'
SUMMARY=OUT.parent/'physical_supply'
MODELS=['s9','independent_neural_seed42']
SEEDS=[42,2026,7,917,20260925]
ARMS=['baseline','perceived_delay','frozen_disrupted','recomputed_disrupted']
BASESUPPLY=copy.copy(old.SUPPLY)

def supply():
    DEST.mkdir(parents=True,exist_ok=True);SUMMARY.mkdir(parents=True,exist_ok=True);sp=DEST/'supply';sp.mkdir(exist_ok=True)
    if (sp/'audit.json').exists():return {**BASESUPPLY,'schedule':sp/'transitSchedule.xml','trips_by_stop':sp/'trips_by_stop.json'}
    write_json(sp/'protocol_amendment.json',dict(reason='Each MATSim transitRoute contains exactly one original GTFS trip; per-route departure thinning would be a no-op.',group='transitLine id and ordered stopFacility references',rule='sort by departure seconds and trip ID; retain indices 0,2,4,...; remove entire other transitRoute elements; singleton groups retained',frozen_before_supply_creation=True,seed=None,source_sha256=file_hash(BASESUPPLY['schedule'])))
    tree=ET.parse(BASESUPPLY['schedule']);root=tree.getroot();kept=set();removed=set();groups=[]
    def seconds(text):
        h,m,s=map(float,text.split(':'));return h*3600+m*60+s
    for line in root.findall('transitLine'):
        by=collections.defaultdict(list)
        for route in line.findall('transitRoute'):
            ds=route.findall('departures/departure');assert len(ds)==1
            signature=tuple(s.get('refId') for s in route.findall('routeProfile/stop'))
            by[signature].append((seconds(ds[0].get('departureTime')),route.get('id'),route))
        for sig,items in by.items():
            items.sort(key=lambda x:x[:2]);ret=items[::2];rem=items[1::2]
            for _,tid,r in ret:kept.add(tid)
            for _,tid,r in rem:removed.add(tid);line.remove(r)
            gaps0=np.diff([x[0] for x in items]);gaps1=np.diff([x[0] for x in ret])
            groups.append(dict(line=line.get('id'),stop_sequence_hash=digest(sig),before=len(items),after=len(ret),before_median_headway_sec=float(np.median(gaps0)) if len(gaps0) else None,after_median_headway_sec=float(np.median(gaps1)) if len(gaps1) else None))
    tree.write(sp/'transitSchedule.xml',encoding='utf-8',xml_declaration=True)
    old.StudentAdapter(old.checkpoint('s9'),device='cpu')._adapter._insert_doctype(sp/'transitSchedule.xml','<!DOCTYPE transitSchedule SYSTEM "http://www.matsim.org/files/dtd/transitSchedule_v1.dtd">')
    index=read_json(BASESUPPLY['trips_by_stop']);original_ids={t['trip_id'] for vals in index.values() for t in vals};assert original_ids==kept|removed
    filtered={sid:[t for t in ts if t['trip_id'] in kept] for sid,ts in index.items()}
    write_json(sp/'trips_by_stop.json',filtered);write_rows(sp/'headway_groups.jsonl',groups)
    new_ids={t['trip_id'] for vals in filtered.values() for t in vals};assert new_ids==kept and not kept&removed and removed
    audit=dict(original_trips=len(original_ids),retained_trips=len(kept),removed_trips=len(removed),retained_fraction=len(kept)/len(original_ids),groups=len(groups),singleton_groups=sum(g['before']==1 for g in groups),schedule_sha256=file_hash(sp/'transitSchedule.xml'),original_schedule_sha256=file_hash(BASESUPPLY['schedule']),index_sha256=file_hash(sp/'trips_by_stop.json'),kept_trip_ids=sorted(kept),removed_trip_ids=sorted(removed),checks=dict(real_supply_changed=True,index_matches_schedule=True,physical_network_unchanged=True,vehicle_types_unchanged=True))
    write_json(sp/'audit.json',audit);write_json(SUMMARY/'supply_audit.json',{k:v for k,v in audit.items() if not k.endswith('trip_ids')});print('Supply modified',len(original_ids),'->',len(kept),'trips',flush=True)
    return {**BASESUPPLY,'schedule':sp/'transitSchedule.xml','trips_by_stop':sp/'trips_by_stop.json'}

def modified_index(idx,new):
    obj=copy.copy(idx);obj.trips_by_stop=read_json(new['trips_by_stop']);obj.trip_seq={};obj.stop_sorted={};obj.stop_trip_map={}
    for sid,entries in obj.trips_by_stop.items():
        obj.stop_sorted[sid]=sorted(entries,key=lambda t:t['dep_sec']);obj.stop_trip_map[sid]={t['trip_id']:t for t in entries}
        for t in entries:obj.trip_seq.setdefault(t['trip_id'],{})[sid]=(t['seq_idx'],t['arr_sec'],t['dep_sec'])
    obj.trip_stops_sorted={tid:sorted((s,sid,a,d) for sid,(s,a,d) in vals.items()) for tid,vals in obj.trip_seq.items()}
    return obj

class Experiment(old.Experiment):
    def prepare_states(self,arm):
        path=DEST/'states'/f'{arm}.jsonl'
        if path.exists():rs=rows(path)
        else:
            source='C3_transit_delay' if arm=='perceived_delay' else 'C0_baseline'
            rs=copy.deepcopy(rows(RAW/f'outputs/revision_20260921/execution/states/{source}.jsonl'))
            if arm=='recomputed_disrupted':
                for i,r in enumerate(rs):
                    st=UniversalTravelerState.model_validate(r['state']);assert st.context.transit_delay_min==0
                    acc=plan_accessibility(self.idx,r['origin_node'],r['destination_node'],st.trip.desired_departure_min*60)
                    st.alternatives=build_real_alternatives(st.persona,st.trip,st.context,self.idx,r['origin_node'],r['destination_node'],acc)
                    r['state']=st.model_dump(mode='json');r['recomputed_accessibility']=acc
                    if (i+1)%100==0:print('recomputed LOS',i+1,flush=True)
            path.parent.mkdir(parents=True,exist_ok=True);write_rows(path,rs)
        assert len(rs)==1000;self.records[arm]=rs;self.states[arm]=[UniversalTravelerState.model_validate(r['state']) for r in rs]

    def prediction(self,model,card):
        path=DEST/'predictions'/model/f'{card}.jsonl'
        if path.exists():return rows(path)
        predictor=self.s9 if model=='s9' else CombinationAdapter(RAW/f'outputs/revision_20260921/baselines/train/{model}/best_departure.pt')
        dec=predictor.predict(self.states[card]);rs=[dict(person_id=st.persona.persona_id,**d) for st,d in zip(self.states[card],dec)]
        path.parent.mkdir(parents=True,exist_ok=True);write_rows(path,rs);return rs

def run():
    new=supply();old.OUT=DEST;torch.set_num_threads(2);ex=Experiment(1000);original_idx=ex.idx;newidx=modified_index(ex.idx,new)
    write_json(SUMMARY/'protocol.json',dict(models=MODELS,seeds=SEEDS,arms=ARMS,n_initial=1000,policy='planner_sampling',timing='independent neural seed42 for MNL; frozen S9 own timing',supply_arm='baseline/perceived_delay use original supply; frozen_disrupted/recomputed_disrupted use retained every-other service',LOS='computed at desired departure, behavior evaluated once, route feasibility at model-adjusted departure; no equilibrium or iterative feedback claim',trip_time_limit='Historical network link free-speed execution retained; trip times not calibrated walking/cycling travel-time benefits.',raw_output=str(DEST),script_sha256=file_hash(__file__),reused_source_sha256=file_hash(old.__file__)))
    for arm in ARMS:
        old.SUPPLY=BASESUPPLY if arm in ARMS[:2] else new;ex.idx=original_idx if arm in ARMS[:2] else newidx;ex.route=functools.lru_cache(maxsize=100000)(ex._route)
        ex.prepare_states(arm);jobs=[]
        # All tasks for an arm finish before changing the module supply pointer.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for model in MODELS:
                dec=ex.prediction(model,arm)
                if arm in ARMS[:2]:
                    oldcard='C0_baseline' if arm=='baseline' else 'C3_transit_delay'
                    pre=rows(RAW/f'outputs/revision_20260921/execution/preflight/{model}/{oldcard}.jsonl')
                    assert all(r['person_id']==d['person_id'] and abs(r['actual_departure_min']-(st.trip.desired_departure_min+d['departure_time_shift_min']))<1e-5 for r,d,st in zip(pre,dec,ex.states[arm]))
                    dst=DEST/'preflight'/model/f'{arm}.jsonl';dst.parent.mkdir(parents=True,exist_ok=True);write_rows(dst,pre)
                else:pre=ex.preflight(model,arm,dec)
                for seed in SEEDS:
                    historic=RAW/f'outputs/revision_20260921/execution/runs/{model}__{"C0_baseline" if arm=="baseline" else "C3_transit_delay"}__planner_sampling__seed{seed}'
                    if arm in ARMS[:2] and (historic/'result.json').exists():
                        result=read_json(historic/'result.json');assert result['exit_code']==0 and result['n_initial']==1000
                        prev=rows(historic/'person_ledger.jsonl');assert len(prev)==len(dec)==len(pre)
                        assert all(p['person_id']==d['person_id']==f['person_id'] and abs(p['departure_shift_min']-d['departure_time_shift_min'])<1e-5 and p['planner_mask']==f['planner_mask'] and max(abs(p['original_probability'].get(m,0)-d['mode_probabilities'].get(m,0)) for m in old.MODES)<1e-6 for p,d,f in zip(prev,dec,pre))
                        dest=DEST/'runs'/f'{model}__{arm}__planner_sampling__seed{seed}';dest.mkdir(parents=True,exist_ok=True)
                        write_json(dest/'reused.json',dict(source=str(historic),result_sha256=file_hash(historic/'result.json'),ledger_sha256=file_hash(historic/'person_ledger.jsonl'),state_sha256=file_hash(DEST/'states'/f'{arm}.jsonl'),checked='same people, probabilities, timing, planner masks, fixed physical supply and model artifact; common random number function identical'))
                        print('Reused',model,arm,seed,flush=True)
                    else:jobs.append(pool.submit(ex.run,model,arm,'planner_sampling',seed,dec,pre))
            for f in concurrent.futures.as_completed(jobs):f.result()
        aggregate()
    aggregate(require_complete=True)

def aggregate(require_complete=False):
    data={};results=[]
    for model in MODELS:
        for arm in ARMS:
            for seed in SEEDS:
                folder=DEST/'runs'/f'{model}__{arm}__planner_sampling__seed{seed}'
                ref=folder/'reused.json';source=Path(read_json(ref)['source']) if ref.exists() else folder
                if not (source/'result.json').exists():continue
                r=read_json(source/'result.json')
                if r['exit_code']!=0:continue
                ledger=rows(source/'person_ledger.jsonl');assert len(ledger)==1000
                data[model,arm,seed]=ledger;results.append(dict(model=model,arm=arm,seed=seed,reused=ref.exists(),source=str(source),result_sha256=file_hash(source/'result.json'),ledger_sha256=file_hash(source/'person_ledger.jsonl')))
    if require_complete:assert len(data)==40,len(data)
    write_json(SUMMARY/'run_inventory.json',results)
    comparisons=[]
    for model in MODELS:
        for arm in ARMS[1:]:
            pairs=[]
            for seed in SEEDS:
                if (model,arm,seed) not in data or (model,'baseline',seed) not in data:continue
                b=data[model,'baseline',seed];d=data[model,arm,seed];assert [r['person_id'] for r in b]==[r['person_id'] for r in d]
                def fields(r):return np.array([r['original_probability']['pt'],r['filtered_probability']['pt'],r['assigned_mode']=='pt',r['routed_mode']=='pt',r['events']['outbound_pt_boardings']>0,r['events']['outbound_arrived']],float)*100
                pairs.append(np.array([fields(y)-fields(x) for x,y in zip(b,d)]))
            if not pairs:continue
            array=np.array(pairs);avg=array.mean(0);rng=np.random.default_rng(20260925);ix=rng.integers(1000,size=(4000,1000));boot=avg[ix].mean(1)
            comparisons.append(dict(model=model,arm=arm,seeds=len(pairs),n_initial=1000,metrics={k:dict(mean_pp=float(avg[:,j].mean()),paired_person_ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),assignment_seed_sd=float(array[:,:,j].mean(1).std(ddof=1)) if len(pairs)>1 else None) for j,k in enumerate(['raw_pt','filtered_pt','assigned_pt','routed_pt','boarded_pt','completed'])}))
    write_json(SUMMARY/'comparisons.json',dict(completed_runs=len(data),expected_runs=40,comparisons=comparisons,uncertainty='Paired initial-person bootstrap after averaging five paired assignment seeds; conditional on two fixed checkpoints, numeric population and synthetic disruption.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['supply','run','aggregate']);a=p.parse_args()
    if a.command=='supply':supply()
    elif a.command=='run':run()
    else:aggregate(True)
