#!/usr/bin/env python
"""321 actual respondents x 10 SP tasks, network-conditioned S9 and MATSim.

No training. OD is deterministic assumed 6 km, shared across cards. The Shanghai
network supplies baseline travel times and PT components. Card prices and additive
perturbations remain explicit assumptions. MATSim executes the resulting choices
on the retained network/schedule; it does not physically implement card disruptions.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from reference_pipeline.matsim_adapter import build_supply_view, parse_events, run_matsim
from reference_pipeline.config import MatsimConfig
from reference_pipeline.student_adapter import StudentAdapter
from traveler_distillation.accessibility.gtfs_accessibility import SupplyIndex, plan_accessibility
from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives
from traveler_distillation.matsim.adapter import _activity_link, _ACTIVITY_DURATIONS_MIN, _LEG_MIN, _to_hms
from traveler_distillation.schemas.state import UniversalTravelerState

ORDER = ['B0','W1','D1','WD1','F1','P1','R1','A_WALK','A_WAIT','A_TRANSFER']
MODES = ['car','pt','bike','walk']
SH = ROOT / 'data/shanghai/transit'
SUPPLY = dict(network=SH/'network_with_transit.xml', schedule=SH/'transitSchedule.xml',
              vehicles=SH/'transitVehicles.xml', stops=SH/'prep_stops.jsonl',
              snapping=SH/'stop_snapping_report.json', trips_by_stop=SH/'trips_by_stop.json',
              activity_nodes=SH/'activity_nodes.json')
CHECKPOINT = ROOT/'releases/s9_supply_aware_v2/checkpoint/model.pt'
EXPECTED_CKPT = '6af79b44bc699c00cc8e913da15043a43398829fdc8bbb4dad2ddcb8e11d31e6'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def readj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def jl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def writej(p, obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def writejl(p, rows): Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def network_card_state(base, template, baseline_card):
    """Apply card deltas exactly once to the network baseline, retaining its mask.

    PT-infeasible sentinels stay infeasible; no imputed PT connection is created.
    Monetary amounts come from the SP card, not Singapore's production cost formula.
    """
    state=base.model_copy(deep=True)
    state.trip=template.trip.model_copy(deep=True)
    state.context=template.context.model_copy(deep=True)
    state.context.context_id='SH_NETWORK_'+template.context.context_id
    b={a.mode:a for a in baseline_card.alternatives}
    t={a.mode:a for a in template.alternatives}
    numeric=['travel_time_min','access_time_min','egress_time_min','wait_time_min',
             'in_vehicle_time_min','transfer_time_min','transfers','reliability_delay_min']
    for a in state.alternatives:
        a.monetary_cost=t[a.mode].monetary_cost
        a.available=t[a.mode].available
        # Weather enters the context, as in the original card table (no extra
        # weather time multiplier is added on top of its explicit numeric table).
        a.weather_exposure=t[a.mode].weather_exposure
        if a.mode=='pt' and not a.pt_feasible:
            continue
        for key in numeric:
            delta=getattr(t[a.mode],key)-getattr(b[a.mode],key)
            setattr(a,key,getattr(a,key)+delta)
    return UniversalTravelerState.model_validate(state.model_dump())

def assign_od(rid, nodes, xy, target_km=6.0):
    seed=int(hashlib.sha256(('shanghai-sp-od-20260919:'+rid).encode()).hexdigest()[:16],16)
    rng=random.Random(seed)
    pts=np.array([xy[n['node']] for n in nodes]); best=None
    for i in rng.sample(range(len(nodes)),min(150,len(nodes))):
        dist=np.linalg.norm(pts-pts[i],axis=1); j=int(np.argmin(np.abs(dist-target_km*1000)))
        candidate=(abs(float(dist[j])-target_km*1000),i,j,float(dist[j])/1000)
        if best is None or candidate[0]<best[0]: best=candidate
    err,i,j,actual=best
    if i==j or err>100: raise ValueError(f'OD matching exceeds 100 m for {rid}')
    return dict(respondent_id=rid,origin_node=nodes[i]['node'],destination_node=nodes[j]['node'],
                target_km=target_km,straight_line_km=actual,error_m=err,observed_od=False)

def build_plans(out, states, decisions, od, adapter, view, idx):
    g,xy,graphs,_nodes=view
    root=ET.Element('population'); manifest=[]
    fallback={'pt_fallback_walk':0,'no_route_car':0,'no_route_bike':0}
    for st,dec in zip(states,decisions):
        rid=st.persona.persona_id; a=od[rid]; home=a['origin_node']; dest=a['destination_node']
        # Explicit map preserves respondent identity and matches the legacy event
        # parser's P<digits> filter without changing model features.
        pid='P'+str(970000+int(rid[3:]))
        p=ET.SubElement(root,'person',id=pid); attrs=ET.SubElement(p,'attributes')
        ET.SubElement(attrs,'attribute',name='survey_respondent_id',**{'class':'java.lang.String'}).text=rid
        plan=ET.SubElement(p,'plan',selected='yes')
        dep=st.trip.desired_departure_min+dec['departure_time_shift_min']
        dur=_ACTIVITY_DURATIONS_MIN.get(st.trip.destination_type,120)
        start=_activity_link(g,home,dec['mode'] if dec['mode']!='pt' else 'walk')
        legs,info,mode,end=adapter._adapter._build_legs(dec['mode'],home,dest,dep*60,start,
            idx.trips_by_stop,idx.stop_node,idx.stops,graphs,300.,fallback,idx.trip_seq)
        retdep=(dep+_LEG_MIN+dur)
        ret,retinfo,retmode,finish=adapter._adapter._build_legs(mode,dest,home,retdep*60,end,
            idx.trips_by_stop,idx.stop_node,idx.stops,graphs,300.,fallback,idx.trip_seq)
        def act(kind,node,link,until=None):
            x,y=xy[node]; kw=dict(type=kind,link=link,x=f'{x:.2f}',y=f'{y:.2f}',z='0.0')
            if until is not None: kw['end_time']=_to_hms(until)
            ET.SubElement(plan,'activity',**kw)
        def add_legs(seq):
            for m,rt,txt,extra in seq:
                el=ET.SubElement(plan,'leg',mode=m)
                if txt: ET.SubElement(el,'route',type=rt,**extra).text=txt
        act('home',home,start,dep);add_legs(legs)
        act(st.trip.destination_type,dest,end,retdep);add_legs(ret);act('home',home,finish)
        manifest.append(dict(respondent_id=rid,person_id=pid,trip_id=st.trip.trip_id,
            origin_node=home,destination_node=dest,student_mode=dec['mode'],
            routed_outbound_mode=mode,routed_return_mode=retmode,
            departure_shift_min=dec['departure_time_shift_min'],outbound=info,return_leg=retinfo))
    ET.ElementTree(root).write(out/'population.xml',encoding='utf-8',xml_declaration=True)
    adapter._adapter._insert_doctype(out/'population.xml','<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">')
    adapter._adapter._write_real_config(out/'config.xml',SUPPLY['network'],SUPPLY['schedule'],SUPPLY['vehicles'],.3,.3)
    cfg=(out/'config.xml').read_text(encoding='utf-8').replace('EPSG:32648','EPSG:32651')
    cfg=cfg.replace('deleteDirectoryIfExists','failIfDirectoryExists')
    (out/'config.xml').write_text(cfg,encoding='utf-8')
    writej(out/'adapter_manifest.json',dict(decisions=manifest,fallback_counts=fallback))
    return manifest,fallback

def score(rows):
    # Identical availability semantics to old SP primary set; retain excluded rows.
    keep=[r for r in rows if r['choice_status']=='selected' and r['chosen_mode'] in r['probabilities']]
    P=np.array([[r['probabilities'].get(m,0.) for m in MODES] for r in keep])
    y=np.array([MODES.index(r['chosen_mode']) for r in keep]);Y=np.eye(4)[y]
    correct=(P.argmax(1)==y).astype(float);baseline=(y==1).astype(float);gap=P[:,1]-baseline
    buckets=defaultdict(list)
    for i,r in enumerate(keep): buckets[r['respondent_id']].append([correct[i],baseline[i],gap[i]])
    per=np.array([np.mean(v,axis=0) for _,v in sorted(buckets.items())])
    draws=np.random.default_rng(917).integers(len(per),size=(2000,len(per)))
    def interval(x): return np.quantile(x[draws].mean(axis=1),[.025,.975]).tolist()
    return dict(n_scored=len(keep),n_respondents=len(per),n_excluded=len(rows)-len(keep),
        accuracy=float(correct.mean()),nll=float(-np.log(np.clip(P[np.arange(len(y)),y],1e-8,1)).mean()),
        brier=float(((P-Y)**2).sum(1).mean()),human_pt=float(baseline.mean()),model_pt=float(P[:,1].mean()),
        respondent_mean_accuracy=float(per[:,0].mean()),
        paired_accuracy_gain_vs_always_pt=float((per[:,0]-per[:,1]).mean()),
        paired_gain_ci95=interval(per[:,0]-per[:,1]),pt_gap_respondent_mean=float(per[:,2].mean()),
        pt_gap_ci95=interval(per[:,2]),bootstrap=dict(unit='respondent',resamples=2000,seed=917))

def prepare_java():
    temp=Path(os.environ['TEMP'])/'shanghai_survey_matsim_20260919'
    release=temp/'release'; classes=temp/'classes';classes.mkdir(exist_ok=True)
    if not (release/'matsim-2026.0.jar').is_file():
        raise RuntimeError('Create the documented ASCII release junction before running Java.')
    cp=f'{classes};{release}/matsim-2026.0.jar;{release}/libs/*'
    src=ROOT/'tools/java/RunMatsimPreloaded.java'
    proc=subprocess.run(['javac','-cp',cp,'-d',str(classes),str(src)],capture_output=True,text=True)
    if proc.returncode: raise RuntimeError(proc.stderr)
    return MatsimConfig(java_xmx='4g',timeout_s=1800,classpath_override=cp)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/shanghai_survey_matsim_321x10_v1')
    ap.add_argument('--build-only',action='store_true')
    ap.add_argument('--cards',nargs='+',choices=ORDER,default=ORDER)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    source=ROOT/'outputs/shanghai_sp_v1/states_p042.jsonl'
    source_rows=jl(source); templates={(r['respondent_id'],r['card_id']):UniversalTravelerState.model_validate(r['state']) for r in source_rows}
    ids=sorted({k[0] for k in templates})
    assert len(ids)==321 and len(templates)==3210
    assert all((rid,c) in templates for rid in ids for c in ORDER)
    assert sha(CHECKPOINT)==EXPECTED_CKPT
    with (ROOT/'outputs/shanghai_sp_v1/sp_responses.csv').open(encoding='utf-8-sig',newline='') as f:
        answers={(r['respondent_id'],r['card_id']):r for r in csv.DictReader(f)}
    source_files=[source,CHECKPOINT,Path(__file__),ROOT/'outputs/shanghai_sp_v1/sp_responses.csv',
        ROOT/'docs/plans/shanghai_survey_v3/cards.json',ROOT/'reference_pipeline/student_adapter.py',
        ROOT/'src/traveler_distillation/matsim/adapter.py',ROOT/'tools/java/RunMatsimPreloaded.java']+list(SUPPLY.values())
    fingerprints={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),respondents=321,tasks=ORDER,
        source_sha256=fingerprints,household_reference=2,od='Assumed 6 km, nearest distance among 150 seeded candidate origins; no filtering by choices, predictions or PT feasibility',
        price='Retain card amounts car 15/35 and PT 4/6; monetary values are scenarios, not observed local fares',
        baseline_supply='Shanghai Xuhui OSM and retained synthetic metro timetable',
        perturbations='Add card-minus-B0 time/component differences once; context from card. A_TRANSFER means one additional transfer relative to network baseline; no route is fabricated.',
        execution='Ten independent 321-person MATSim populations, outbound and return; fixed underlying network and schedule. Card perturbations affect S9 inputs, not physical supply files.',
        departure='Frozen S9 predicts shift; original departure output has no human survey label',
        selection='All 321 people execute all tasks, including four labels excluded from old primary scoring',
        limitations=['OD, 6 km trip, household size and timetable are assumptions','Car-or-taxi versus car ownership mask mismatch retained for comparability','Network walk/bike execution uses link freespeeds as in existing production pipeline; realized times are not calibrated','Single MATSim iteration; no equilibrium or observed behavior claim'])
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        old=readj(protocol_path)
        if old['source_sha256']!=fingerprints: raise RuntimeError('Source changed; select a new output directory.')
    else: writej(protocol_path,protocol)
    print('Loading Shanghai supply...',flush=True)
    view=build_supply_view(SUPPLY['network'],SUPPLY['activity_nodes']);idx=SupplyIndex(SUPPLY)
    adapter=StudentAdapter(CHECKPOINT,device='cpu');adapter._adapter._access_max_m=700.
    original_shortest=adapter._adapter._shortest
    adapter._adapter._shortest=lru_cache(maxsize=300000)(original_shortest)
    od_path=out/'od_assignments.json'
    if od_path.exists(): od={r['respondent_id']:r for r in readj(od_path)}
    else:
        od={rid:assign_od(rid,view[3],view[1],templates[(rid,'B0')].trip.distance_km) for rid in ids}
        writej(od_path,list(od.values()))
    baseline_path=out/'network_baseline_states.jsonl'
    if baseline_path.exists():
        base={r['respondent_id']:UniversalTravelerState.model_validate(r['state']) for r in jl(baseline_path)}
    else:
        base={};access=[]
        for i,rid in enumerate(ids):
            t=templates[(rid,'B0')];o=od[rid]
            acc=plan_accessibility(idx,o['origin_node'],o['destination_node'],t.trip.desired_departure_min*60.)
            alts=build_real_alternatives(t.persona,t.trip,t.context,idx,o['origin_node'],o['destination_node'],acc)
            base[rid]=UniversalTravelerState(persona=t.persona,trip=t.trip,context=t.context,alternatives=alts)
            access.append(dict(respondent_id=rid,accessibility=acc))
            if (i+1)%40==0: print(f'Network supply {i+1}/321',flush=True)
        writejl(baseline_path,[dict(respondent_id=r,state=base[r].model_dump(mode='json')) for r in ids])
        writejl(out/'network_accessibility.jsonl',access)
    java=None if args.build_only else prepare_java()
    for card in args.cards:
        cd=out/card;cd.mkdir(exist_ok=True)
        if (cd/'run_summary.json').exists() and readj(cd/'run_summary.json').get('complete'):
            print(card,'already complete',flush=True);continue
        if (cd/'output').exists(): raise RuntimeError(f'{card}: existing unfinished MATSim output; preserve and choose a fresh directory.')
        start=time.perf_counter()
        states=[network_card_state(base[r],templates[(r,card)],templates[(r,'B0')]) for r in ids]
        writejl(cd/'states.jsonl',[dict(card_id=card,state=s.model_dump(mode='json'),**od[r]) for r,s in zip(ids,states)])
        dec=adapter.predict(states)
        rows=[dict(respondent_id=r,card_id=card,probabilities=d['mode_probabilities'],
            departure_time_shift_min=d['departure_time_shift_min'],choice_status=answers[(r,card)]['choice_status'],chosen_mode=answers[(r,card)]['chosen_mode']) for r,d in zip(ids,dec)]
        writejl(cd/'predictions.jsonl',rows)
        manifest,fallback=build_plans(cd,states,dec,od,adapter,view,idx)
        summary=dict(card=card,respondents=len(ids),states=len(states),checkpoint_sha256=EXPECTED_CKPT,
            student_counts=dict(Counter(d['mode'] for d in dec)),
            routed_outbound_counts=dict(Counter(d['routed_outbound_mode'] for d in manifest)),
            fallback_counts=fallback,pt_feasible_count=sum(s.alternatives[1].pt_feasible==1 for s in states),
            human_alignment=score(rows),complete=False,build_seconds=time.perf_counter()-start)
        writej(cd/'run_summary.json',summary)
        if args.build_only: print(card,'build complete',flush=True);continue
        print(card,'MATSim starting; choices',summary['student_counts'],flush=True)
        started=time.perf_counter();code,tail=run_matsim(cd,java)
        summary['matsim']=dict(exit_code=code,wall_seconds=time.perf_counter()-started,log_tail=tail)
        if code==0:
            metrics=parse_events(cd);summary['matsim']['events']=metrics
            summary['complete']=not metrics.get('events_missing',False)
        summary['total_seconds']=time.perf_counter()-start
        writej(cd/'run_summary.json',summary)
        print(card,'MATSim complete=',summary['complete'],'exit=',code,flush=True)
        if not summary['complete']: raise RuntimeError(f'{card} failed; inspect {cd}/java_run.log')
    summaries={c:readj(out/c/'run_summary.json') for c in ORDER if (out/c/'run_summary.json').exists()}
    allrows=[r for c in ORDER if (out/c/'predictions.jsonl').exists() for r in jl(out/c/'predictions.jsonl')]
    combined=dict(n_completed=sum(s['complete'] for s in summaries.values()),cards=summaries,
        overall_human_alignment=score(allrows),source_files_unchanged=all(sha(ROOT/p)==h for p,h in fingerprints.items()))
    writej(out/'summary.json',combined)
    print('Completed cards:',combined['n_completed'],'sources unchanged:',combined['source_files_unchanged'],flush=True)
    return 0

if __name__=='__main__': raise SystemExit(main())
