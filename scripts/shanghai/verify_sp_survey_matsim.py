"""Verify executed survey identities/OD/events and compare saved S9 probabilities."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
import zstandard

from sp_survey_matsim import ROOT, ORDER, MODES, sha, readj, jl, writej, writejl

def audit_events(path, expected):
    people={pid:dict(departures=0,arrivals=0,outbound_arrived=0,return_arrived=0,
        pt_boardings=0,stuck=0,outbound_modes=set(),outbound_pt_boardings=0) for pid in expected}
    event_types=Counter()
    with path.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as f:
        for _,e in ET.iterparse(f,events=('end',)):
            if e.tag!='event': continue
            kind=e.get('type');pid=e.get('person');event_types[kind]+=1
            if pid in people:
                p=people[pid]
                if kind=='departure':
                    p['departures']+=1
                    if not p['outbound_arrived']: p['outbound_modes'].add(e.get('legMode'))
                elif kind=='arrival': p['arrivals']+=1
                elif kind=='actstart':
                    if e.get('actType')=='shop': p['outbound_arrived']+=1
                    elif e.get('actType')=='home': p['return_arrived']+=1
                elif kind=='PersonEntersPtVehicle':
                    p['pt_boardings']+=1
                    if not p['outbound_arrived']: p['outbound_pt_boardings']+=1
                elif kind=='stuckAndAbort': p['stuck']+=1
            e.clear()
    for p in people.values():
        modes=p['outbound_modes']
        p['outbound_mode']='pt' if p['outbound_pt_boardings'] else ('car' if 'car' in modes else ('bike' if 'bike' in modes else ('walk' if 'walk' in modes else 'none')))
        p['outbound_modes']=sorted(modes)
    return people,event_types

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'outputs/shanghai_survey_matsim_321x10_v1')
    args=ap.parse_args();run=args.run
    protocol=readj(run/'protocol.json');od={r['respondent_id']:r for r in readj(run/'od_assignments.json')}
    ids=set(od);old={(r['respondent_id'],r['card_id']):r for r in jl(ROOT/'outputs/shanghai_sp_v1/predictions/s9_p042.jsonl')}
    baseline={(r['respondent_id'],r['card_id']):r['state'] for r in jl(ROOT/'outputs/shanghai_sp_v1/states_p042.jsonl')}
    checks=[];all_new={};artifacts=[]
    for card in ORDER:
        p=run/card;s=readj(p/'run_summary.json');st=jl(p/'states.jsonl')
        manifest=readj(p/'adapter_manifest.json')['decisions'];byid={r['person_id']:r for r in manifest}
        expected=set(byid);assert len(expected)==321
        population=ET.parse(p/'population.xml').getroot()
        popids={x.get('id') for x in population.findall('person')}
        assert popids==expected
        people,types=audit_events(p/'output/ITERS/it.0/0.events.xml.zst',expected)
        personrows=[dict(person_id=pid,respondent_id=byid[pid]['respondent_id'],**v) for pid,v in people.items()]
        writejl(p/'verified_person_events.jsonl',personrows)
        fixed_fields=True
        for r in st:
            b=baseline[(r['respondent_id'],card)];n=r['state']
            fixed_fields &= n['persona']==b['persona'] and n['trip']==b['trip']
            fixed_fields &= all(a['available']==ba['available'] and a['monetary_cost']==ba['monetary_cost'] for a,ba in zip(n['alternatives'],b['alternatives']))
        item=dict(card=card,exit_code=s['matsim']['exit_code'],planned_people=len(expected),
            observed_departing_people=sum(v['departures']>0 for v in people.values()),
            outbound_completed_people=sum(v['outbound_arrived']==1 for v in people.values()),
            return_completed_people=sum(v['return_arrived']==1 for v in people.values()),
            unmatched_leg_counts=sum(v['departures']!=v['arrivals'] for v in people.values()),
            stuck_people=sum(v['stuck']>0 for v in people.values()),
            executed_outbound_counts=dict(Counter(v['outbound_mode'] for v in people.values())),
            routed_vs_executed_disagreements=sum(v['outbound_mode']!=byid[pid]['routed_outbound_mode'] for pid,v in people.items()),
            od_matches=all((r['origin_node'],r['destination_node'])==(od[r['respondent_id']]['origin_node'],od[r['respondent_id']]['destination_node']) for r in manifest),
            persona_trip_prices_masks_unchanged=bool(fixed_fields),
            pt_boardings=sum(v['pt_boardings'] for v in people.values()),
            event_count=sum(types.values()))
        item['pass']=all([item['exit_code']==0,item['planned_people']==321,item['observed_departing_people']==321,
            item['outbound_completed_people']==321,item['return_completed_people']==321,
            item['unmatched_leg_counts']==0,item['stuck_people']==0,item['od_matches'],
            item['persona_trip_prices_masks_unchanged'],item['routed_vs_executed_disagreements']==0])
        checks.append(item)
        for r in jl(p/'predictions.jsonl'): all_new[(r['respondent_id'],card)]=r
        warnings=(p/'output/logfileWarningsErrors.log').read_text(encoding='utf-8')
        item['log_warning_classes']=dict(Counter(re.findall(r'(?:WARN|ERROR)\s+([^ ]+)',warnings)))
        item['exception_messages']=list(dict.fromkeys(x for x in warnings.splitlines() if x.startswith('java.') and 'Exception' in x))
        for f in ('states.jsonl','predictions.jsonl','population.xml','config.xml','adapter_manifest.json','run_summary.json','verified_person_events.jsonl','output/ITERS/it.0/0.events.xml.zst'):
            artifacts.append(dict(path=str((p/f).relative_to(run)),sha256=sha(p/f)))
        print(card,'event verification',item['pass'],'people',item['outbound_completed_people'],item['return_completed_people'],flush=True)
    vectors=[]
    for rid in sorted(ids):
        rows=[]
        for card in ORDER:
            k=(rid,card);a=old[k];b=all_new[k];mode=a['chosen_mode']
            if a['choice_status']!='selected' or a['probabilities'].get(mode,0)==0: continue
            assert mode in b['probabilities']
            pa=np.array([a['probabilities'].get(m,0) for m in MODES]);pb=np.array([b['probabilities'].get(m,0) for m in MODES]);y=MODES.index(mode)
            rows.append([float(pb.argmax()==y)-float(pa.argmax()==y),
                -np.log(max(pb[y],1e-8))+np.log(max(pa[y],1e-8)),
                float(((pb-np.eye(4)[y])**2).sum()-((pa-np.eye(4)[y])**2).sum()),pb[1]-pa[1]])
        vectors.append(np.mean(rows,axis=0))
    x=np.array(vectors);ix=np.random.default_rng(917).integers(len(x),size=(4000,len(x)))
    comparison={k:dict(mean=float(x[:,i].mean()),ci95=np.quantile(x[ix,i].mean(axis=1),[.025,.975]).tolist()) for i,k in enumerate(['accuracy','nll','brier','pt_probability'])}
    report=dict(cards=checks,n_cards=len(checks),all_execution_checks_pass=all(r['pass'] for r in checks),
        total_respondent_tasks=3210,total_completed_outbound=sum(x['outbound_completed_people'] for x in checks),
        total_completed_return=sum(x['return_completed_people'] for x in checks),
        paired_new_minus_original=comparison,comparison_unit='equal-weight respondent, all valid tasks paired; 4000 bootstrap draws, seed917',
        source_hashes_unchanged=all(sha(ROOT/p)==h for p,h in protocol['source_sha256'].items()),
        frozen_output_artifacts=artifacts,
        warning_interpretation='Network walk access legs trigger legacy PrepareForSim logs; optional experienced-plan/score dumps report missing files with writePlansInterval=0. Raw iteration events and person completions independently verified; not a warning-free run.')
    writej(run/'execution_verification.json',report)
    with (run/'scenario_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        rows=[]
        for c in ORDER:
            s=readj(run/c/'run_summary.json');v=next(x for x in checks if x['card']==c)
            rows.append(dict(card=c,n=321,accuracy=s['human_alignment']['accuracy'],
                human_pt=s['human_alignment']['human_pt'],model_pt_probability=s['human_alignment']['model_pt'],
                student_pt_argmax=s['student_counts'].get('pt',0),actual_outbound_pt=v['executed_outbound_counts'].get('pt',0),
                outbound_arrived=v['outbound_completed_people'],return_arrived=v['return_completed_people'],
                car_no_route_fallback_legs=s['fallback_counts'].get('no_route_car',0),exit_code=s['matsim']['exit_code']))
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cards','frozen_output_artifacts')},ensure_ascii=False,indent=2))
    return 0 if report['all_execution_checks_pass'] and report['source_hashes_unchanged'] else 1

if __name__=='__main__':raise SystemExit(main())
