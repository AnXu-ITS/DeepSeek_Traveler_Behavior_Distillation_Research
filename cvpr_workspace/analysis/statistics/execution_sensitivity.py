"""Same Shanghai OD/supply, model x selection x route mask with common uniforms."""
from reviewer_closure import *
from functools import lru_cache
import time
from sp_survey_matsim import (SUPPLY,build_supply_view,SupplyIndex,StudentAdapter,
 CHECKPOINT,build_plans,prepare_java,run_matsim,UniversalTravelerState,_activity_link)
from verify_sp_survey_matsim import audit_events
from traveler_distillation.accessibility.gtfs_accessibility import plan_accessibility

def common_uniform(rid,seed):
 return (int(hashlib.sha256(f'reviewer-policy:{seed}:{rid}'.encode()).hexdigest()[:13],16)+.5)/16**13
def select(prob,mask,rid,seed,sample):
 p=np.array([prob.get(m,0) for m in MODES]);p*=mask
 if p.sum()<=0:raise ValueError('No feasible positive-probability alternative')
 p/=p.sum();j=min(int(np.searchsorted(np.cumsum(p),common_uniform(rid,seed))),3) if sample else int(p.argmax())
 return MODES[j],{m:float(p[i]) for i,m in enumerate(MODES) if mask[i]}

def main():
 torch.set_num_threads(4);base=OUT/'execution';base.mkdir(exist_ok=True)
 view=build_supply_view(SUPPLY['network'],SUPPLY['activity_nodes']);idx=SupplyIndex(SUPPLY)
 adapter=StudentAdapter(CHECKPOINT,device='cpu');adapter._adapter._access_max_m=700.
 adapter._adapter._shortest=lru_cache(maxsize=300000)(adapter._adapter._shortest)
 od={r['respondent_id']:r for r in read(NET/'od_assignments.json')}
 linear=LinearUtility().load();java=prepare_java();results=[]
 print('Execution sensitivity initialized',flush=True)
 for model in ['s9','supply_mnl']:
  for card in read(OUT/'protocol.json')['execution']['cards']:
   states=[UniversalTravelerState.model_validate(r['state']) for r in jl(NET/card/'states.jsonl')]
   saved=jl(NET/card/'predictions.jsonl') if model=='s9' else jl(OUT/f'baseline/survey/{card}.jsonl')
   decisions=[dict(mode=max(r['probabilities'],key=r['probabilities'].get),mode_probabilities=r['probabilities'],departure_time_shift_min=r['departure_time_shift_min']) for r in saved]
   prepath=base/f'{model}_{card}_preflight.json'
   if prepath.exists():preflight=read(prepath)
   else:
    preflight=[]
    for st,dec in zip(states,decisions):
     rid=st.persona.persona_id;o=od[rid];home=o['origin_node'];dest=o['destination_node'];shift=dec['departure_time_shift_min'];requested=st.trip.desired_departure_min;adjusted=requested+shift
     modes={}
     for a in st.alternatives:
      if not a.available:continue
      initial=_activity_link(view[0],home,a.mode if a.mode!='pt' else 'walk');versions={}
      for label,departure in [('requested',requested),('adjusted',adjusted)]:
       legs,info,routed,end=adapter._adapter._build_legs(a.mode,home,dest,departure*60,initial,idx.trips_by_stop,idx.stop_node,idx.stops,view[2],300.,collections.Counter(),idx.trip_seq)
       # No persistent route cache: key includes model/card, OD, time and mode.
       versions[label]=dict(routed_mode=routed,info=info,route_signature=hashlib.sha256(json.dumps(legs,sort_keys=True).encode()).hexdigest())
      modes[a.mode]=dict(**versions,feasible=versions['adjusted']['routed_mode']==a.mode,route_changed=versions['requested']['route_signature']!=versions['adjusted']['route_signature'])
     acc=plan_accessibility(idx,home,dest,adjusted*60)
     preflight.append(dict(respondent_id=rid,requested_departure_min=requested,adjusted_departure_min=adjusted,shift_min=shift,supply_evaluated_at_min=requested,route_evaluated_at_min=adjusted,adjusted_supply_accessibility=acc,modes=modes))
    write(prepath,preflight)
   for maskname in ['soft','planner_mask']:
    for sample,seed in [(False,0)]+[(True,s) for s in [42,2026,7]]:
     name=f'{model}_{card}_{"sample" if sample else "argmax"}_{maskname}_seed{seed}';cd=base/name
     if (cd/'verified.json').exists():results.append(read(cd/'verified.json'));continue
     cd.mkdir(exist_ok=True)
     if (cd/'output').exists():raise RuntimeError(f'Unfinished existing output: {cd}')
     if (cd/'failure.json').exists() and (cd/'java_run.log').exists() and not (cd/'pre_resume_failure.log').exists():
      (cd/'pre_resume_failure.log').write_bytes((cd/'java_run.log').read_bytes())
     chosen=[];masked_count=0
     for st,d,pr in zip(states,decisions,preflight):
      mask=np.array([a.available and (maskname=='soft' or pr['modes'][a.mode]['feasible']) for a in st.alternatives]);masked_count+=sum(a.available and not ok for a,ok in zip(st.alternatives,mask))
      mode,p=select(d['mode_probabilities'],mask,st.persona.persona_id,seed,sample)
      chosen.append(dict(d,mode=mode,mode_probabilities=p))
     writejl(cd/'decisions.jsonl',[dict(respondent_id=s.persona.persona_id,**d) for s,d in zip(states,chosen)])
     manifest,fallback=build_plans(cd,states,chosen,od,adapter,view,idx)
     started=time.perf_counter();print(name,'running',flush=True)
     for attempt in range(1,4):
      code,tail=run_matsim(cd,java)
      (cd/f'java_attempt_{attempt}.log').write_bytes((cd/'java_run.log').read_bytes())
      if code==0:break
      log=(cd/'java_run.log').read_text(encoding='utf-8',errors='replace')
      if (cd/'output').exists() or 'Read timed out' not in log:break
      print(name,'retrying external DTD timeout with identical inputs',flush=True)
     if code:write(cd/'failure.json',dict(code=code,tail=tail));raise RuntimeError(f'MATSim {name} failed')
     people,types=audit_events(cd/'output/ITERS/it.0/0.events.xml.zst',{r['person_id'] for r in manifest});by={r['person_id']:r for r in manifest}
     writejl(cd/'verified_people.jsonl',[dict(person_id=k,**v) for k,v in people.items()])
     outcome=dict(run=name,model=model,card=card,policy='sample' if sample else 'argmax',mask=maskname,seed=seed,n=321,
      original_pt_probability=float(np.mean([d['mode_probabilities'].get('pt',0) for d in decisions])),policy_pt_probability=float(np.mean([d['mode_probabilities'].get('pt',0) for d in chosen])),
      intention_counts=dict(collections.Counter(d['mode'] for d in chosen)),routed_counts=dict(collections.Counter(r['routed_outbound_mode'] for r in manifest)),executed_counts=dict(collections.Counter(v['outbound_mode'] for v in people.values())),
      outbound_completed=sum(v['outbound_arrived']==1 for v in people.values()),return_completed=sum(v['return_arrived']==1 for v in people.values()),stuck=sum(v['stuck'] for v in people.values()),unmatched_legs=sum(v['arrivals']!=v['departures'] for v in people.values()),
      routed_actual_mismatch=sum(v['outbound_mode']!=by[k]['routed_outbound_mode'] for k,v in people.items()),boardings=sum(v['pt_boardings'] for v in people.values()),departed_legs=sum(v['departures'] for v in people.values()),fallback=fallback,masked_alternatives=masked_count,exit_code=code,wall_seconds=time.perf_counter()-started,
      original_shift_used=True,network_sha256=sha(SUPPLY['network']),schedule_sha256=sha(SUPPLY['schedule']))
     outcome['pass']=all([outcome['outbound_completed']==321,outcome['return_completed']==321,outcome['stuck']==0,outcome['unmatched_legs']==0,outcome['routed_actual_mismatch']==0])
     write(cd/'verified.json',outcome);results.append(outcome);write(base/'summary.json',results)
     print(name,'PASS',outcome['pass'],'PT',outcome['executed_counts'].get('pt',0),flush=True)
     if not outcome['pass']:raise RuntimeError('Person-level execution verification failed')
 write(base/'summary.json',results)
 flat=[{k:v for k,v in r.items() if not isinstance(v,dict)}|{f'intended_{m}':r['intention_counts'].get(m,0) for m in MODES}|{f'executed_{m}':r['executed_counts'].get(m,0) for m in MODES}|r['fallback'] for r in results]
 # union columns, since fallback types may vary.
 keys=list(dict.fromkeys(k for r in flat for k in r));csvout(base/'ledger.csv',[{k:r.get(k,0) for k in keys} for r in flat])
 departure=[]
 for p in base.glob('*_preflight.json'):
  rows=read(p)
  departure.append(dict(run=p.stem,n=len(rows),shift_mae_from_requested=float(np.mean([abs(r['shift_min']) for r in rows])),shift_median=float(np.median([r['shift_min'] for r in rows])),pt_route_changed=sum(r['modes']['pt']['route_changed'] for r in rows),pt_requested_fallback=sum(r['modes']['pt']['requested']['routed_mode']!='pt' for r in rows),pt_adjusted_fallback=sum(r['modes']['pt']['adjusted']['routed_mode']!='pt' for r in rows),any_mode_route_changed=sum(any(m['route_changed'] for m in r['modes'].values()) for r in rows)))
 csvout(base/'departure_timing.csv',departure)
 print('ALL 48 EXECUTIONS COMPLETE',flush=True)
if __name__=='__main__':main()
