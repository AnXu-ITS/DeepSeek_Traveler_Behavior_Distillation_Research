"""Unified historical execution denominators and factual provenance inventory."""
from reviewer_closure import *
import xml.etree.ElementTree as ET
import zstandard

def parse_person_journeys(path,ids):
 people={p:dict(departures=0,arrivals=0,outbound=0,return_home=0,boardings=0,stuck=0,first_modes=set(),first_pt=0) for p in ids}
 with path.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as stream:
  parser=ET.iterparse(stream,events=('start','end'));_,root=next(parser);n=0
  for event,e in parser:
   if event!='end' or e.tag!='event':continue
   n+=1;p=people.get(e.get('person'));kind=e.get('type')
   if p is not None:
    if kind=='departure':
     p['departures']+=1
     if not p['outbound']:p['first_modes'].add(e.get('legMode'))
    elif kind=='arrival':p['arrivals']+=1
    elif kind=='PersonEntersPtVehicle':
     p['boardings']+=1
     if not p['outbound']:p['first_pt']+=1
    elif kind=='stuckAndAbort':p['stuck']+=1
    elif kind=='actstart':
     a=e.get('actType','')
     if a=='home':p['return_home']+=1
     elif a!='pt interaction' and not a.endswith('interaction'):p['outbound']+=1
   e.clear()
   if n%10000==0:root.clear()
 for p in people.values():
  ms=p['first_modes'];p['observed_first_mode']='pt' if p['first_pt'] else next((m for m in ['car','bike','walk'] if m in ms),'none');p['first_modes']=sorted(ms)
 return people,n

def main():
 dest=OUT/'historical';dest.mkdir(exist_ok=True);ledger=[]
 for city,directory,filename in [('Singapore','singapore_phase_c_s9','phase_c_records.json'),('Helsinki','e5_helsinki','e5_records.json')]:
  records=read(ROOT/f'outputs/{directory}/{filename}')
  for rec in records:
   card=rec['scenario'];cd=ROOT/f'outputs/{directory}/{card}';manifest=read(cd/'adapter_manifest.json');dec=manifest['decisions'];by={r['persona_id']:r for r in dec};assert len(by)==len(dec)
   row=dict(city=city,scenario=card,n_decision_people=len(dec),probability_pt=None,probability_note='Not saved in legacy adapter manifest; no reconstructed substitute',intended_pt=sum(r['student_mode']=='pt' for r in dec),routed_outbound_pt=sum(r['outbound_mode']=='pt' for r in dec),routed_return_pt=sum(r['return_mode']=='pt' for r in dec),routed_pt_retention=None,
    observed_outbound_pt=None,outbound_completed_people=None,return_completed_people=None,departed_people=None,departed_legs=None,arrived_legs=None,person_pt_boardings=None,stuck_people=None,
    reported_boardings=rec.get('metrics',{}).get('pt_boardings'),reported_stuck_people=rec.get('metrics',{}).get('stuck_persons'),verification='legacy manifest routing counts; actual journey counters unavailable without raw event pass',manifest_sha256=sha(cd/'adapter_manifest.json'))
   row['routed_pt_retention']=row['routed_outbound_pt']/row['intended_pt'] if row['intended_pt'] else None
   # All six production scenarios per city, no pilot mixing.
   verified=dest/f'{city}_{card}_events.json'
   if verified.exists():counts=read(verified)
   else:
    print(city,card,'streaming person-level events',flush=True)
    events=cd/'output/ITERS/it.0/0.events.xml.zst';people,n=parse_person_journeys(events,set(by))
    writejl(dest/f'{city}_{card}_people.jsonl',[dict(person_id=p,**r) for p,r in people.items()])
    counts=dict(observed_outbound_pt=sum(p['observed_first_mode']=='pt' for p in people.values()),outbound_completed_people=sum(p['outbound']>=1 for p in people.values()),return_completed_people=sum(p['return_home']>=1 for p in people.values()),departed_people=sum(p['departures']>0 for p in people.values()),departed_legs=sum(p['departures'] for p in people.values()),arrived_legs=sum(p['arrivals'] for p in people.values()),person_pt_boardings=sum(p['boardings'] for p in people.values()),stuck_people=sum(p['stuck']>0 for p in people.values()),event_count=n,event_sha256=sha(events))
    write(verified,counts)
   row.update(counts);row['verification']='Raw events streamed with expected person IDs; actual first-journey mode based on boarding/leg events, separately from arrival completion';ledger.append(row)
   print(city,card,'completed',row['outbound_completed_people'],row['return_completed_people'],flush=True)
 for c in read(NET/'execution_verification.json')['cards']:
  r=read(NET/c['card']/'run_summary.json');people=jl(NET/c['card']/'verified_person_events.jsonl');n=321
  ledger.append(dict(city='Shanghai',scenario=c['card'],n_decision_people=n,probability_pt=float(np.mean([x['probabilities'].get('pt',0) for x in jl(NET/c['card']/'predictions.jsonl')])),probability_note='Mean over all 321 executed people; includes four globally excluded human labels',intended_pt=r['student_counts'].get('pt',0),routed_outbound_pt=r['routed_outbound_counts'].get('pt',0),routed_return_pt=sum(d['routed_return_mode']=='pt' for d in read(NET/c['card']/'adapter_manifest.json')['decisions']),routed_pt_retention=r['routed_outbound_counts'].get('pt',0)/r['student_counts'].get('pt',1),observed_outbound_pt=c['executed_outbound_counts'].get('pt',0),outbound_completed_people=c['outbound_completed_people'],return_completed_people=c['return_completed_people'],departed_people=c['observed_departing_people'],departed_legs=sum(p['departures'] for p in people),arrived_legs=sum(p['arrivals'] for p in people),person_pt_boardings=c['pt_boardings'],stuck_people=c['stuck_people'],verification='Existing frozen person-level raw event verification, rechecked hash',manifest_sha256=sha(NET/c['card']/'adapter_manifest.json')))
 keys=list(dict.fromkeys(k for r in ledger for k in r));csvout(dest/'three_city_execution.csv',[{k:r.get(k) for k in keys} for r in ledger]);write(dest/'three_city_execution.json',ledger)
 print('Unified 22-scenario execution ledger complete',flush=True)
if __name__=='__main__':main()
