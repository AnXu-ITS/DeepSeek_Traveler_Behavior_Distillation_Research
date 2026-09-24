"""Fixed historical Helsinki population: probabilities -> policy -> route -> events.

All additions are under revision_20260921; frozen populations/survey data are read-only.
PT delay is a perceived input perturbation; the physical network/schedule is unchanged.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, os, subprocess, sys, threading, time
import xml.etree.ElementTree as ET
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
import zstandard

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/shanghai')]
from reference_pipeline.student_adapter import StudentAdapter
from reference_pipeline.matsim_adapter import run_matsim
from reference_pipeline.route_cache import RouteCache
from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives
from traveler_distillation.config import load_yaml
from traveler_distillation.generators import PersonaGenerator,TripGenerator
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.matsim.adapter import _activity_link,_ACTIVITY_DURATIONS_MIN,_LEG_MIN,_to_hms
from singapore.run_phase_c import make_shared_idx,make_context
from sp_survey_matsim import prepare_java

OUT=ROOT/'outputs/revision_20260921/execution'
MODES=['car','pt','bike','walk']
CARDS=['C0_baseline','C3_transit_delay']
SEEDS=[42,2026,7]
SUPPLY={k:ROOT/'data/helsinki/transit'/v for k,v in dict(network='network_with_transit.xml',schedule='transitSchedule.xml',vehicles='transitVehicles.xml',stops='prep_stops.jsonl',snapping='stop_snapping_report.json',trips_by_stop='trips_by_stop.json',activity_nodes='activity_nodes.json').items()}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def jl(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}.{threading.get_ident()}');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def writejl(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def csvout(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r));Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def uniform(pid,seed):return (int(hashlib.sha256(f'helsinki-revision-20260921:{seed}:{pid}'.encode()).hexdigest()[:13],16)+.5)/16**13

def events(path,expected):
 """Initial-person denominator; first destination activity closes outbound trip."""
 people={pid:dict(departures=0,arrivals=0,stuck=0,outbound_arrived=False,return_arrived=False,
  outbound_pt_boardings=0,pt_boardings=0,outbound_modes=set(),outbound_start=None,outbound_end=None,
  return_start=None,return_end=None,completed_leg_seconds=0.,completed_legs=0,pending=[]) for pid in expected}
 types=collections.Counter()
 assert all(pid.startswith('P') for pid in expected)
 with Path(path).open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as f:
  # MATSim's actual writer emits one complete event per line. Parse XML only
  # for population-person events; still count every event type exactly.
  # This avoids building millions of transit-vehicle/link event XML objects.
  for line in io.BufferedReader(f):
   if b'<event ' not in line:continue
   assert line.lstrip().startswith(b'<event ') and line.rstrip().endswith(b'/>'),'Unexpected event serialization'
   a=line.index(b' type="')+7;b=line.index(b'"',a);typ=line[a:b].decode();types[typ]+=1
   if b' person="P' not in line:continue
   e=ET.fromstring(line);pid=e.get('person')
   if pid in people:
    p=people[pid];t=float(e.get('time'))
    if typ=='departure':
     p['departures']+=1;p['pending'].append(t)
     if not p['outbound_arrived']:
      p['outbound_modes'].add(e.get('legMode'))
      if p['outbound_start'] is None:p['outbound_start']=t
     elif p['return_start'] is None:p['return_start']=t
    elif typ=='arrival':
     p['arrivals']+=1
     if p['pending']:p['completed_leg_seconds']+=t-p['pending'].pop();p['completed_legs']+=1
    elif typ=='actstart':
     act=e.get('actType')
     if act!='home' and not act.endswith('interaction') and not p['outbound_arrived']:
      p['outbound_arrived']=True;p['outbound_end']=t
     elif act=='home' and p['outbound_arrived']:
      p['return_arrived']=True;p['return_end']=t
    elif typ=='PersonEntersPtVehicle':
     p['pt_boardings']+=1
     if not p['outbound_arrived']:p['outbound_pt_boardings']+=1
    elif typ=='stuckAndAbort':p['stuck']+=1
   e.clear()
 for p in people.values():
  modes=p['outbound_modes'];p['outbound_mode']='pt' if p['outbound_pt_boardings'] else next((m for m in ['car','bike','walk'] if m in modes),'none')
  p['outbound_modes']=sorted(modes);p['unfinished_leg_count']=len(p.pop('pending'))
  p['outbound_trip_min']=(p['outbound_end']-p['outbound_start'])/60 if p['outbound_end'] is not None and p['outbound_start'] is not None else None
  p['return_trip_min']=(p['return_end']-p['return_start'])/60 if p['return_end'] is not None and p['return_start'] is not None else None
  # Administrative upper-bound horizon, never impute failure as zero duration.
  p['outbound_time_to_completion_or_horizon_min']=(p['outbound_end']-p['outbound_start'])/60 if p['outbound_trip_min'] is not None else ((108000-p['outbound_start'])/60 if p['outbound_start'] is not None else None)
 return people,dict(types)

def summarize_people(people):
 vals=list(people.values());complete=[p['outbound_trip_min'] for p in vals if p['outbound_trip_min'] is not None]
 return dict(n_initial=len(vals),n_departed=sum(p['departures']>0 for p in vals),outbound_completed=sum(p['outbound_arrived'] for p in vals),return_completed=sum(p['return_arrived'] for p in vals),stuck_people=sum(p['stuck']>0 for p in vals),unfinished_legs=sum(p['unfinished_leg_count'] for p in vals),actual_outbound_pt_people=sum(p['outbound_pt_boardings']>0 for p in vals),pt_boardings=sum(p['pt_boardings'] for p in vals),executed_counts=dict(collections.Counter(p['outbound_mode'] for p in vals)),completed_outbound_trip_mean_min=float(np.mean(complete)) if complete else None,completed_outbound_trip_median_min=float(np.median(complete)) if complete else None,completed_leg_mean_min=sum(p['completed_leg_seconds'] for p in vals)/max(1,sum(p['completed_legs'] for p in vals))/60,n_completed_legs=sum(p['completed_legs'] for p in vals),mean_time_to_completion_or_horizon_min=float(np.mean([p['outbound_time_to_completion_or_horizon_min'] for p in vals if p['outbound_time_to_completion_or_horizon_min'] is not None])))

def checkpoint(name):
 if name=='s9':return ROOT/'releases/s9_supply_aware_v2/checkpoint/model.pt'
 return ROOT/'outputs/matched_response_v1/train'/name/'best.pt'

class Experiment:
 def __init__(self,n):
  self.n=n;OUT.mkdir(parents=True,exist_ok=True);self.s9=StudentAdapter(checkpoint('s9'),device='cpu');self.adapter=self.s9._adapter
  print('Loading fixed Helsinki supply',flush=True);self.idx=make_shared_idx(SUPPLY)
  self.cache=RouteCache(ROOT/'outputs/reference_pipeline/validation/helsinki_c0/cache/route_cache.pkl',SUPPLY,reuse=True,rebuild=False)
  assert self.cache.loaded,'Historical cache must match all current supply hashes and routing constants'
  original_tt=self.idx.mode_travel_time
  self.idx.mode_travel_time=lambda m,o,d:self.cache.mode_travel_time(m,o,d,lambda:original_tt(m,o,d))
  # Reuse identical graphs, attaching true link IDs needed by population writer.
  for m,g in self.idx.tt_graphs.items():
   for u,v,d in g.edges(data=True):d['id']=self.idx.g.edges[u,v].get('id',f'{u}_{v}')
  self.adapter._access_max_m=700.;original_shortest=self.adapter._shortest
  graph_modes={id(g):m for m,g in self.idx.tt_graphs.items()}
  self.adapter._shortest=lru_cache(maxsize=300000)(lambda g,o,d:self.cache.shortest_path(graph_modes[id(g)],o,d,lambda:original_shortest(g,o,d)))
  self.route=lru_cache(maxsize=100000)(self._route)
  self.java=prepare_java();self.java.java_xmx='6g'
  self.hist={c:read(ROOT/'outputs/e5_helsinki'/c/'adapter_manifest.json')['decisions'][:n] for c in CARDS}
  self.states={};self.records={}

 def prepare(self):
  cfg=load_yaml(ROOT/'configs/generation_v0_1.yaml')
  # Generate historical full stream before taking fixed prefix: no outcome selection.
  persons=PersonaGenerator(seed=2026,config=cfg).generate(10000)[:self.n]
  trips=TripGenerator(seed=2026,config=cfg).generate(10000)[:self.n]
  for card in CARDS:
   path=OUT/'states'/f'{card}.jsonl'
   if path.exists():rows=jl(path)
   else:
    rows=[];ctx=make_context(card)
    for i,(p,t,h) in enumerate(zip(persons,trips,self.hist[card])):
     assert (p.persona_id,t.trip_id)==(h['persona_id'],h['trip_id'])
     alts=build_real_alternatives(p,t,ctx,self.idx,h['home_node'],h['dest_node'],h['accessibility'])
     for a in alts:
      if a.mode=='pt' and ctx.transit_delay_min:a.travel_time_min=round(a.travel_time_min+ctx.transit_delay_min,3);a.reliability_delay_min=round(a.reliability_delay_min+ctx.transit_delay_min,3)
     st=UniversalTravelerState(persona=p,trip=t,context=ctx,alternatives=alts)
     rows.append(dict(person_id=p.persona_id,origin_node=h['home_node'],destination_node=h['dest_node'],state=st.model_dump(mode='json')))
     if (i+1)%100==0:print(card,'states',i+1,flush=True)
    writejl(path,rows)
   assert len(rows)==self.n
   self.records[card]=rows;self.states[card]=[UniversalTravelerState.model_validate(r['state']) for r in rows]
  write(OUT/'protocol.json',dict(n=self.n,population_seed=2026,population_selection='first 1000 from unchanged historical 10000 stream, matched person/trip/OD to historical manifest',sampling_seeds=SEEDS,common_random_numbers='SHA256 seed+person; shared across scenario/model/policy',perturbation='perceived PT delay +15 min in context and legacy alternative travel-time/reliability fields; original unchanged physical schedule',primary_denominator='same initial N persons; unique outbound PT people',retention_denominator='assigned outbound PT persons',simulation='MATSim 2026.0, iteration 0, fixed full Helsinki network/schedule, capacity factors .3/.3, original network link-freespeed execution',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [checkpoint('s9'),Path(__file__),ROOT/'src/traveler_distillation/matsim/adapter.py',*SUPPLY.values()]}))

 def _route(self,mode,src,dst,dep,start):
  return self.adapter._build_legs(mode,src,dst,dep,start,self.idx.trips_by_stop,self.idx.stop_node,self.idx.stops,self.idx.tt_graphs,300.,collections.Counter(),self.idx.trip_seq)

 def prediction(self,model,card):
  p=OUT/'predictions'/model/f'{card}.jsonl'
  if p.exists():return jl(p)
  if model=='mnl_s':
   sys.path.insert(0,str(ROOT/'cvpr_workspace/analysis/statistics'))
   from reviewer_closure import LinearUtility
   dec=LinearUtility().load().decisions(self.states[card])
  elif model.startswith('bounded_linear_seed') or model.startswith('independent_neural_seed'):
   from departure_baselines import CombinationAdapter
   dec=CombinationAdapter(ROOT/'outputs/revision_20260921/baselines/train'/model/'best_departure.pt').predict(self.states[card])
  else:dec=StudentAdapter(checkpoint(model),device='cpu').predict(self.states[card])
  rows=[dict(person_id=s.persona.persona_id,**d) for s,d in zip(self.states[card],dec)];writejl(p,rows)
  if model=='s9':
   mismatch=sum(r['mode']!=h['student_mode'] for r,h in zip(rows,self.hist[card]));assert mismatch==0,(card,mismatch)
   assert max(abs(r['departure_time_shift_min']-h['departure_shift_min']) for r,h in zip(rows,self.hist[card]))<.011
  return rows

 def preflight(self,model,card,dec):
  path=OUT/'preflight'/model/f'{card}.jsonl'
  if path.exists():return jl(path)
  rows=[]
  for i,(st,r,d) in enumerate(zip(self.states[card],self.records[card],dec)):
   requested=float(st.trip.desired_departure_min);adj=requested+d['departure_time_shift_min'];o=r['origin_node'];dest=r['destination_node'];masks={};ptdiag={}
   for a in st.alternatives:
    if not a.available:masks[a.mode]=False;continue
    start=_activity_link(self.idx.g,o,a.mode if a.mode!='pt' else 'walk')
    after=self.route(a.mode,o,dest,adj*60,start);masks[a.mode]=after[2]==a.mode
    if a.mode=='pt':
     before=self.route(a.mode,o,dest,requested*60,start)
     ptdiag=dict(requested_route=before[1],adjusted_route=after[1],requested_feasible=before[2]=='pt',adjusted_feasible=after[2]=='pt',route_changed=before[0]!=after[0])
   rows.append(dict(person_id=r['person_id'],requested_departure_min=requested,actual_departure_min=adj,planner_mask=masks,pt_timing_audit=ptdiag))
   if (i+1)%100==0:print(model,card,'preflight',i+1,flush=True)
  writejl(path,rows);return rows

 def run(self,model,card,policy,seed,dec,pre):
  name=f'{model}__{card}__{policy}__seed{seed}';out=OUT/'runs'/name;resultpath=out/'result.json'
  if resultpath.exists() and read(resultpath).get('exit_code')==0:return read(resultpath)
  out.mkdir(parents=True,exist_ok=True)
  if (out/'output').exists():raise RuntimeError(f'Preserve unfinished run and use a distinct output: {out}')
  pop=ET.Element('population');ledger=[]
  for st,r,d,pr in zip(self.states[card],self.records[card],dec,pre):
   raw=np.array([d['mode_probabilities'].get(m,0.) for m in MODES]);p=raw.copy()
   if policy=='planner_sampling':p*=np.array([pr['planner_mask'].get(m,False) for m in MODES])
   if p.sum()<=0:raise RuntimeError('No positive feasible alternative')
   p/=p.sum();u=uniform(r['person_id'],seed);mode=MODES[int(p.argmax())] if policy=='soft_argmax' else MODES[min(3,int(np.searchsorted(np.cumsum(p),u)))]
   dep=pr['actual_departure_min'];o=r['origin_node'];dst=r['destination_node'];start=_activity_link(self.idx.g,o,mode if mode!='pt' else 'walk')
   legs,info,routed,end=self.route(mode,o,dst,dep*60,start)
   retdep=dep+_LEG_MIN+_ACTIVITY_DURATIONS_MIN.get(st.trip.destination_type,120)
   ret,retinfo,retmode,finish=self.route(routed,dst,o,retdep*60,end)
   person=ET.SubElement(pop,'person',id=r['person_id']);plan=ET.SubElement(person,'plan',selected='yes')
   def act(kind,node,link,until=None):
    x,y=self.idx.xy[node];kw=dict(type=kind,link=link,x=f'{x:.2f}',y=f'{y:.2f}',z='0.0')
    if until is not None:kw['end_time']=_to_hms(until)
    ET.SubElement(plan,'activity',**kw)
   def add(seq):
    for m,rt,txt,extra in seq:
     el=ET.SubElement(plan,'leg',mode=m)
     if txt:ET.SubElement(el,'route',type=rt,**extra).text=txt
   act('home',o,start,dep);add(legs);act(st.trip.destination_type,dst,end,retdep);add(ret);act('home',o,finish)
   ledger.append(dict(person_id=r['person_id'],model=model,card=card,policy=policy,sampling_seed=seed,uniform=u,original_probability={m:float(raw[j]) for j,m in enumerate(MODES)},filtered_probability={m:float(p[j]) for j,m in enumerate(MODES)},assigned_mode=mode,routed_mode=routed,routed_return_mode=retmode,outbound_route=info,return_route=retinfo,fallback=bool(mode!=routed),departure_shift_min=d['departure_time_shift_min'],actual_departure_min=dep,planner_mask=pr['planner_mask']))
  ET.ElementTree(pop).write(out/'population.xml',encoding='utf-8',xml_declaration=True)
  self.adapter._insert_doctype(out/'population.xml','<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">')
  self.adapter._write_real_config(out/'config.xml',SUPPLY['network'],SUPPLY['schedule'],SUPPLY['vehicles'],.3,.3)
  cfg=(out/'config.xml').read_text(encoding='utf-8').replace('EPSG:32648','EPSG:32635').replace('deleteDirectoryIfExists','failIfDirectoryExists');(out/'config.xml').write_text(cfg,encoding='utf-8')
  writejl(out/'person_ledger.jsonl',ledger);print(name,'MATSim start',flush=True);started=time.time()
  code,tail=run_matsim(out,self.java)
  source=(ROOT/'outputs/reviewer_closure_20260920/baseline/model.npz') if model=='mnl_s' else ((ROOT/'outputs/revision_20260921/baselines/train'/model/'best_departure.pt') if model.startswith(('bounded_linear_seed','independent_neural_seed')) else checkpoint(model))
  result=dict(run=name,model=model,card=card,policy=policy,sampling_seed=seed,n_initial=self.n,exit_code=code,runtime_s=time.time()-started,model_path=str(source.relative_to(ROOT)),model_sha256=sha(source))
  if code:result['log_tail']=tail;write(resultpath,result);raise RuntimeError(f'MATSim failed {name}: {tail}')
  people,types=events(out/'output/ITERS/it.0/0.events.xml.zst',{r['person_id'] for r in ledger})
  for r in ledger:r['events']=people[r['person_id']]
  writejl(out/'person_ledger.jsonl',ledger)
  intended=[r for r in ledger if r['assigned_mode']=='pt']
  result.update(summarize_people(people));result.update(dict(original_pt_probability=float(np.mean([r['original_probability']['pt'] for r in ledger])),filtered_pt_probability=float(np.mean([r['filtered_probability']['pt'] for r in ledger])),assigned_pt_people=len(intended),routed_pt_people=sum(r['routed_mode']=='pt' for r in ledger),fallback_people=sum(r['fallback'] for r in ledger),pt_retained_people=sum(r['events']['outbound_pt_boardings']>0 for r in intended),pt_retention_rate=sum(r['events']['outbound_pt_boardings']>0 for r in intended)/max(1,len(intended)),event_types=types,population_sha256=sha(out/'population.xml'),event_sha256=sha(out/'output/ITERS/it.0/0.events.xml.zst')))
  write(resultpath,result);print(name,'done PT',result['actual_outbound_pt_people'],'arrived',result['outbound_completed'],flush=True);return result

def aggregate():
 results=[read(p) for p in (OUT/'runs').glob('*/result.json') if read(p).get('exit_code')==0]
 write(OUT/'run_summary.json',results)
 flat=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results];csvout(OUT/'run_summary.csv',flat)
 lookup={(r['model'],r['card'],r['policy'],r['sampling_seed']):r for r in results};pairs=[]
 for (m,c,p,s),b in lookup.items():
  if c!='C0_baseline':continue
  d=lookup.get((m,'C3_transit_delay',p,s))
  if d is None:continue
  n=b['n_initial'];r=dict(model=m,policy=p,sampling_seed=s,n_initial=n)
  for field,outkey in [('original_pt_probability','predicted_response_pp'),('filtered_pt_probability','filtered_response_pp')]:r[outkey]=100*(d[field]-b[field])
  for field,outkey in [('assigned_pt_people','assigned_response_pp'),('routed_pt_people','routed_response_pp'),('actual_outbound_pt_people','executed_response_pp')]:r[outkey]=100*(d[field]-b[field])/n
  r['execution_minus_prediction_pp']=r['executed_response_pp']-r['predicted_response_pp'];r['baseline_completion']=b['outbound_completed']/n;r['delay_completion']=d['outbound_completed']/n;pairs.append(r)
 write(OUT/'response_summary.json',pairs);csvout(OUT/'response_summary.csv',pairs)
 timing=[]
 for p in (OUT/'preflight').glob('*/*.jsonl'):
  rows=jl(p);timing.append(dict(model=p.parent.name,card=p.stem,n=len(rows),pt_itinerary_changed=sum(r['pt_timing_audit'].get('route_changed',False) for r in rows),requested_pt_feasible=sum(r['pt_timing_audit'].get('requested_feasible',False) for r in rows),adjusted_pt_feasible=sum(r['pt_timing_audit'].get('adjusted_feasible',False) for r in rows),mean_abs_shift_min=float(np.mean([abs(r['actual_departure_min']-r['requested_departure_min']) for r in rows]))))
 csvout(OUT/'departure_timing_audit.csv',timing)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['s9']);ap.add_argument('--policies',nargs='+',default=['soft_argmax','soft_sampling','planner_sampling']);ap.add_argument('--n',type=int,default=1000);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--aggregate-only',action='store_true');a=ap.parse_args()
 if a.aggregate_only:aggregate();return
 torch.set_num_threads(4);ex=Experiment(a.n);ex.prepare()
 if a.prepare_only:return
 # Prepare subsequent model states/routes while at most two 6 GB JVMs execute
 # previously prepared plans. Routing reads a shared immutable supply; each
 # task owns its population, configuration, ledger and event output directory.
 aggregate_lock=threading.Lock();futures=[]
 def finished(f):
  if not f.cancelled() and f.exception() is None:
   with aggregate_lock:aggregate()
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for model in a.models:
   data={card:ex.prediction(model,card) for card in CARDS}
   pre={card:ex.preflight(model,card,data[card]) for card in CARDS}
   for f in futures:
    if f.done():f.result()
   for policy in a.policies:
    for seed in ([0] if policy=='soft_argmax' else SEEDS):
     for card in CARDS:
      f=pool.submit(ex.run,model,card,policy,seed,data[card],pre[card]);f.add_done_callback(finished);futures.append(f)
  for f in as_completed(futures):f.result()
 aggregate()

if __name__=='__main__':main()
