"""Verify all 140 expected model/policy/scenario keys and every initial person."""
import argparse,collections,time,statistics,re
import xml.etree.ElementTree as ET
import numpy as np
from helsinki_execution import ROOT,OUT,CARDS,SEEDS,MODES,checkpoint,read,jl,write,sha,uniform,StudentAdapter,UniversalTravelerState,torch

MODELS=['s9']+[f'{m}_seed{s}' for m in ['soft_kl','direction_magnitude','independent_neural'] for s in SEEDS]

def main(partial=False):
 expected=[(m,c,p,s) for m in MODELS for p in ['soft_argmax','soft_sampling','planner_sampling'] for s in ([0] if p=='soft_argmax' else SEEDS) for c in CARDS]
 assert len(expected)==140 and len(set(expected))==140
 initial={c:jl(OUT/'states'/f'{c}.jsonl') for c in CARDS};ids={c:[r['person_id'] for r in v] for c,v in initial.items()};assert ids[CARDS[0]]==ids[CARDS[1]] and len(ids[CARDS[0]])==1000
 historical_coords={};historical_pop=ROOT/'outputs/e5_helsinki/C0_baseline/population.xml'
 for _,e in ET.iterparse(historical_pop,events=('end',)):
  if e.tag!='person':continue
  if e.get('id') in set(ids[CARDS[0]]):
   acts=e.findall('plan/activity');historical_coords[e.get('id')]=[(a.get('type'),a.get('x'),a.get('y')) for a in acts]
  e.clear()
  if len(historical_coords)==1000:break
 assert len(historical_coords)==1000
 population_pair_checks={k:True for k in ['same_persona','same_trip','same_od','unchanged_nonpt_alternatives','only_perceived_pt_time_changes','only_perceived_context_delay_changes']}
 for b,d in zip(initial[CARDS[0]],initial[CARDS[1]]):
  sb,sd=b['state'],d['state'];population_pair_checks['same_persona'] &= sb['persona']==sd['persona'];population_pair_checks['same_trip'] &= sb['trip']==sd['trip'];population_pair_checks['same_od'] &= (b['origin_node'],b['destination_node'])==(d['origin_node'],d['destination_node'])
  population_pair_checks['only_perceived_context_delay_changes'] &= all(sd['context'][k]-v==15 if k=='transit_delay_min' else sd['context'][k]==v for k,v in sb['context'].items() if k!='context_id')
  for a,z in zip(sb['alternatives'],sd['alternatives']):
   if a['mode']!='pt':population_pair_checks['unchanged_nonpt_alternatives'] &= a==z
   else:population_pair_checks['only_perceived_pt_time_changes'] &= all(abs(z[k]-a[k]-15)<1e-6 if k in ['travel_time_min','reliability_delay_min'] else z[k]==a[k] for k in a)
 results=[];pending=[];errors=[];preds={};preflights={};sources={};raw_results={}
 reference_config=sha(OUT/'runs/s9__C0_baseline__soft_argmax__seed0/config.xml')
 for m,c,policy,seed in expected:
  name=f'{m}__{c}__{policy}__seed{seed}';folder=OUT/'runs'/name;rp=folder/'result.json'
  if not rp.exists():pending.append(name);continue
  r=read(rp)
  if r.get('exit_code')!=0:errors.append(dict(run=name,error='nonzero Java exit'));continue
  raw_results[m,c,policy,seed]=r
  people=jl(folder/'person_ledger.jsonl');checks={}
  checks['expected_run_key']=(r['model'],r['card'],r['policy'],r['sampling_seed'])==(m,c,policy,seed)
  checks['exact_initial_persons_in_order']=[x['person_id'] for x in people]==ids[c]
  checks['unique_initial_persons']=len({x['person_id'] for x in people})==1000
  xml_ids=[];xml_od=True
  for _,e in ET.iterparse(folder/'population.xml',events=('end',)):
   if e.tag!='person':continue
   pid=e.get('id');xml_ids.append(pid);acts=e.findall('plan/activity');xml_od &= [(a.get('type'),a.get('x'),a.get('y')) for a in acts]==historical_coords.get(pid);e.clear()
  checks['population_xml_initial_ids']=xml_ids==ids[c]
  checks['population_xml_historical_od_and_activity_types']=bool(xml_od)
  if (m,c) not in preds:preds[m,c]={x['person_id']:x for x in jl(OUT/'predictions'/m/f'{c}.jsonl')}
  if (m,c) not in preflights:preflights[m,c]={x['person_id']:x for x in jl(OUT/'preflight'/m/f'{c}.jsonl')}
  raw_equal=True;policy_equal=True;choice_equal=True;crn_equal=True
  preflight_equal=True;departure_equal=True;routing_equal=True
  for x in people:
   saved=preds[m,c][x['person_id']];raw=np.array([x['original_probability'][mode] for mode in MODES]);q=np.array([x['filtered_probability'][mode] for mode in MODES])
   pr=preflights[m,c][x['person_id']]
   preflight_equal &= x['planner_mask']==pr['planner_mask']
   departure_equal &= abs(x['actual_departure_min']-pr['actual_departure_min'])<1e-9 and abs(x['departure_shift_min']-saved['departure_time_shift_min'])<1e-9 and abs(x['actual_departure_min']-pr['requested_departure_min']-x['departure_shift_min'])<1e-9
   routing_equal &= (x['routed_mode']==x['assigned_mode'])==pr['planner_mask'].get(x['assigned_mode'],False) and x['fallback']==(x['routed_mode']!=x['assigned_mode'])
   raw_equal &= all(abs(raw[i]-saved['mode_probabilities'].get(mode,0))<1e-9 for i,mode in enumerate(MODES))
   expected_q=raw.copy()
   if policy=='planner_sampling':expected_q*=np.array([x['planner_mask'].get(mode,False) for mode in MODES])
   expected_q/=expected_q.sum();policy_equal &= bool(np.allclose(q,expected_q,rtol=0,atol=1e-12) and abs(q.sum()-1)<1e-12)
   u=uniform(x['person_id'],seed);crn_equal &= u==x['uniform']
   selected=MODES[int(q.argmax())] if policy=='soft_argmax' else MODES[min(3,int(np.searchsorted(np.cumsum(q),u)))]
   choice_equal &= selected==x['assigned_mode']
  checks.update(raw_probability_matches_model=bool(raw_equal),probability_transformation_correct=bool(policy_equal),assignment_matches_policy=bool(choice_equal),common_random_numbers_exact=bool(crn_equal))
  checks.update(mask_matches_actual_departure_preflight=bool(preflight_equal),shift_applied_before_actual_departure_routing=bool(departure_equal),routed_or_fallback_matches_preflight_feasibility=bool(routing_equal),identical_physical_simulation_config=sha(folder/'config.xml')==reference_config)
  counts=dict(n_initial=len(people),n_departed=sum(x['events']['departures']>0 for x in people),assigned_pt_people=sum(x['assigned_mode']=='pt' for x in people),routed_pt_people=sum(x['routed_mode']=='pt' for x in people),actual_outbound_pt_people=sum(x['events']['outbound_pt_boardings']>0 for x in people),outbound_completed=sum(x['events']['outbound_arrived'] for x in people),return_completed=sum(x['events']['return_arrived'] for x in people),stuck_people=sum(x['events']['stuck']>0 for x in people),unfinished_legs=sum(x['events']['unfinished_leg_count'] for x in people),pt_boardings=sum(x['events']['pt_boardings'] for x in people),pt_retained_people=sum(x['assigned_mode']=='pt' and x['events']['outbound_pt_boardings']>0 for x in people),fallback_people=sum(x['assigned_mode']!=x['routed_mode'] for x in people))
  checks['all_person_counts_match_summary']=all(r[k]==v for k,v in counts.items())
  checks['completed_leg_count_matches_summary']=sum(x['events']['completed_legs'] for x in people)==r['n_completed_legs']
  checks['unmatched_legs_consistent']=all(x['events']['unfinished_leg_count']==max(0,x['events']['departures']-x['events']['arrivals']) for x in people)
  checks['pt_boarding_requires_routed_pt']=all(x['routed_mode']=='pt' for x in people if x['events']['outbound_pt_boardings']>0)
  checks['outbound_boardings_bounded_by_all_boardings']=all(x['events']['outbound_pt_boardings']<=x['events']['pt_boardings'] for x in people)
  checks['mean_probability_matches_summary']=all(abs(np.mean([x[field]['pt'] for x in people])-r[key])<1e-12 for field,key in [('original_probability','original_pt_probability'),('filtered_probability','filtered_pt_probability')])
  checks['pt_retention_denominator_correct']=abs(r['pt_retention_rate']-counts['pt_retained_people']/max(1,counts['assigned_pt_people']))<1e-12
  checks['unique_pt_not_boarding_counts']=counts['actual_outbound_pt_people']<=counts['pt_boardings'] and counts['actual_outbound_pt_people']<=1000
  checks['executed_modes_match_summary']=dict(collections.Counter(x['events']['outbound_mode'] for x in people))==r['executed_counts']
  checks['population_hash_matches']=sha(folder/'population.xml')==r['population_sha256']
  checks['raw_events_hash_matches']=sha(folder/'output/ITERS/it.0/0.events.xml.zst')==r['event_sha256']
  checks['checkpoint_hash_matches']=sha(ROOT/r['model_path'])==r['model_sha256'];sources[r['model_path']]=r['model_sha256']
  entry=dict(run=name,model=m,training_seed=None if m=='s9' else int(m.rsplit('seed',1)[1]),sampling_seed=None if policy=='soft_argmax' else seed,policy=policy,card=c,checks=checks,counts=counts,person_ledger_sha256=sha(folder/'person_ledger.jsonl'),population_sha256=r['population_sha256'],raw_events_sha256=r['event_sha256'],java_and_postprocess_seconds=r['runtime_s']+max(0,rp.stat().st_mtime-(folder/'java_run.log').stat().st_mtime),pass_all=all(checks.values()))
  if not entry['pass_all']:errors.append(dict(run=name,failed_checks=[k for k,v in checks.items() if not v]))
  results.append(entry)
 protocol=read(OUT/'protocol.json');original_sources={p:sha(ROOT/p)==h for p,h in protocol['source_hashes'].items() if p!='scripts/revision_20260921/helsinki_execution.py'}
 combo=ROOT/'outputs/reviewer_closure_20260920/baseline/model.npz';sources[str(combo.relative_to(ROOT))]=sha(combo)
 original_sources['fixed_mnl_s_choice_model']=sources[str(combo.relative_to(ROOT))]=='a0871249e33b86a11b6204a980026e583705944a8d79e70f18d7ef47c7da8e31'
 fixed=read(OUT/'fixed_population_audit.json');original_sources['fixed_original_population_intact']=sha(historical_pop)==fixed['historical_population_sha256'];original_sources['fixed_baseline_states_intact']=sha(OUT/'states/C0_baseline.jsonl')==fixed['revision_baseline_states_sha256']
 report=dict(expected_run_count=140,completed_run_count=len(results),pending_count=len(pending),all_expected_keys=[dict(model=m,card=c,policy=p,training_seed=None if m=='s9' else int(m.rsplit('seed',1)[1]),sampling_seed=None if p=='soft_argmax' else s) for m,c,p,s in expected],pending_runs=pending,errors=errors,all_completed_run_checks_pass=not errors,complete=len(results)==140 and not pending and not errors and all(original_sources.values()) and all(population_pair_checks.values()),initial_persons=1000,population_seed=2026,population_pair_checks=population_pair_checks,historical_population_sha256=sha(historical_pop),training_seeds=[42,2026,7],sampling_seeds=[42,2026,7],seed_semantics='Training seed identifies a separately trained model; sampling seed identifies a CRN allocation replicate. Population seed2026 and MATSim seed4711 are fixed. S9 has one frozen artifact and no new training-seed replication.',aggregation='Average sampling seeds within each trained model, then compare three training models; never treat the9 combinations as9 independent training replicates.',source_hashes_unchanged=original_sources,model_source_sha256=sources,script_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts/revision_20260921').glob('*execution*.py')},failed_heap4g_attempt='failed_attempts/s9__C0_baseline__soft_argmax__seed0_heap4g: excluded; rerun with historical6GB heap',per_run=results)
 runtime_paths=[ROOT/'tools/java/RunMatsimPreloaded.java',ROOT/'tools/matsim-2026.0-release/matsim-2026.0/matsim-2026.0.jar']
 report['runtime_artifact_sha256']={str(p.relative_to(ROOT)):sha(p) for p in runtime_paths}
 report['state_sha256']={c:sha(OUT/'states'/f'{c}.jsonl') for c in CARDS}
 pairs=read(OUT/'response_summary.json');pair_keys={(p['model'],p['policy'],p['sampling_seed']) for p in pairs}
 summary_checks={'all_70_unique_pairs':len(pairs)==len(pair_keys)==70,'response_values_match_verified_runs':True,'all_12_gap_groups_complete':False,'gap_absolute_aggregation_correct':True}
 for p in pairs:
  b=raw_results.get((p['model'],CARDS[0],p['policy'],p['sampling_seed']));d=raw_results.get((p['model'],CARDS[1],p['policy'],p['sampling_seed']))
  if b is None or d is None:summary_checks['response_values_match_verified_runs']=False;continue
  pred=100*(d['original_pt_probability']-b['original_pt_probability']);exe=(d['actual_outbound_pt_people']-b['actual_outbound_pt_people'])/10
  summary_checks['response_values_match_verified_runs'] &= p['n_initial']==1000 and abs(pred-p['predicted_response_pp'])<1e-10 and abs(exe-p['executed_response_pp'])<1e-10 and abs(exe-pred-p['execution_minus_prediction_pp'])<1e-10
 gaps=read(OUT/'response_gap_cancellation_summary.json')['summary']
 summary_checks['all_12_gap_groups_complete']=len(gaps)==12 and all(g['complete'] for g in gaps)
 for g in gaps:
  rs=[p for p in pairs if re.sub(r'_seed\d+$','',p['model'])==g['model_family'] and p['policy']==g['policy']];ms={p['model'] for p in rs}
  av=[statistics.mean(p['execution_minus_prediction_pp'] for p in rs if p['model']==m) for m in ms]
  summary_checks['gap_absolute_aggregation_correct'] &= len(rs)==g['n_runs'] and len(ms)==g['n_models'] and abs(abs(statistics.mean(av))-g['abs_family_signed_mean_pp'])<1e-10 and abs(statistics.mean(abs(x) for x in av)-g['mean_abs_model_gap_after_sampling_mean_pp'])<1e-10 and abs(statistics.mean(abs(p['execution_minus_prediction_pp']) for p in rs)-g['mean_abs_per_run_gap_pp'])<1e-10
 report['derived_summary_checks']=summary_checks;report['complete'] &= all(summary_checks.values())
 if not partial:
  # One fresh deterministic inference per checkpoint/scenario proves that the
  # cached probabilities and timing inputs correspond to the hashed artifacts.
  torch.set_num_threads(4)
  states={c:[UniversalTravelerState.model_validate(r['state']) for r in initial[c]] for c in CARDS}
  prediction_checks=[]
  for m in MODELS:
   if m.startswith('independent_neural_seed'):
    from departure_baselines import CombinationAdapter
    adapter=CombinationAdapter(ROOT/'outputs/revision_20260921/baselines/train'/m/'best_departure.pt')
   else:adapter=StudentAdapter(checkpoint(m),device='cpu')
   for c in CARDS:
    fresh=adapter.predict(states[c]);saved=jl(OUT/'predictions'/m/f'{c}.jsonl')
    max_p=max(abs(a['mode_probabilities'].get(mode,0)-b['mode_probabilities'].get(mode,0)) for a,b in zip(fresh,saved) for mode in MODES)
    max_shift=max(abs(a['departure_time_shift_min']-b['departure_time_shift_min']) for a,b in zip(fresh,saved))
    prediction_checks.append(dict(model=m,card=c,n_fresh=len(fresh),n_cached=len(saved),max_abs_probability_difference=max_p,max_abs_shift_difference_min=max_shift,pass_all=len(fresh)==len(saved)==1000 and max_p<1e-6 and max_shift<1e-5))
  report['fresh_checkpoint_inference_checks']=prediction_checks
  report['complete'] &= all(r['pass_all'] for r in prediction_checks)
 write(OUT/'verification.json',report)
 print(f'Execution verification: {len(results)}/140; pending={len(pending)}; errors={len(errors)}; complete={report["complete"]}',flush=True)
 if not partial and not report['complete']:raise RuntimeError('Execution verification incomplete or inconsistent')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args();main(a.partial)
