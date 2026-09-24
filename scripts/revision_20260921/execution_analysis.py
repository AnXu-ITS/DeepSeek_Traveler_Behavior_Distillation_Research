"""Paired initial-person uncertainty and audit tables from actual MATSim events."""
import collections,re
import numpy as np
from helsinki_execution import OUT,ROOT,read,jl,write,csvout,aggregate,MODES

def interval(x,seed=917):
 x=np.asarray(x,dtype=float);rng=np.random.default_rng(seed);means=[]
 # Same 4000 RNG draws in bounded blocks, avoiding >600 MB temporary arrays
 # for the 10,000-person Singapore audit while two JVMs are running.
 for start in range(0,4000,200):
  ix=rng.integers(len(x),size=(min(200,4000-start),len(x)))
  means.extend(x[ix].mean(axis=1).tolist())
 return np.quantile(means,[.025,.975]).tolist()

def main():
 aggregate();pairs=[]
 for r in read(OUT/'response_summary.json'):
  m,p,s=r['model'],r['policy'],r['sampling_seed'];runs=OUT/'runs'
  b=jl(runs/f'{m}__C0_baseline__{p}__seed{s}'/'person_ledger.jsonl');d=jl(runs/f'{m}__C3_transit_delay__{p}__seed{s}'/'person_ledger.jsonl')
  assert [x['person_id'] for x in b]==[x['person_id'] for x in d]
  pred=np.array([y['original_probability']['pt']-x['original_probability']['pt'] for x,y in zip(b,d)])*100
  exe=np.array([float(y['events']['outbound_pt_boardings']>0)-float(x['events']['outbound_pt_boardings']>0) for x,y in zip(b,d)])*100
  row=dict(r)
  for label,v in [('predicted',pred),('executed',exe),('execution_minus_prediction',exe-pred)]:
   low,high=interval(v);row[f'{label}_ci95_low_pp']=low;row[f'{label}_ci95_high_pp']=high
  row['bootstrap']='4000 paired initial-person resamples, seed917, conditional on model and sampling seed'
  pairs.append(row)
 write(OUT/'response_intervals.json',pairs);csvout(OUT/'response_intervals.csv',pairs)
 groups=collections.defaultdict(list)
 for r in pairs:groups[(re.sub(r'_seed\d+$','',r['model']),r['policy'])].append(r)
 family=[]
 for (model,policy),rs in groups.items():
  models=sorted({r['model'] for r in rs});row=dict(model_family=model,policy=policy,n_training_models=len(models),n_paired_runs=len(rs))
  for key in ['predicted_response_pp','filtered_response_pp','assigned_response_pp','routed_response_pp','executed_response_pp','execution_minus_prediction_pp']:
   per_model=[np.mean([r[key] for r in rs if r['model']==m]) for m in models]
   row[key]=float(np.mean(per_model));row[key+'_training_sd']=float(np.std(per_model,ddof=1)) if len(models)>1 else None
  replicate_groups=[[r['executed_response_pp'] for r in rs if r['model']==m] for m in models]
  available_sds=[float(np.std(v,ddof=1)) for v in replicate_groups if len(v)>1]
  row['mean_within_model_sampling_sd_pp']=float(np.mean(available_sds)) if policy!='soft_argmax' and available_sds else None
  row['aggregation']='average sampling seeds within each trained model, then equal-weight training models; training and sampling variation kept separate'
  family.append(row)
 write(OUT/'response_family_summary.json',family);csvout(OUT/'response_family_summary.csv',family)
 mechanism=[]
 for result in read(OUT/'run_summary.json'):
  rows=jl(OUT/'runs'/result['run']/'person_ledger.jsonl')
  for feasible in [False,True]:
   sub=[r for r in rows if r['planner_mask']['pt']==feasible]
   if not sub:continue
   mechanism.append(dict(model=result['model'],card=result['card'],policy=result['policy'],sampling_seed=result['sampling_seed'],pt_planner_feasible=feasible,n_initial=len(rows),n_subgroup=len(sub),subgroup_mean_raw_pt_probability=float(np.mean([r['original_probability']['pt'] for r in sub])),raw_pt_probability_mass_per_initial_person=sum(r['original_probability']['pt'] for r in sub)/len(rows),assigned_pt_people=sum(r['assigned_mode']=='pt' for r in sub),routed_pt_people=sum(r['routed_mode']=='pt' for r in sub),actual_pt_people=sum(r['events']['outbound_pt_boardings']>0 for r in sub),outbound_completed=sum(r['events']['outbound_arrived'] for r in sub)))
 csvout(OUT/'pt_feasibility_mechanism.csv',mechanism)
 sh=[]
 for setting in ['historical_link_speed','corrected_mode_max_speed']:
  b=jl(OUT/f'audits/shanghai/B0_{setting}/person_trip_events.jsonl');d=jl(OUT/f'audits/shanghai/D1_{setting}/person_trip_events.jsonl');b={r['person_id']:r for r in b};d={r['person_id']:r for r in d};assert b.keys()==d.keys()
  dif=np.array([d[k]['outbound_trip_min']-b[k]['outbound_trip_min'] for k in sorted(b)])
  low,high=interval(dif);sh.append(dict(setting=setting,n_initial=len(b),n_complete_paired=len(dif),paired_delay_minus_baseline_mean_min=float(dif.mean()),ci95_low_min=low,ci95_high_min=high,interpretation='Perceived-input delay; physical timetable unchanged; within-person simulated-trip response, not human delay benefit.'))
 write(OUT/'audits/shanghai_time_response.json',sh);csvout(OUT/'audits/shanghai_time_response.csv',sh)
 sg=[]
 for card in ['C1_heavy_rain','C2_fare_increase','C3_transit_delay','C4_road_disruption','C5_joint_rain_delay']:
  bp=OUT/'audits/singapore/C0_baseline/person_trip_events.jsonl';dp=OUT/f'audits/singapore/{card}/person_trip_events.jsonl'
  if not bp.exists() or not dp.exists():continue
  b={r['person_id']:r for r in jl(bp)};d={r['person_id']:r for r in jl(dp)};ids=[k for k in sorted(b) if b[k]['outbound_trip_min'] is not None and d[k]['outbound_trip_min'] is not None]
  dif=np.array([d[k]['outbound_trip_min']-b[k]['outbound_trip_min'] for k in ids]);low,high=interval(dif)
  sg.append(dict(card=card,n_initial=len(b),n_complete_in_both=len(ids),baseline_complete=sum(r['outbound_arrived'] for r in b.values()),scenario_complete=sum(r['outbound_arrived'] for r in d.values()),paired_complete_trip_response_min=float(dif.mean()),ci95_low_min=low,ci95_high_min=high,interpretation='Legacy link-freespeed execution; timing restricted to people completing outbound in both scenarios, so no full-population efficiency conclusion.'))
 write(OUT/'audits/singapore_paired_complete_time.json',sg);csvout(OUT/'audits/singapore_paired_complete_time.csv',sg)
 runs=read(OUT/'run_summary.json')
 report=['# Helsinki execution and timing audits','',f'Completed Helsinki simulations: {len(runs)}/140; completed baseline-delay pairs: {len(pairs)}/70. Tables use successful actual MATSim runs only.','',
  'Population: the fixed first 1000 historical Helsinki people (population seed 2026), baseline and perceived PT delay, and original full supply. Sampling seeds 42/2026/7 are coupled by person. The physical GTFS timetable is unchanged. These repeated simulations do not constitute additional independent people.','',
  'Model artifacts: frozen S9; original validation-KL-selected controlled soft KL and direction-magnitude checkpoints (three training seeds each); fixed MNL-S choice combined with independently trained departure branches selected by validation departure MAE (three training seeds). This execution comparison concerns these deployed artifacts; the separately controlled checkpoint-selection comparison is reported in the baseline experiment.','',
  f'Observed completion: {sum(r["outbound_completed"] for r in runs)}/{sum(r["n_initial"] for r in runs)} outbound person-run records and {sum(r["return_completed"] for r in runs)}/{sum(r["n_initial"] for r in runs)} return records. There are {sum(r["stuck_people"] for r in runs)} stuck person-run records and {sum(r["unfinished_legs"] for r in runs)} unfinished legs.','',
  'The production and Shanghai writers already replan after applying the model departure shift. No departure-shift ordering bug was found. `departure_timing_audit.csv` reports the actual requested-versus-adjusted itinerary changes.','',
  'Shanghai: official MATSim mode vehicle types cap walk at 1.34 m/s and bike at 4.17 m/s. All 321 people complete outbound and return in both B0 and D1; population XML hashes are unchanged. The historical link-speed mean-time reduction reverses after speed correction. See `audits/shanghai_time_response.csv`.','',
  'Singapore: historical `mean_trip_time_min` is an arrival-matched completed-leg average. It excludes failed/incomplete legs and is not a person-trip mean. The new per-person ledgers preserve all 10000 initial people and all failures. `audits/singapore_paired_complete_time.csv` conditions explicitly on completing both outbound trips. The completion-or-horizon measure assigns incomplete departed trips the 30 h administrative horizon; it is a descriptive failure penalty, not an inferred survival-time mean.','',
  'All experimental files are additive. Survey forms/responses, original model checkpoints and historical events/populations are read-only. Helsinki retains original network-mode link-speed execution for the probability-to-PT-user analysis; its trip times are not calibrated.','',
  'Per-person ledgers preserve raw/filtered probability, assignment, route/fallback, observed mode, PT boarding counts, arrival and failure state. Primary shares use initial N; PT retention separately uses assigned PT persons. Bootstraps resample paired people within a fixed trained model and sampling seed; variation across training seeds is reported separately.','',
  '| Model | Policy | Trained artifacts | Predicted response (pp) | Executed response (pp) | Training SD (pp) | Mean within-model sampling SD (pp) |',
  '|---|---|---:|---:|---:|---:|---:|']
 for r in family:
  fmt=lambda x:'NA' if x is None else f'{x:.3f}'
  report.append(f'| {r["model_family"]} | {r["policy"]} | {r["n_training_models"]} | {fmt(r["predicted_response_pp"])} | {fmt(r["executed_response_pp"])} | {fmt(r["executed_response_pp_training_sd"])} | {fmt(r["mean_within_model_sampling_sd_pp"])} |')
 report+=['','Positive execution-minus-prediction values indicate a less negative executed PT response for this delay contrast. Planner screening guarantees only the availability represented by the current planner and input supply; it does not establish universally improved response fidelity or calibrated trip-time benefits.','',
  '`response_gap_cancellation_summary.csv` separately reports the absolute family signed mean, the mean absolute model gap after averaging sampling seeds within each model, and the mean absolute per-run gap. A near-zero signed family average can conceal opposite model errors; it is not evidence that every model preserves its predicted response. These are descriptive summaries, with no additional inferential claim.','',
  'Final completeness and provenance are recorded in `verification.json`. It checks all 140 expected keys, the initial population and paired inputs, policy transformation, adjusted-departure feasibility, route/fallback, person counts, identical physical configuration, model/source hashes and raw-event hashes. A failed 4 GB startup attempt is retained under `failed_attempts` and excluded from all results; the successful simulations use 6 GB.']
 (OUT/'EXECUTION_REVISION_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 from execution_gap_audit import main as gap_audit
 gap_audit()

if __name__=='__main__':main()
