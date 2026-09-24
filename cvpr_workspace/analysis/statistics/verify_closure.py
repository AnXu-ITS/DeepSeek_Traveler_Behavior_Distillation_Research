"""Independent arithmetic and raw-artifact checks, no training or metric tuning."""
from reviewer_closure import *
from sp_survey_matsim import ORDER
from traveler_distillation.student.features import GLOBAL_CAT
def main():
 checks={};protocol=read(OUT/'protocol.json');modified=[]
 for name,h in protocol['source_sha256'].items():
  if sha(ROOT/name)!=h:modified.append(name)
 expected=['scripts\\shanghai\\sp_survey_score.py','scripts\\shanghai\\sp_survey_compare.py']
 checks['only_two_intentional_source_repairs']=set(n.replace('\\','/') for n in modified).issubset(set(n.replace('\\','/') for n in expected))
 checks['original_script_snapshots']=all(sha(OUT/('original_'+Path(p).name))==protocol['source_sha256'][p] for p in modified)
 checks['old_network_80_artifacts']=all(sha(NET/r['path'])==r['sha256'] for r in read(NET/'execution_verification.json')['frozen_output_artifacts'])
 checks['matched_audit_pass']=read(OUT/'matched_integrity/audit.json')['status']=='PASS'
 checks['dataset_hashes_match']=all(r['matches'] for r in read(OUT/'provenance/dataset_hash_audit.json'))
 model=LinearUtility().load();checks['categorical_order']=list(model.spec['cat_vocab_sizes'])==GLOBAL_CAT
 data=jl(BUNDLE/'test_endpoints.jsonl');pred=jl(OUT/'baseline/predictions.jsonl');X,M=model.design([e['state'] for e in data]);P,D=model.predict_arrays(X,M)
 checks['baseline_probability_valid']=bool(np.isfinite(P).all() and np.allclose(P.sum(1),1) and np.all(P[~M]==0))
 checks['baseline_prediction_replay']=bool(np.allclose(P,[r['student'] for r in pred],atol=1e-12) and np.allclose(D,[r['student_departure'] for r in pred],atol=1e-12))
 meta=read(OUT/'baseline/selection.json');candidates=read(OUT/'baseline/candidates.json');chosen=min((c for c in candidates if c['success']),key=lambda c:c['val_macro_kl'])
 checks['validation_only_selection']=chosen['l2']==meta['selected_l2'] and not meta['test_used_for_selection']
 bm=read(OUT/'baseline/metrics.json');Y=np.array([e['teacher_probs'] for e in data]);kl=np.array([sum(t*np.log(t/max(q,1e-8)) for t,q in zip(y,p) if t>0) for y,p in zip(Y,P)]);macro=np.mean([kl[[i for i,e in enumerate(data) if e['source']==s]].mean() for s in sorted({e['source'] for e in data})])
 checks['baseline_independent_macro_kl']=abs(macro-bm['state']['macro_source_kl'])<1e-12
 by={e['id']:i for i,e in enumerate(data)};gaps=[]
 for r in jl(BUNDLE/'test_pairs.jsonl'):
  a=by[r['base']];b=by[r['cf']];mask=M[a]|M[b];gaps.append(abs((Y[b]-Y[a])-(P[b]-P[a]))[mask].mean())
 checks['baseline_independent_response']=abs(float(np.mean(gaps))-bm['response']['response_gap'])<1e-12
 altered=json.loads(json.dumps(data[0]['state']));altered['persona']['persona_id']='ID_NOT_A_FEATURE';altered['trip']['trip_id']='ID_NOT_A_FEATURE';XX,MM=model.design([data[0]['state'],altered]);checks['no_identity_features']=bool(np.array_equal(XX[0],XX[1]) and np.array_equal(MM[0],MM[1]))
 checks['all_39_corrected_scores']=len(list((OUT/'survey_corrected_full').glob('*_p04*.json')))==39
 checks['all_14_network_models']=len(read(OUT/'network_survey/scores.json'))==14
 checks['all_14_choice_set_models']=len(read(OUT/'choice_set_sensitivity/scores.json'))==14
 checks['all_22_historical_scenarios']=len(read(OUT/'historical/three_city_execution.json'))==22
 fvr=read(OUT/'fvr/analysis.json');checks['fvr_exact_correction']=fvr['improved']==3 and fvr['worsened']==0 and fvr['exact_mcnemar_two_sided']==.25 and fvr['exact_iid_bootstrap_ci']==[-.5,0.]
 executions=[read(p) for p in sorted((OUT/'execution').glob('*/verified.json'))];checks['all_48_execution_runs']=len(executions)==48 and all(r['pass'] and r['exit_code']==0 for r in executions)
 identity_checks=[]
 import xml.etree.ElementTree as ET
 for r in executions:
  cd=OUT/'execution'/r['run'];dec=jl(cd/'decisions.jsonl');man=read(cd/'adapter_manifest.json')['decisions'];people=jl(cd/'verified_people.jsonl');population=ET.parse(cd/'population.xml').getroot()
  identity_checks.append(len(dec)==len(man)==len(people)==321 and {p['person_id'] for p in man}=={p['person_id'] for p in people}=={p.get('id') for p in population.findall('person')} and sum(p['outbound_arrived']==1 for p in people)==r['outbound_completed'] and sum(p['return_arrived']==1 for p in people)==r['return_completed'])
 checks['execution_identity_and_journey_arithmetic']=all(identity_checks) and len(identity_checks)==48
 checks['s9_argmax_reproduces_original_three_cards']=all(read(OUT/f'execution/s9_{c}_argmax_soft_seed0/verified.json')['intention_counts']==read(NET/c/'run_summary.json')['student_counts'] for c in ['B0','D1','A_TRANSFER'])
 checks['common_physical_supply']=len({r['network_sha256'] for r in executions})==1 and len({r['schedule_sha256'] for r in executions})==1
 stats={}
 for m,mask,policy in itertools.product(['s9','supply_mnl'],['soft','planner_mask'],['argmax','sample']):
  rows=[r for r in executions if (r['model'],r['mask'],r['policy'])==(m,mask,policy)];label=f'{m}_{policy}_{mask}';cards={}
  for c in ['B0','D1','A_TRANSFER']:
   rs=[r for r in rows if r['card']==c];values=[r['executed_counts'].get('pt',0)/321 for r in rs]
   cards[c]=dict(n_runs=len(rs),original_pt_probability=rs[0]['original_pt_probability'],policy_pt_probability=rs[0]['policy_pt_probability'],executed_pt_share_mean=float(np.mean(values)),executed_pt_share_seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,all_completed=all(r['outbound_completed']==321 and r['return_completed']==321 for r in rs),fallback_legs=[r['fallback'] for r in rs])
  deltas={}
  for c in ['D1','A_TRANSFER']:
   b={r['seed']:r for r in rows if r['card']=='B0'};cs={r['seed']:r for r in rows if r['card']==c};values=[(cs[s]['executed_counts'].get('pt',0)-b[s]['executed_counts'].get('pt',0))/321 for s in sorted(b)]
   deltas[c]=dict(mean=float(np.mean(values)),per_seed=values,training_uncertainty='none; these are policy sampling seeds, not training seeds',seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
  stats[label]=dict(cards=cards,paired_executed_response=deltas)
 write(OUT/'execution/policy_aggregates.json',stats)
 checks={k:bool(v) for k,v in checks.items()}
 result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,changed_source_files=modified,environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,scipy=__import__('scipy').__version__),limitations=['No training replay of historical neural models','Exploratory analyses on historically reused holdout','Deployment-native departure heads differ: S9 tanh bounded +-60 min, ridge baseline unbounded; not a mode-only causal comparison'])
 write(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
