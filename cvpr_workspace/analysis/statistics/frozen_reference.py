"""Frozen S7/S9 reference on the common pool, not a matched-loss treatment."""
from reviewer_closure import *
from traveler_distillation.matched_response.metrics import encode,predict,pair_scores,interaction_scores,summarize,paired_cluster_difference
from traveler_distillation.student import FeatureExtractor,TravelerStudent
from traveler_distillation.accessibility.accessibility_features import S8FeatureExtractor,TravelerStudentS8
def main():
 torch.set_num_threads(4);endpoints=jl(BUNDLE/'test_endpoints.jsonl');pairdata=jl(BUNDLE/'test_pairs.jsonl');idata=jl(BUNDLE/'test_interactions.jsonl');cache={}
 for name,release,extcls,modelcls in [('s7','s7_w3_generic_core_v1',FeatureExtractor,TravelerStudent),('s9','s9_supply_aware_v2',S8FeatureExtractor,TravelerStudentS8)]:
  p=ROOT/f'releases/{release}/checkpoint/model.pt';c=torch.load(p,map_location='cpu',weights_only=False);ext=extcls().from_state_dict(c['extractor_state']);m=modelcls(c['config'],c['feature_spec']);m.load_state_dict(c['model_state']);m.eval()
  pred=predict(m,endpoints,{e['id']:encode(e,ext) for e in endpoints},'cpu',128);pairs=pair_scores(pred,pairdata,[.0001,.001,.01],'union');inter=interaction_scores(pred,idata)
  d=OUT/f'frozen_reference/{name}';writejl(d/'predictions.jsonl',pred);writejl(d/'pairs.jsonl',pairs);writejl(d/'interactions.jsonl',inter)
  metrics=summarize(pred,pairs,inter);metrics['state']['departure_median']=float(np.median([r['departure_mae'] for r in pred]));metrics['note']='Historical frozen checkpoint, differing training exposures and initialization; reference only, not a neutral matched loss contrast'
  for source in metrics['state']['by_source']:metrics['state']['by_source'][source]['departure_median']=float(np.median([r['departure_mae'] for r in pred if r['source']==source]))
  metrics['checkpoint_sha256']=sha(p);write(d/'metrics.json',metrics);cache[name]=(pred,pairs,inter)
 comparisons={key:paired_cluster_difference(cache['s9'][i],cache['s7'][i],key,10000,917) for i,key in [(0,'kl'),(0,'departure_mae'),(0,'fvr'),(0,'infeasible_pt_mass'),(1,'response_gap'),(2,'interaction_gap')]}
 write(OUT/'frozen_reference/paired_s9_minus_s7.json',comparisons)
 # Confirm raw FVR pairing remains identical under batch inference.
 paired=jl(OUT/'fvr/paired_12.jsonl');lookup={r['id'].split(':')[-1]:r for r in cache['s9'][0] if r['fvr'] is not None}
 assert all(lookup[r['id']]['fvr']==r['s9_violation'] for r in paired)
 print('Frozen S7/S9 full metrics and departure distribution complete',flush=True)
if __name__=='__main__':main()
