"""Isolate the indexing fix using the ORIGINAL 2000 bootstrap draws.
Other historical score fields copied only after point-metric revalidation.
"""
from reviewer_closure import *
from sp_survey_score import interval
def main():
 changes=[]
 for path in sorted((ROOT/'outputs/shanghai_sp_v1/predictions').glob('*.jsonl')):
  rows=jl(path);ids=sorted({r['respondent_id'] for r in rows});kept=[r for r in rows if valid(r)];old=read(ROOT/f'outputs/shanghai_sp_v1/scores/{path.stem}.json');new=json.loads(json.dumps(old))
  P=np.array([[r['probabilities'].get(m,0) for m in MODES] for r in kept]);y=np.array([MODES.index(r['chosen_mode']) for r in kept])
  assert len(kept)==old['n_primary']
  assert abs(float((P.argmax(1)==y).mean())-old['accuracy'])<1e-12
  assert abs(float(-np.log(np.clip(P[np.arange(len(y)),y],1e-8,None)).mean())-old['nll'])<1e-10
  ix=np.random.default_rng(old['bootstrap']['seed']).integers(len(ids),size=(old['bootstrap']['resamples'],len(ids)))
  for card in new['cards']:
   by={r['respondent_id']:r for r in kept if r['card_id']==card}
   x=np.array([by[r]['probabilities'].get('pt',0)-float(by[r]['chosen_mode']=='pt') if r in by else np.nan for r in ids])
   mean,bounds=interval(x,ix);new['cards'][card]['model_minus_human_pt_ci']=bounds
   assert abs(mean-old['cards'][card]['model_minus_human_pt'])<1e-12
   if old['cards'][card]['model_minus_human_pt_ci']!=bounds:
    changes.append(dict(run=path.stem,card=card,n=len(by),old_ci=old['cards'][card]['model_minus_human_pt_ci'],corrected_ci=bounds,max_abs_change=max(abs(a-b) for a,b in zip(old['cards'][card]['model_minus_human_pt_ci'],bounds))))
  new['correction']=dict(original_score_sha256=sha(ROOT/f'outputs/shanghai_sp_v1/scores/{path.stem}.json'),policy='Same original respondent bootstrap indices; omit NaN sampled respondents instead of mapping to last valid respondent; no missing imputation. Other fields preserved after point-score recheck.',point_metrics_revalidated=True)
  write(OUT/'survey_corrected_full'/f'{path.stem}.json',new)
 write(OUT/'survey_corrected_full/change_log.json',changes)
 print('Legacy correction',len(changes),'changed card intervals in 39 runs; original files unchanged',flush=True)
if __name__=='__main__':main()
