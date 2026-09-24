"""Explicit drive-or-taxi availability sensitivity, not calibrated taxi validation."""
from reviewer_closure import *
from reference_pipeline.student_adapter import StudentAdapter
from traveler_distillation.schemas.state import UniversalTravelerState
from sp_survey_matsim import ORDER,score
from survey_comparison import compare
def main():
 torch.set_num_threads(4);out=OUT/'choice_set_sensitivity';out.mkdir(exist_ok=True)
 source=[r for c in ORDER for r in jl(NET/c/'states.jsonl')];states=[UniversalTravelerState.model_validate(r['state']) for r in source];original=[r for c in ORDER for r in jl(NET/c/'predictions.jsonl')]
 changed=0
 for st in states:
  for a in st.alternatives:
   if a.mode=='car':changed+=int(not a.available);a.available=True
 write(out/'protocol.json',dict(status='Exploratory diagnostic added after primary results; no selection or overwrite',intervention='Allow car-mode alternative for all respondents to approximate the questionnaire drive-or-taxi choice set; keep ownership/license features, prices, supply and other modes unchanged',changed_state_alternatives=changed,limitations=['No distinct taxi mode, waiting time or calibrated taxi fare','Car availability without ownership/license may be outside training support','Cannot declare this the uniquely correct questionnaire encoding','Inference only; do not reinterpret as physical taxi simulation']))
 scores={};paired={};subgroups={}
 for name in ['s9','supply_mnl']+[f'{v}_seed{s}' for v,s in itertools.product(VARIANTS,SEEDS)]:
  if name=='supply_mnl':dec=LinearUtility().load().decisions(states);reference=[r for c in ORDER for r in jl(OUT/f'baseline/survey/{c}.jsonl')]
  else:
   ck=ROOT/'releases/s9_supply_aware_v2/checkpoint/model.pt' if name=='s9' else ROOT/f'outputs/matched_response_v1/train/{name}/best.pt'
   dec=StudentAdapter(ck,device='cpu').predict(states);reference=original if name=='s9' else jl(OUT/f'network_survey/{name}.jsonl')
  rows=[dict(r,probabilities=d['mode_probabilities'],departure_time_shift_min=d['departure_time_shift_min']) for r,d in zip(original,dec)];writejl(out/f'{name}.jsonl',rows)
  scores[name]=score(rows);paired[name]=compare(rows,reference)
  eligible={r['respondent_id'] for r in source if r['state']['alternatives'][0]['available']}
  subgroups[name]={label:dict(original_mask=score([r for r in reference if (r['respondent_id'] in eligible)==own]),car_available=score([r for r in rows if (r['respondent_id'] in eligible)==own])) for label,own in [('original_car_offered',True),('original_car_masked',False)]}
 write(out/'scores.json',scores);write(out/'paired_vs_original_mask.json',paired);write(out/'subgroups.json',subgroups)
 csvout(out/'scores.csv',[dict(model=name,**{k:v for k,v in r.items() if isinstance(v,(float,int))}) for name,r in scores.items()])
 print('Drive-or-taxi availability diagnostic completed for 14 models',flush=True)
if __name__=='__main__':main()
