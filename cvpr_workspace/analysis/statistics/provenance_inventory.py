"""Observed lineage and split facts; absence of evidence remains unknown."""
from reviewer_closure import *
import yaml
def main():
 out=OUT/'provenance';out.mkdir(exist_ok=True);stages=[]
 for p in sorted((ROOT/'outputs').glob('student*/checkpoints/best.pt')):
  c=torch.load(p,map_location='cpu',weights_only=False)
  meta={k:v for k,v in c.items() if k not in ['model_state','extractor_state','optimizer_state','scheduler_state'] and isinstance(v,(str,int,float,bool,type(None)))}
  split=p.parent.parent/'split_manifest.json';hist=p.parent.parent/'training_history.json'
  stages.append(dict(stage=p.parent.parent.name,checkpoint=str(p.relative_to(ROOT)),sha256=sha(p),keys=sorted(c),metadata=meta,
   split_path=str(split.relative_to(ROOT)) if split.exists() else None,split_sha256=sha(split) if split.exists() else None,history_path=str(hist.relative_to(ROOT)) if hist.exists() else None,history_sha256=sha(hist) if hist.exists() else None))
 write(out/'model_lineage_inventory.json',stages)
 splits=[]
 paths=list((ROOT/'outputs').glob('student*/split_manifest.json'))+list((ROOT/'data').glob('*/split_manifest.json'))
 for p in paths:
  x=read(p);personas=x.get('personas',x.get('persona_split'));checks=None
  if isinstance(personas,dict) and all(k in personas for k in ('train','val','test')):
   checks={f'{a}_{b}_overlap':sorted(set(personas[a])&set(personas[b])) for a,b in itertools.combinations(['train','val','test'],2)}
  splits.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),manifest=x,persona_overlap_checks=checks))
 for split in ['train','val','test']:
  es=jl(BUNDLE/f'{split}_endpoints.jsonl');splits.append(dict(path=str((BUNDLE/f'{split}_endpoints.jsonl').relative_to(ROOT)),sha256=sha(BUNDLE/f'{split}_endpoints.jsonl'),matched_split=split,n=len(es),personas=sorted({e['persona'] for e in es}),input_hashes=sorted({e['input_hash'] for e in es})))
 write(out/'split_manifest_all_stages.json',splits)
 datasets=[]
 for r in read(ROOT/'releases/s7_w3_generic_core_v1/data_manifest/data_manifest.json')['datasets']:
  p=ROOT/Path(r['path']);datasets.append(dict(kind=r['kind'],path=r['path'],recorded_sha256=r['sha256'],current_sha256=sha(p),matches=sha(p)==r['sha256'],meaning='S7 mechanism n_states in old manifest counts quadruplet records, not individual endpoints' if 'quadruplets' in r['kind'] else 'See archived dataset manifest'))
 write(out/'dataset_hash_audit.json',datasets)
 surveys=[]
 import openpyxl
 for directory in ['上海调查问卷','新加披调查问卷']:
  for p in (ROOT/directory).glob('*.xlsx'):
   wb=openpyxl.load_workbook(p,read_only=True,data_only=True);ws=wb.worksheets[0];all_rows=list(ws.iter_rows(values_only=True));head=[str(v) if v is not None else '' for v in all_rows[0]];datecol=next((i for i,h in enumerate(head) if ('提交' in h and '时间' in h) or '时间戳' in h or h=='Timestamp'),None);times=[];unparsed=0
   if datecol is not None:
    for row in all_rows[1:]:
     value=row[datecol]
     if isinstance(value,datetime.datetime):times.append(value.isoformat());continue
     parsed=None
     for fmt in ['%Y/%m/%d %H:%M:%S','%Y-%m-%d %H:%M:%S','%m/%d/%Y %H:%M:%S']:
      try:parsed=datetime.datetime.strptime(str(value),fmt);break
      except ValueError:pass
     if parsed:times.append(parsed.isoformat())
     elif value is not None:unparsed+=1
   surveys.append(dict(file=str(p.relative_to(ROOT)),sha256=sha(p),sheet=ws.title,rows_excluding_header=len(all_rows)-1,columns=len(head),submission_time_column=head[datecol] if datecol is not None else None,first_submission=min(times) if times else None,last_submission=max(times) if times else None,unparsed_timestamps=unparsed));wb.close()
 write(out/'survey_files.json',surveys)
 facts=dict(shanghai=dict(original_submissions=341,mappable_respondents=321,tasks_per_person=10,scored_choices=3206,excluded_labels='2 unable and 2 bike outside frozen ownership availability mask',authenticity='User explicitly confirmed independent real respondents; fill-plan not used to guide responses. This is attestation, not independently verifiable recruitment proof.',order='Fixed questionnaire task order; no randomization evidence',recruitment='Not established from retained records',incentives='Not established',ethics_approval_or_exemption='Not established',od='Assumed 6 km matched on distance, not observed residential addresses',household_size='Not collected; 1/2/4 sensitivity, reference 2',car_semantics='Survey drive-or-taxi versus model ownership AND license; interpretation limitation retained'),
 singapore=dict(original_submissions=334,eligible=332,tasks_per_person=10,alignment='Structured mapping under assumptions, not direct text input; no Teacher labels on the same respondent tasks'),
 history=dict(training_leakage='Current matched endpoints and persona splits re-audited; no overlap found by matched audit',adaptive_test_access='Historical S3/S5/S7/S8/S9 test reports repeatedly inspected during project development; cannot call the current 437-state benchmark a newly blind final test',what_is_not_proved='No evidence that test labels entered optimizer; no complete timestamped access log proving no adaptive use',policy='Report reused historical benchmark and fixed seeds explicitly. No post-result reshuffle to manufacture a new independent test. New independent Teacher-labelled holdout would require new data collection beyond this local closure.',main_line='S3-C -> S5 joint M2 -> S7-W3 -> S9; S8 deprecated speed-error branch is not S9 weight initializer',source_refs=['releases/s7_w3_generic_core_v1/checkpoint/model.pt.meta.json','releases/s9_supply_aware_v2/config/training_s9.json','outputs/matched_response_v1/bundle/manifest.json']))
 write(out/'survey_and_holdout_facts.json',facts)
 print('Lineage',len(stages),'split files',len(paths),'survey files',len(surveys),flush=True)
if __name__=='__main__':main()
