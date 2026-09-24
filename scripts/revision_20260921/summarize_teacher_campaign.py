"""Recompute metadata for completed Teacher conditions; never performs API calls."""
from pathlib import Path
import hashlib,importlib.util,json

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'outputs/revision_20260921'
SOURCE=Path(__file__).with_name('teacher_same_task.py')
CONDITIONS={'teacher':1440,'teacher_ownership':120,
            'teacher_accessibility_neutral_ownership':1440,'teacher_neutral_ownership':120}

def main():
    spec=importlib.util.spec_from_file_location('revision_teacher_metadata',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records={}
    for name,expected in CONDITIONS.items():
        base=RESULTS/name
        config=json.loads((base/'config.json').read_text(encoding='utf-8-sig'))
        calls=[json.loads(x) for x in (base/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
        valid=[x for x in calls if x.get('valid')]
        keys={(x['city'],x['respondent_id'],str(x['task']),x['repeat']) for x in valid}
        if len(keys)!=expected or len(valid)!=expected:raise RuntimeError(f'Incomplete or duplicate targets in {name}')
        module.OUT=base;module.CFG=config
        module.summarize()
        summary=json.loads((base/'run_summary.json').read_text())
        records[name]={'expected_valid_targets':expected,'scope':'full diagnostic cohort' if expected==1440 else 'four-person pilot',
                       'summary':summary,'calls_sha256':hashlib.sha256((base/'calls.jsonl').read_bytes()).hexdigest()}
    result={'status':'complete','conditions':records,'total_valid_targets':sum(x['expected_valid_targets'] for x in records.values()),
            'total_attempts':sum(x['summary']['attempts'] for x in records.values()),
            'estimated_cost_usd_offpeak_to_peak':[sum(x['summary']['estimated_cost_usd_offpeak_to_peak'][i] for x in records.values()) for i in range(2)],
            'cost_is_estimate_not_billed_amount':True,
            'no_new_api_calls':True,'temperature_interpretation':'Requested 0.2; thinking/effort omitted. Provider documents default thinking with temperature ignored; see api_behavior_note.json.'}
    (RESULTS/'teacher_campaign_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='conditions'},indent=2))

if __name__=='__main__':main()
