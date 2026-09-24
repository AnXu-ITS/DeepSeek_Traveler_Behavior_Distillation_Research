"""Before-call neutral-metadata preparation; refuses any already called condition."""
from pathlib import Path
import json, shutil, subprocess, sys
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/revision_20260921'
runner=Path(__file__).with_name('teacher_same_task.py')
for name,args in [('teacher_neutral_ownership',['--neutral-identifiers']),('teacher_accessibility_neutral_ownership',['--prompt','accessibility'])]:
    out=BASE/name
    if (out/'calls.jsonl').exists() and (out/'calls.jsonl').stat().st_size:
        raise RuntimeError('Refusing to modify a condition with API attempts')
    old=[json.loads(x) for x in (out/'selected_states.jsonl').read_text(encoding='utf-8').splitlines()]
    archive=out/'prepared_only_metadata_v1';archive.mkdir(exist_ok=True)
    for f in ['selected_states.jsonl','config.json']:
        if (archive/f).exists():raise RuntimeError('Before-call archive already exists; inspect rather than overwriting')
        shutil.copy2(out/f,archive/f)
    (out/'selected_states.jsonl').unlink()
    subprocess.run([sys.executable,str(runner),'prepare','--condition','ownership',*args],check=True)
    new=[json.loads(x) for x in (out/'selected_states.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(old)==len(new)==480
    for a,b in zip(old,new):
        assert (a['city'],a['respondent_id'],a['task'])==(b['city'],b['respondent_id'],b['task'])
        aa=a['state'];bb=b['state']
        for v in [aa,bb]:
            v['persona'].pop('persona_id');v['trip'].pop('trip_id');v['context'].pop('context_id')
            v['trip'].pop('origin_type');v['trip'].pop('destination_type')
        assert aa==bb
    (out/'before_call_metadata_audit.json').write_text(json.dumps({'api_attempts_before_change':0,'same_480_selected_targets':True,'all_student_encoded_values_unchanged':True,'archived_preparation':'prepared_only_metadata_v1','purpose':'Remove task/city hints and Student-unused origin/destination types before querying either neutral prompt condition.'},indent=2),encoding='utf-8')
print('Both uncalled neutral conditions finalized; every Student feature unchanged.')
