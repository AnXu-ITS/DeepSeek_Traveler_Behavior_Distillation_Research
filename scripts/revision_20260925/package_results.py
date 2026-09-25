"""Package small auditable experimental outputs; keep large MATSim events local."""
from pathlib import Path
import json,shutil,zipfile
from controlled import ROOT,OUT,file_hash,write_json
from offline_audits import RAW

def main():
    source=OUT.parent;target=ROOT/'evidence/experiments_20260925';target.mkdir(parents=True,exist_ok=True)
    status=json.loads((source/'stage_status.json').read_text());assert status['A']=='complete' and status['D']['completed']==40
    assert json.loads((source/'physical_supply/verification.json').read_text())['verified']==40
    for rel in ['EXPERIMENT_REPORT_CN.md','stage_status.json','protocol.json','training_verification.json','weight_selection_summary.json','weight_selection_summary.csv','controlled/paired_comparisons.json','delay_family/delay_only_test_summary.json','audit/combination_leakage.json','audit/repeat_disjoint_summary.json','physical_supply/supply_audit.json','physical_supply/run_metrics.json','physical_supply/verified_contrasts.json','physical_supply/verification.json']:
        out=target/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,out)
    entries={}
    # Include checkpoints, per-unit predictions, frozen synthetic states and all
    # failed API attempts. No .env, credentials, raw questionnaires or events.
    for p in source.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.csv','.npz','.pt','.md','.xml','.yaml','.py']:
            entries['outputs/revision_20260925/'+p.relative_to(source).as_posix()]=p
    physical=RAW/'outputs/revision_20260925/physical_supply'
    for p in physical.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(physical)
        if 'output' in rel.parts:continue
        if p.suffix in ['.json','.jsonl'] or p.name=='config.xml':entries['physical_supply/'+rel.as_posix()]=p
    # Include the actual reused ledgers too, so paired statistics can be checked
    # without access to the author's original execution directory.
    for r in json.loads((source/'physical_supply/run_inventory.json').read_text()):
        logical=f'{r["model"]}__{r["arm"]}__planner_sampling__seed{r["seed"]}'
        for filename in ['person_ledger.jsonl','result.json','config.xml']:
            entries['physical_supply/runs/'+logical+'/'+filename]=Path(r['source'])/filename
    for p in (ROOT/'scripts/revision_20260925').glob('*.py'):entries['scripts/revision_20260925/'+p.name]=p
    for rel in ['cvpr_workspace/analysis/statistics/synthetic_audit.py','docs/SYNTHETIC_ANALYSIS_PROTOCOL.md']:
        entries[rel]=ROOT/rel
    manifest={name:dict(sha256=file_hash(p),bytes=p.stat().st_size) for name,p in sorted(entries.items())}
    write_json(target/'package_manifest.json',manifest)
    archive=target/'reproducible_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(entries.items()):z.write(p,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        import hashlib
        assert all(hashlib.sha256(z.read(name)).hexdigest()==info['sha256'] for name,info in manifest.items())
    write_json(target/'package_verification.json',dict(files=len(entries),bytes=archive.stat().st_size,sha256=file_hash(archive),crc='pass',snapshot_consistency='all embedded files match manifest',large_events='Retained in local physical_supply runs, excluded from Git package; event SHA256 and scripts preserved',status=status))
    print(json.dumps(json.loads((target/'package_verification.json').read_text())),flush=True)

if __name__=='__main__':main()
