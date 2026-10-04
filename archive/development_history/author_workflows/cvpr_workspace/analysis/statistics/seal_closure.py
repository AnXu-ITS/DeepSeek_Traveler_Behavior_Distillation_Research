"""Strict manifest validation, final tests, copies and immutable artifact inventory."""
from reviewer_closure import *
import importlib.util,subprocess
def main():
 assert read(OUT/'verification.json')['status']=='PASS'
 manifest=ROOT/'cvpr_workspace/result_package/statistics/ait_reviewer_closure_20260920/statistics.yaml'
 validator=Path.home()/'.codex/skills/cvpr-statistics/scripts/validate_statistics_manifest.py'
 spec=importlib.util.spec_from_file_location('statistics_validator',validator);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 v=module.Validation();m=read(manifest);m['status']='accepted';module.validate_manifest(m,ROOT,v)
 assert not v.errors,v.errors
 write(OUT/'statistics_contract_validation.json',dict(status='PASS',manifest=str(manifest.relative_to(ROOT)),manifest_status=read(manifest)['status'],strict_accepted_checks=True,errors=v.errors,validator_sha256=sha(validator)))
 proc=subprocess.run([sys.executable,'-m','pytest','tests/test_reviewer_closure.py','tests/test_shanghai_sp_matsim.py','-q'],cwd=ROOT,capture_output=True,text=True)
 write(OUT/'tests.json',dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr));assert proc.returncode==0
 report=(OUT/'REPORT.md').read_text(encoding='utf-8');assert '|\n\n|' not in report
 paper=ROOT.parent/'蒸馏出行意图paper/ccfa-review-reports/AIT_REVIEW_EXECUTION_COMPLETED_20260920.md'
 paper.write_text(report,encoding='utf-8')
 # Capture every result and run artifact, including failed attempts; no self-hash.
 artifacts=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name not in ['artifact_manifest.json','seal_verification.json'] and '__pycache__' not in p.parts:
   artifacts.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
 codefiles=list((ROOT/'cvpr_workspace/analysis/statistics').glob('*.py'))+[ROOT/'cvpr_workspace/analysis/statistics/REPRODUCE.md',ROOT/'tests/test_reviewer_closure.py',ROOT/'scripts/shanghai/sp_survey_score.py',ROOT/'scripts/shanghai/sp_survey_compare.py',manifest,ROOT/'docs/AIT_REVIEW_CLOSURE_20260920.md']
 artifacts += [dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in codefiles]
 write(OUT/'artifact_manifest.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifacts=artifacts,paper_report=dict(path=str(paper),sha256=sha(paper)),note='Files immutable after this seal; artifact_manifest and seal_verification excluded from self-hashing'))
 assert all((ROOT/r['path']).stat().st_size==r['bytes'] for r in artifacts)
 write(OUT/'seal_verification.json',dict(status='PASS',artifact_count=len(artifacts),total_bytes=sum(r['bytes'] for r in artifacts),main_checks=len(read(OUT/'verification.json')['checks']),tests=proc.stdout.strip(),statistics_contract_strict_pass=True,artifact_manifest_sha256=sha(OUT/'artifact_manifest.json')))
 print(json.dumps(read(OUT/'seal_verification.json'),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
