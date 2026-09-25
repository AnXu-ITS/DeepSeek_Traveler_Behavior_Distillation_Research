"""Final cross-cohort/protocol/preservation checks; never edits API evidence."""
import collections,json
from pathlib import Path
from controlled import ROOT,OUT,read_json,write_json,file_hash
from offline_audits import rows
from synthetic_teacher import DEST

def main():
    plan=read_json(DEST/'formal_launch_plan.json')
    assert file_hash(DEST/'pilot/analysis.json')==plan['pilot_analysis_sha256']
    assert file_hash(DEST/'pilot/acquisition_audit.json')==plan['pilot_audit_sha256']
    assert file_hash(DEST/'acquisition_protocol.json')==plan['protocol_sha256']
    assert file_hash(DEST/'transport_recovery_amendment.json')==plan['amendment_sha256']
    allrecords=[];usage={};valid_ids={};unknown=[]
    for cohort in ['pilot','formal']:
        folder=DEST/cohort;status=read_json(folder/'status.json');audit=read_json(folder/'acquisition_audit.json')
        assert status['complete'] and audit['all_checks_passed']
        records=[read_json(p) for p in sorted(folder.glob('attempt_*.json'))]
        assert len(records)==status['attempts']
        allrecords+=records;usage[cohort]=audit['usage']
        valid_ids[cohort]={r['state_id'].split(':')[0] for r in records if r.get('valid')}
        if cohort=='formal':
            assert valid_ids[cohort]==set(plan['persona_ids'])
            assert status['valid']==plan['required_valid']
            assert min(r['utc'] for r in records)>plan['utc']
    assert not valid_ids['pilot']&valid_ids['formal']
    completion_ids=[r['completion_id'] for r in allrecords if r.get('completion_id')]
    assert len(completion_ids)==len(set(completion_ids)), 'Duplicate completion across acquisition attempts/cohorts'
    manifest=read_json(ROOT/'evidence/paper_20260924/artifact_manifest.json')
    assert all(file_hash(ROOT/a['path'])==a['sha256'] for a in manifest['artifacts'])
    sync=read_json(ROOT.parent/'elsevier_sync_report.json')
    assert all(file_hash(Path(sync[side])/a['path'])==a['sha256'] for a in sync['files'] for side in ['source','destination'])
    total={k:sum(u[k] for u in usage.values()) for k in ['total_attempts','usage_reported_attempts','usage_unknown_attempts','input_tokens','output_tokens','cached_input_tokens','total_tokens','request_time_rate_quota_estimate_usd','known_usage_all_offpeak_cache_estimate_usd','known_usage_peak_no_cache_usd']}
    write_json(OUT.parent/'final_delivery_audit.json',dict(passed=True,formal_frozen_before_first_request=True,formal_ids_match_frozen_plan=True,pilot_records_unchanged_since_formal_lock=True,cross_cohort_unique_completion_ids=len(completion_ids),historical_pinned_artifacts_unchanged=len(manifest['artifacts']),manuscript_files_unchanged=len(sync['files']),valid_responses=sum(read_json(DEST/c/'status.json')['valid'] for c in ['pilot','formal']),attempt_outcomes=dict(collections.Counter(str(r.get('http_status',r.get('error_type'))) for r in allrecords)),combined_usage=total,cost_interpretation='Quota-equivalent estimate on reported usage only, not an invoice or verified balance debit. Unreported usage remains unknown. Monthly subscription is not added per request.',new_human_validation='deferred by author',manuscript_writing='not started',driver_sha256=file_hash(__file__)))
    print(json.dumps(read_json(OUT.parent/'final_delivery_audit.json')),flush=True)

if __name__=='__main__':main()
