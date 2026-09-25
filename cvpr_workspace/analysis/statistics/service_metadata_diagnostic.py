"""Post hoc service-metadata stratification; primary analysis is unchanged."""
from synthetic_audit import *

def main():
    calls=[];persons=[];groups=[]
    protocol=read_json(DEST/'service_metadata_diagnostic_protocol.json')
    for cohort in ['pilot','formal']:
        folder=DEST/cohort;assert read_json(folder/'status.json')['complete']
        primary={r['persona']:r['signed_minus_softkl'] for r in read_json(folder/'analysis.json')['primary']}
        by=collections.defaultdict(list)
        for p in sorted(folder.glob('attempt_*.json')):
            r=read_json(p)
            if not r.get('valid'):continue
            u=r.get('usage',{});reasoning=(u.get('completion_tokens_details') or {}).get('reasoning_tokens')
            regime='unreported' if reasoning is None else 'positive' if reasoning>0 else 'zero'
            row=dict(cohort=cohort,attempt=r['attempt'],persona=r['state_id'].split(':')[0],state_id=r['state_id'],repeat=r['repeat'],utc=r['utc'],reported_reasoning=regime,reasoning_tokens=reasoning,output_tokens=u.get('completion_tokens'),input_tokens=u.get('prompt_tokens'),elapsed_s=r.get('elapsed_s'),returned_model=r.get('returned_model'),system_fingerprint=r.get('system_fingerprint'),usage_detail_keys=sorted((u.get('prompt_tokens_details') or {}).keys()))
            calls.append(row);by[row['persona']].append(row)
        for pid,rs in by.items():
            assert len(rs)==36
            counts=collections.Counter(r['reported_reasoning'] for r in rs)
            regime='all_'+next(iter(counts)) if len(counts)==1 else 'mixed'
            persons.append(dict(cohort=cohort,persona=pid,regime=regime,counts=dict(counts),primary_signed_minus_softkl=primary[pid],first_utc=min(r['utc'] for r in rs),last_utc=max(r['utc'] for r in rs)))
        for regime in sorted({r['regime'] for r in persons if r['cohort']==cohort}):
            xs=[r['primary_signed_minus_softkl'] for r in persons if r['cohort']==cohort and r['regime']==regime]
            stats=interval(xs) if len(xs)>1 else dict(n_personas=len(xs),mean=float(np.mean(xs)),ci_low=None,ci_high=None)
            groups.append(dict(cohort=cohort,regime=regime,**stats))
    summary=[]
    for cohort in ['pilot','formal']:
        for regime in ['positive','zero','unreported']:
            rs=[r for r in calls if r['cohort']==cohort and r['reported_reasoning']==regime]
            if rs:summary.append(dict(cohort=cohort,regime=regime,n_calls=len(rs),first_utc=min(r['utc'] for r in rs),last_utc=max(r['utc'] for r in rs),median_output_tokens=float(np.median([r['output_tokens'] for r in rs])),returned_models=sorted({r['returned_model'] for r in rs})))
    write_rows(DEST/'service_metadata_calls.jsonl',calls);write_rows(DEST/'service_metadata_personas.jsonl',persons)
    write_json(DEST/'service_metadata_diagnostic.json',dict(call_summary=summary,primary_persona_groups=groups,post_hoc=True,protocol_sha256=file_hash(DEST/'service_metadata_diagnostic_protocol.json'),code_sha256=file_hash(__file__),primary_samples_unchanged=True,interpretation=protocol['interpretation'],additional_limit='Reported reasoning-token differences can reflect serving/reporting changes; they do not prove a different backend model or the absence of internal reasoning. Pointwise group intervals are descriptive, not a causal comparison.'))
    print(json.dumps(read_json(DEST/'service_metadata_diagnostic.json')),flush=True)

if __name__=='__main__':main()
