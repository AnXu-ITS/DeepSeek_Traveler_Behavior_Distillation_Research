"""Offline token accounting. No credentials, network, training, or model calls.

Author mode reads retained JSONL files and exports aggregate-only evidence.
Public/reviewer mode reprices that aggregate evidence without private payloads.
Costs are September 25, 2026 rate-card scenarios, never an invoice.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ('prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens')
RATES = {'input_hit': .044, 'input_miss': 1.32, 'output': 3.96}
CNY_RATES = {'input_hit': .30, 'input_miss': 9.0, 'output': 27.0}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def usage(row):
    u = row.get('usage')
    if not u:
        return None
    if not all(k in u for k in FIELDS):
        raise ValueError('Incomplete usage: do not silently treat missing tokens as zero')
    u = {k: int(u[k]) for k in FIELDS}
    assert u['prompt_tokens'] == u['prompt_cache_hit_tokens'] + u['prompt_cache_miss_tokens']
    assert all(v >= 0 for v in u.values())
    return u

def aggregate(rows):
    unique, without_id = {}, 0
    for row in rows:
        cid = row.get('completion_id')
        if not cid:
            without_id += 1
            continue
        if cid in unique:
            assert usage(unique[cid]) == usage(row), 'Conflicting duplicated usage'
        unique[cid] = row
    values = list(unique.values())
    measured = [usage(row) for row in values if usage(row) is not None]
    sums = {k: sum(u[k] for u in measured) for k in FIELDS}
    times = [row['created'] for row in values if isinstance(row.get('created'), (int, float))]
    return dict(records=len(rows), unique_completions=len(unique),
                duplicate_completion_records=len(rows)-without_id-len(unique),
                records_without_completion_id=without_id,
                completions_with_usage=len(measured),
                completions_missing_usage=len(values)-len(measured),
                returned_models=dict(Counter(row.get('model', 'not_retained') for row in values)),
                created_utc_range=[datetime.fromtimestamp(f(times), timezone.utc).isoformat() for f in (min,max)] if times else [],
                usage=sums, total_tokens=sums['prompt_tokens']+sums['completion_tokens'])

def reprice(group):
    u=group['usage']
    peak=(u['prompt_cache_hit_tokens']*RATES['input_hit']+
          u['prompt_cache_miss_tokens']*RATES['input_miss']+
          u['completion_tokens']*RATES['output'])/1e6
    cny=(u['prompt_cache_hit_tokens']*.30+u['prompt_cache_miss_tokens']*9.0+u['completion_tokens']*27.0)/1e6
    return {'offpeak_usd':peak/2, 'peak_usd':peak, 'offpeak_cny':cny/2, 'peak_cny':cny}

def collect(root):
    files=sorted((root/'data').glob('*/repeat_records.jsonl'))
    files+=sorted((root/'outputs/revision_20260921').glob('teacher*/calls.jsonl'))
    files+=[root/'evidence/e2_efficiency/deepseek_calls.jsonl']
    assert all(p.is_file() for p in files)
    sources, groups, allrows=[], {}, []
    primary={'student_v0_3_s3','student_s5_joint','singapore_accessibility'}
    buckets={'primary_supervision_archives':[], 'other_historical_archives':[]}
    for p in files:
        rows=[json.loads(line) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        rel=p.relative_to(root).as_posix()
        a=aggregate(rows)
        a['path']=rel; a['sha256']=digest(p)
        a['http_status_counts']=dict(Counter(str(row.get('http_status','not_retained')) for row in rows))
        a['validity_counts']=dict(Counter(str(row.get('valid',row.get('ok','not_retained'))) for row in rows))
        a['request_max_tokens_with_usage']=dict(Counter(str(row.get('max_tokens','not_retained')) for row in rows if row.get('usage')))
        sources.append(a);allrows.extend(rows)
        if p.name=='repeat_records.jsonl':
            buckets['primary_supervision_archives' if p.parent.name in primary else 'other_historical_archives'].extend(rows)
        else:
            groups[p.parent.name]=aggregate(rows)
    # Primary archives take precedence; exclude their replicated completions from the historical remainder.
    primary_ids={row['completion_id'] for row in buckets['primary_supervision_archives'] if row.get('completion_id')}
    buckets['other_historical_archives']=[row for row in buckets['other_historical_archives'] if row.get('completion_id') not in primary_ids]
    groups.update({name:aggregate(rows) for name,rows in buckets.items()})
    groups['retained_total_deduplicated']=aggregate(allrows)
    # Groups are a partition by completion ID; unsuccessful rows without an ID are not charged as zero.
    assert sum(g['total_tokens'] for k,g in groups.items() if k!='retained_total_deduplicated')==groups['retained_total_deduplicated']['total_tokens']
    return {'schema_version':1, 'pricing_checked':'2026-09-25',
            'pricing_source':'https://api-docs.deepseek.com/quick_start/pricing/',
            'model':'deepseek-v4-pro', 'currency':'USD', 'peak_rates_per_million':RATES,
            'cny_peak_rates_per_million':CNY_RATES,
            'cny_pricing_source':'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
            'offpeak_multiplier':.5,
            'scope':'Retained records only; union by completion_id; includes billed-token parse failures when usage exists. Not complete lifecycle cost or provider invoice.',
            'limitations':['Missing usage is unknown, not zero.',
                          'Primary archives contain acquired labels beyond the final benchmark subset; not exact marginal final-model acquisition.',
                          'Other historical archives include deprecated S8 and earlier development; not current-model training cost.',
                          'Reasoning tokens are included in completion_tokens; never add them twice.',
                          'No historic rate schedule, credits, taxes, exchange conversion, complete failed-attempt ledger or hardware tariff is reconstructed.'],
            'sources':sources,'groups':groups}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    src=parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--retained-root',type=Path)
    src.add_argument('--aggregate',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    evidence=collect(args.retained_root) if args.retained_root else json.loads(args.aggregate.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    prices={name:reprice(g) for name,g in evidence['groups'].items()}
    (args.output/'api_usage_aggregate.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    # Forecast a different future workload from the FINAL accessibility-neutral campaign only.
    ref=evidence['groups']['teacher_accessibility_neutral_ownership']
    n=ref['completions_with_usage']
    means={k:v/n for k,v in ref['usage'].items()}
    budget=[]
    for name,calls in [('synthetic_pilot_12x12x3',432),('formal_example_60x12x3',2160),('human_example_120x10x3',3600)]:
        expected={k:v*calls*1.1 for k,v in means.items()}
        # A planning envelope, not an enforced hard cap: 10% attempts, 2,000 uncached input, 16,384 output each.
        envelope_calls=(calls*110+99)//100
        envelope=(envelope_calls*(2000*1.32+16384*3.96))/1e6
        budget.append({'id':name,'successful_calls':calls,'attempt_allowance':envelope_calls,
                       'assumption':'10% extra usage-bearing calls; reference token mix; not a confidence interval',
                       'expected_tokens_with_allowance':sum(expected[k] for k in ('prompt_tokens','completion_tokens')),
                       **reprice({'usage':expected}), 'peak_no_cache_token_envelope_usd':envelope,
                       'peak_no_cache_token_envelope_cny':envelope_calls*(2000*9+16384*27)/1e6})
    report={'cost_is_estimate_not_invoice':True,'prices':prices,'planning_reference_mean_tokens':means,
            'budget':budget,'api_calls_executed_by_this_script':0}
    (args.output/'cost_estimates.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (args.output/'api_cost_table.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['group','unique_completions','completions_with_usage','completions_missing_usage','prompt_tokens','completion_tokens','offpeak_usd','peak_usd','offpeak_cny','peak_cny'])
        writer.writeheader()
        for name,g in evidence['groups'].items():
            writer.writerow({'group':name,**{k:g[k] for k in ('unique_completions','completions_with_usage','completions_missing_usage')},
                             **{k:g['usage'][k] for k in ('prompt_tokens','completion_tokens')},**prices[name]})
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
