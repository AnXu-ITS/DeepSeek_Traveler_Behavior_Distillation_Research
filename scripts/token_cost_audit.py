#!/usr/bin/env python
"""Aggregate real token usage across all repeat_records.jsonl files and project cost."""
import json, glob, os, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sorted(glob.glob(os.path.join(ROOT, "data", "**", "repeat_records.jsonl"), recursive=True))

# Dedup by completion_id across files (superset files repeat earlier records)
seen = {}          # completion_id -> record
file_records = {}  # file -> list of records

for f in files:
    recs = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            recs.append(r)
            cid = r.get("completion_id")
            if cid:
                seen[cid] = r
    file_records[f] = recs

def stats(recs):
    n = len(recs)
    p = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in recs)
    c = sum(r.get("usage", {}).get("completion_tokens", 0) for r in recs)
    t = p + c
    cache_hit = sum(r.get("usage", {}).get("prompt_cache_hit_tokens", 0) for r in recs)
    cache_miss = sum(r.get("usage", {}).get("prompt_cache_miss_tokens", 0) for r in recs)
    reason = sum((r.get("usage", {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for r in recs)
    with_usage = sum(1 for r in recs if r.get("usage"))
    return dict(n=n, prompt=p, completion=c, total=t, cache_hit=cache_hit,
                cache_miss=cache_miss, reasoning=reason, with_usage=with_usage)

print("=== per-file stats ===")
for f in files:
    s = stats(file_records[f])
    rel = os.path.relpath(f, ROOT)
    print(f"{rel}: n={s['n']} (with_usage={s['with_usage']}) prompt={s['prompt']:,} completion={s['completion']:,} total={s['total']:,} cache_hit={s['cache_hit']:,} cache_miss={s['cache_miss']:,} reasoning={s['reasoning']:,}")

# Deduplicated aggregate (all unique completion_ids across all files)
uniq = list(seen.values())
s = stats(uniq)
print("\n=== DEDUPED ALL-TIME TOTAL (unique completion_id) ===")
print(f"unique calls: {s['n']}")
print(f"prompt_tokens: {s['prompt']:,}")
print(f"completion_tokens: {s['completion_tokens'] if False else s['completion']:,} (reasoning {s['reasoning']:,})")
print(f"total_tokens: {s['total']:,}")
print(f"cache_hit: {s['cache_hit']:,}  cache_miss: {s['cache_miss']:,}")

# per-call averages
if s['n']:
    print("\n=== per-call averages (deduped) ===")
    print(f"avg prompt: {s['prompt']/s['n']:.0f}")
    print(f"avg completion: {s['completion']/s['n']:.0f}")
    print(f"avg total: {s['total']/s['n']:.0f}")
    print(f"avg cache_hit: {s['cache_hit']/s['n']:.0f}")
    print(f"avg reasoning: {s['reasoning']/s['n']:.0f}")

# completion token distribution
comp = sorted(r.get('usage', {}).get('completion_tokens', 0) for r in uniq)
if comp:
    print("\n=== completion_tokens distribution (deduped) ===")
    print(f"min={comp[0]} p25={comp[len(comp)//4]} median={comp[len(comp)//2]} p75={comp[3*len(comp)//4]} max={comp[-1]}")

# Cost projection using candidate price scenarios (USD per 1M tokens)
print("\n=== COST (USD, deduped all-time) ===")
scenarios = {
    "deepseek-chat (V3): $0.27 in miss / $0.07 hit / $1.10 out": (0.27, 0.07, 1.10),
    "deepseek-reasoner (R1): $0.55 in miss / $0.14 hit / $2.19 out": (0.55, 0.14, 2.19),
    "hi-end proxy: $2 in / $8 out": (2.0, 2.0, 8.0),
}
for name, (pmiss, phit, pout) in scenarios.items():
    in_cost = (s['cache_miss']/1e6)*pmiss + (s['cache_hit']/1e6)*phit
    out_cost = (s['completion']/1e6)*pout
    total = in_cost + out_cost
    print(f"{name}: in=${in_cost:.2f} out=${out_cost:.2f} total=${total:.2f}")
