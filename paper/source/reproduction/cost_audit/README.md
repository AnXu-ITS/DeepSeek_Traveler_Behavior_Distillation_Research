# Offline API cost reconstruction

These aggregate records contain no participant payloads. They reprice retained usage at the official 25 September 2026 V4 Pro rate card; they are not an invoice or complete lifecycle total.

Run from this directory:

```bash
python audit_api_costs_20260925.py --aggregate api_usage_aggregate.json --output recomputed
```

The script makes no API calls. Raw-ledger mode requires the separately retained author workspace. File hashes and coverage are in `api_usage_aggregate.json`. Future-experiment entries in `cost_estimates.json` are planning scenarios, not completed results or approved sample sizes.
