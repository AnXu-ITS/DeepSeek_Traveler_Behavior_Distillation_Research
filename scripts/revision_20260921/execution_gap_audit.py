"""Separate signed cancellation from descriptive model/run absolute gaps."""
import collections
import csv
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/revision_20260921/execution'


def main():
    pairs = json.loads((OUT / 'response_summary.json').read_text(encoding='utf-8'))
    keys = [(r['model'], r['policy'], r['sampling_seed']) for r in pairs]
    assert len(keys) == len(set(keys))
    groups = collections.defaultdict(list)
    for row in pairs:
        family = re.sub(r'_seed\d+$', '', row['model'])
        groups[family, row['policy']].append(row)
    result = []
    model_details = []
    for (family, policy), rows in sorted(groups.items()):
        models = sorted({r['model'] for r in rows})
        per_model = {
            model: statistics.mean(r['execution_minus_prediction_pp'] for r in rows if r['model'] == model)
            for model in models
        }
        expected_models = 1 if family == 's9' else 3
        expected_samples = {0} if policy == 'soft_argmax' else {42, 2026, 7}
        complete = len(models) == expected_models and all(
            {r['sampling_seed'] for r in rows if r['model'] == model} == expected_samples
            for model in models
        )
        mean_gap = statistics.mean(per_model.values())
        result.append(dict(
            model_family=family,
            policy=policy,
            n_models=len(models),
            n_runs=len(rows),
            expected_models=expected_models,
            expected_runs=expected_models * len(expected_samples),
            complete=complete,
            family_signed_mean_gap_pp=mean_gap,
            abs_family_signed_mean_pp=abs(mean_gap),
            mean_abs_model_gap_after_sampling_mean_pp=statistics.mean(abs(g) for g in per_model.values()),
            mean_abs_per_run_gap_pp=statistics.mean(abs(r['execution_minus_prediction_pp']) for r in rows),
            min_model_signed_mean_gap_pp=min(per_model.values()),
            max_model_signed_mean_gap_pp=max(per_model.values()),
            n_negative_model_gaps=sum(g < 0 for g in per_model.values()),
            n_positive_model_gaps=sum(g > 0 for g in per_model.values()),
        ))
        model_details.extend(dict(model_family=family, policy=policy, model=model,
                                  sampling_mean_signed_gap_pp=gap,
                                  n_runs=sum(r['model'] == model for r in rows))
                             for model, gap in per_model.items())
    report = dict(
        metric='executed outbound PT response minus raw PT probability response, percentage points',
        semantics='n_runs counts baseline-delay pairs (two MATSim simulations per pair). Average sampling seeds within each trained model before averaging models. Absolute value before versus after averaging is reported separately; these are descriptive summaries, not inferential intervals.',
        source='response_summary.json',
        summary=result,
        model_details=model_details,
    )
    (OUT / 'response_gap_cancellation_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (OUT / 'response_gap_cancellation_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f'Gap audit: {sum(r["complete"] for r in result)}/12 complete family-policy groups; {len(pairs)} observed pairs')


if __name__ == '__main__':
    main()
