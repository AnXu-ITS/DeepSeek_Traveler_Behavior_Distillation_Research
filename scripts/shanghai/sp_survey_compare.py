#!/usr/bin/env python
"""Paired comparison of every model against frozen S9 on the Shanghai stated choices.

Per-respondent (cluster) paired differences with bootstrap CIs, on the primary subset and
for the primary P04 reference encoding. Also prints per-card stated vs predicted shares and
the S9 persona subgroups, so the report tables are reproducible from the saved predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WB = Path(__file__).resolve().parents[2]
MODES = ['car', 'pt', 'bike', 'walk']
VARIANT_NAMES = [f'{v}_seed{s}' for v in ('soft_kl', 'ce_kl', 'signed_l1', 'direction_magnitude')
                 for s in (42, 2026, 7)]


def load_pred(path: Path):
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r['respondent_id'], r['card_id'])] = r
    return out


def boot_ci(x: np.ndarray, rng_idx: np.ndarray):
    mask = ~np.isnan(x)
    if mask.sum() == 0:
        return float('nan'), [float('nan'), float('nan')]
    # Resample original respondent positions, then omit missing observations.
    # A missing draw must never become index -1 (the last valid respondent).
    draws = x[rng_idx]
    counts = np.isfinite(draws).sum(axis=1)
    nonempty = counts > 0
    if not nonempty.any():
        raise ValueError('All bootstrap replicates contain only missing respondents')
    boots = np.nansum(draws[nonempty], axis=1) / counts[nonempty]
    return float(x[mask].mean()), [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, default=WB / 'outputs/shanghai_sp_v1')
    ap.add_argument('--p04', type=int, default=2)
    ap.add_argument('--reference', default='s9')
    ap.add_argument('--resamples', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=917)
    args = ap.parse_args()

    run = args.run_dir
    with (run / 'sp_responses.csv').open(encoding='utf-8-sig', newline='') as f:
        resp = {(r['respondent_id'], r['card_id']): r for r in csv.DictReader(f)}
    models = [args.reference] + [n for n in VARIANT_NAMES if n != args.reference]
    pred, kept = {}, {}
    for name in models:
        path = run / 'predictions' / f'{name}_p04{args.p04}.jsonl'
        if not path.exists():
            continue
        pr = load_pred(path)
        pred[name] = pr
        keep = []
        for k, r in pr.items():
            row = resp[k]
            if row['choice_status'] != 'selected':
                continue
            if row['chosen_mode'] not in ('pt', 'walk') and r['probabilities'][row['chosen_mode']] == 0.0:
                continue
            keep.append(k)
        kept[name] = sorted(keep)
    models = [m for m in models if m in pred]
    base_keys = kept[args.reference]
    resp_ids = sorted({k[0] for k in base_keys})
    rng_idx = np.random.default_rng(args.seed).integers(len(resp_ids), size=(args.resamples, len(resp_ids)))

    def per_resp(vals):
        b = defaultdict(list)
        for k, v in vals.items():
            b[k[0]].append(v)
        return np.array([np.mean(b[r]) if r in b else np.nan for r in resp_ids])

    def metrics(name):
        pr = pred[name]
        acc = {k: float(np.array([pr[k]['probabilities'][m] for m in MODES]).argmax() == MODES.index(resp[k]['chosen_mode']))
               for k in base_keys}
        ptgap = {k: pr[k]['probabilities']['pt'] - (1.0 if resp[k]['chosen_mode'] == 'pt' else 0.0) for k in base_keys}
        nll = {}
        for k in base_keys:
            p = pr[k]['probabilities'][resp[k]['chosen_mode']]
            nll[k] = float(-np.log(max(p, 1e-8)))
        return acc, ptgap, nll

    ref_acc, ref_pt, ref_nll = metrics(args.reference)
    out = dict(p04_reference=args.p04, reference=args.reference, n_primary=len(base_keys),
               n_respondents=len(resp_ids), resamples=args.resamples, seed=args.seed,
               reference_metrics=dict(
                   accuracy=boot_ci(per_resp(ref_acc), rng_idx)[0],
                   model_minus_human_pt=boot_ci(per_resp(ref_pt), rng_idx)[0],
                   nll=boot_ci(per_resp(ref_nll), rng_idx)[0]),
               comparisons={})
    print(f"{'model':<28}{'acc':>7}{'d_acc vs s9':>26}{'pt gap':>9}{'d_ptgap vs s9':>26}{'nll':>8}")
    for name in models:
        acc, pt, nll = metrics(name)
        row = dict(accuracy=boot_ci(per_resp(acc), rng_idx)[0],
                   model_minus_human_pt=boot_ci(per_resp(pt), rng_idx)[0],
                   nll=boot_ci(per_resp(nll), rng_idx)[0])
        if name != args.reference:
            da = boot_ci(per_resp({k: acc[k] - ref_acc[k] for k in base_keys}), rng_idx)
            dp = boot_ci(per_resp({k: pt[k] - ref_pt[k] for k in base_keys}), rng_idx)
            row['delta_accuracy_vs_reference'] = dict(mean=da[0], ci95=da[1])
            row['delta_pt_gap_vs_reference'] = dict(mean=dp[0], ci95=dp[1])
        out['comparisons'][name] = row
        print(f"{name:<28}{row['accuracy']:>7.3f}"
              f"{('%+.3f [%+.3f,%+.3f]' % (row.get('delta_accuracy_vs_reference', {}).get('mean', 0.0), *row.get('delta_accuracy_vs_reference', {}).get('ci95', [0, 0]))) if name != args.reference else '-':>26}"
              f"{row['model_minus_human_pt']:>9.3f}"
              f"{('%+.3f [%+.3f,%+.3f]' % (row.get('delta_pt_gap_vs_reference', {}).get('mean', 0.0), *row.get('delta_pt_gap_vs_reference', {}).get('ci95', [0, 0]))) if name != args.reference else '-':>26}"
              f"{row['nll']:>8.3f}")

    # per-card table for reference + best two by |pt gap|
    ranked = sorted([m for m in models if m != args.reference],
                    key=lambda m: abs(out['comparisons'][m]['model_minus_human_pt']))[:2]
    show = [args.reference] + ranked
    human = defaultdict(list)
    for k in base_keys:
        human[k[1]].append(1.0 if resp[k]['chosen_mode'] == 'pt' else 0.0)
    print('\nper-card PT share (human vs models):')
    print(f"{'card':<12}{'n':>5}{'human':>8}" + ''.join(f'{m[:18]:>20}' for m in show))
    cards = {}
    for card in sorted(human):
        cells = {}
        line = f'{card:<12}{len(human[card]):>5}{np.mean(human[card]):>8.3f}'
        for m in show:
            v = float(np.mean([pred[m][k]['probabilities']['pt'] for k in base_keys if k[1] == card]))
            cells[m] = v
            line += f'{v:>20.3f}'
        cards[card] = dict(n=len(human[card]), human_pt=float(np.mean(human[card])), model_pt=cells)
        print(line)
    out['per_card_pt_share'] = cards
    out['models_shown'] = show
    (run / f'comparison_p04{args.p04}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n',
                                                        encoding='utf-8')
    print('\nwrote', run / f'comparison_p04{args.p04}.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
