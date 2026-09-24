#!/usr/bin/env python
"""Frozen S9 (and matched checkpoints) vs the Shanghai stated choices.

Reuses reference_pipeline.StudentAdapter (the production load path that hard-asserts the
frozen S9 schema) and reports, per model and per P04 reference encoding:
  accuracy / NLL / Brier / L1 on the pre-declared primary subset, per-card stated vs
  predicted mode shares, per-persona breakdown, and respondent-cluster bootstrap CIs for
  paired differences (model accuracy vs always-PT baseline, model-minus-human PT share).

Primary subset (fixed before scoring): complete-input respondents, cards whose
choice_status == selected, and choices inside the model's offered-mode set. Choices of a
mode the frozen availability rule masks off (2 bike cases) are excluded here and counted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WB / 'src'))
sys.path.insert(0, str(WB))

from reference_pipeline.student_adapter import StudentAdapter  # noqa: E402
from traveler_distillation.schemas.state import UniversalTravelerState  # noqa: E402

MODES = ['car', 'pt', 'bike', 'walk']
VARIANT_NAMES = [f'{v}_seed{s}' for v in ('soft_kl', 'ce_kl', 'signed_l1', 'direction_magnitude')
                 for s in (42, 2026, 7)]


def load_states(path: Path):
    states, keys = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        states.append(UniversalTravelerState.model_validate(row['state']))
        keys.append((row['respondent_id'], row['card_id']))
    return states, keys


def cluster_mean(values: dict, resp_ids: list) -> np.ndarray:
    """Per-respondent mean over that respondent's kept cards (NaN if none kept)."""
    bucket = defaultdict(list)
    for (rid, _card), v in values.items():
        bucket[rid].append(v)
    return np.array([np.mean(bucket[rid]) if rid in bucket else np.nan for rid in resp_ids])


def interval(x: np.ndarray, rng_idx: np.ndarray):
    """Cluster bootstrap over respondents. ``x`` is indexed by respondent position and may
    contain NaN for respondents without a kept row; ``rng_idx`` holds respondent positions."""
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
    ap.add_argument('--package', type=Path, default=WB / 'docs/plans/shanghai_survey_v3')
    ap.add_argument('--p04', type=int, nargs='+', default=[1, 2, 4])
    ap.add_argument('--models', nargs='+', default=['s9'] + VARIANT_NAMES)
    ap.add_argument('--resamples', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=917)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()

    sys.path.insert(0, str(args.package))
    import survey_adapter as sa  # noqa: E402

    ckpt = {'s9': WB / 'releases/s9_supply_aware_v2/checkpoint/model.pt'}
    ckpt.update({n: WB / f'outputs/matched_response_v1/train/{n}/best.pt' for n in VARIANT_NAMES})

    choices = {}
    with (args.run_dir / 'sp_responses.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            choices[(r['respondent_id'], r['card_id'])] = r

    (args.run_dir / 'predictions').mkdir(exist_ok=True)
    (args.run_dir / 'scores').mkdir(exist_ok=True)
    report = {}
    for p04 in args.p04:
        states, keys = load_states(args.run_dir / f'states_p04{p04}.jsonl')
        with (args.run_dir / f'respondents_p04{p04}.csv').open(encoding='utf-8-sig', newline='') as f:
            raw_rows = {r['respondent_id']: r for r in csv.DictReader(f)}
        people = {rid: sa.build_persona(raw) for rid, raw in raw_rows.items()}
        resp_ids = sorted({k[0] for k in keys})
        rng = np.random.default_rng(args.seed)
        rng_idx = rng.integers(len(resp_ids), size=(args.resamples, len(resp_ids)))
        report[f'p04_{p04}'] = {}
        for model_name in args.models:
            adapter = StudentAdapter(ckpt[model_name], device='cpu')
            decisions = adapter.predict(states, batch_size=args.batch_size)
            probs, human = {}, {}
            for (rid, card), dec in zip(keys, decisions):
                probs[(rid, card)] = np.array([dec['mode_probabilities'].get(m, 0.0) for m in MODES])
                row = choices[(rid, card)]
                y = np.zeros(4)
                if row['choice_status'] == 'selected' and row['chosen_mode'] in MODES:
                    y[MODES.index(row['chosen_mode'])] = 1.0
                human[(rid, card)] = (row['choice_status'], row['chosen_mode'], y)
            with (args.run_dir / 'predictions' / f'{model_name}_p04{p04}.jsonl').open('w', encoding='utf-8') as out:
                for (rid, card), p in probs.items():
                    out.write(json.dumps(dict(respondent_id=rid, card_id=card,
                                              probabilities={m: float(p[i]) for i, m in enumerate(MODES)},
                                              choice_status=human[(rid, card)][0],
                                              chosen_mode=human[(rid, card)][1]), ensure_ascii=False) + '\n')
            keep, excluded = [], Counter()
            for k in keys:
                status, mode, _y = human[k]
                if status != 'selected':
                    excluded[f'choice_{status}'] += 1
                    continue
                if mode not in ('pt', 'walk') and probs[k][MODES.index(mode)] == 0.0:
                    excluded[f'choice_{mode}_not_offered'] += 1
                    continue
                keep.append(k)
            yi = np.array([human[k][2].argmax() for k in keep])
            P = np.array([probs[k] for k in keep])
            Y = np.eye(4)[yi]
            acc = float((P.argmax(1) == yi).mean())
            nll = float(-np.log(np.clip(P[np.arange(len(keep)), yi], 1e-8, None)).mean())
            brier = float(((P - Y) ** 2).sum(1).mean())
            l1 = float(np.abs(P - Y).sum(1).mean())
            correct = {k: float(probs[k].argmax() == human[k][2].argmax()) for k in keep}
            pt_always = {k: float(human[k][2][1] == 1.0) for k in keep}
            pt_gap = {k: float(probs[k][1] - human[k][2][1]) for k in keep}
            acc_curve, acc_ci = interval(cluster_mean(correct, resp_ids), rng_idx)
            base_curve, base_ci = interval(cluster_mean(pt_always, resp_ids), rng_idx)
            gain_curve, gain_ci = interval(cluster_mean({k: correct[k] - pt_always[k] for k in keep}, resp_ids), rng_idx)
            pt_curve, pt_ci = interval(cluster_mean(pt_gap, resp_ids), rng_idx)
            cards = {}
            for card in sorted({k[1] for k in keep}):
                ks = [k for k in keep if k[1] == card]
                Pc = np.array([probs[k] for k in ks]); yc = np.array([human[k][2] for k in ks])
                m, ci = interval(cluster_mean({k: pt_gap[k] for k in ks}, resp_ids), rng_idx)
                cards[card] = dict(n=len(ks),
                                   human_share={mm: float(yc[:, i].mean()) for i, mm in enumerate(MODES)},
                                   model_prob={mm: float(Pc[:, i].mean()) for i, mm in enumerate(MODES)},
                                   model_minus_human_pt=m, model_minus_human_pt_ci=ci,
                                   accuracy=float((Pc.argmax(1) == yc.argmax(1)).mean()))
            subgroups = {}
            for field in ('age_group', 'income_group', 'habitual_mode'):
                g = defaultdict(list)
                for k in keep:
                    g[getattr(people[k[0]], field)].append(k)
                subgroups[field] = {str(key): dict(
                    n=len(ks),
                    accuracy=float(np.mean([probs[k].argmax() == human[k][2].argmax() for k in ks])),
                    model_pt=float(np.mean([probs[k][1] for k in ks])),
                    human_pt=float(np.mean([human[k][2][1] for k in ks])))
                    for key, ks in sorted(g.items(), key=lambda kv: str(kv[0]))}
            res = dict(model=model_name, checkpoint=str(ckpt[model_name]),
                       checkpoint_sha256=adapter.checkpoint_sha256(), p04_reference=p04,
                       n_primary=len(keep), n_excluded=dict(excluded),
                       accuracy=acc, accuracy_cluster_ci=acc_ci,
                       always_pt_accuracy=base_curve, always_pt_ci=base_ci,
                       accuracy_minus_always_pt=dict(mean=gain_curve, ci95=gain_ci),
                       nll=nll, brier=brier, probability_l1=l1,
                       model_minus_human_pt=dict(mean=pt_curve, ci95=pt_ci, n_clusters=int((~np.isnan(cluster_mean(pt_gap, resp_ids))).sum())),
                       cards=cards, subgroups=subgroups,
                       bootstrap=dict(unit='respondent (all ten cards kept together)',
                                      resamples=args.resamples, seed=args.seed),
                       primary_subset_policy='complete input AND choice_status=selected AND chosen mode offered by the frozen availability rule')
            (args.run_dir / 'scores' / f'{model_name}_p04{p04}.json').write_text(
                json.dumps(res, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            report[f'p04_{p04}'][model_name] = dict(accuracy=acc, always_pt=base_curve, nll=nll, l1=l1,
                                                    model_minus_human_pt=pt_curve, pt_ci=pt_ci,
                                                    accuracy_minus_always_pt=gain_curve,
                                                    accuracy_minus_always_pt_ci=gain_ci, n_primary=len(keep))
            print(f'p04={p04} {model_name:<26} n={len(keep):<5} acc={acc:.3f} (always-PT {base_curve:.3f}, '
                  f'gain {gain_curve:+.3f} [{gain_ci[0]:+.3f},{gain_ci[1]:+.3f}]) nll={nll:.3f} '
                  f'model-human_pt={pt_curve:+.3f} [{pt_ci[0]:+.3f},{pt_ci[1]:+.3f}]', flush=True)
    (args.run_dir / 'score_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                                     encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
