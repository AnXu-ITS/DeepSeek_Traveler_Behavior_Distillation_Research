#!/usr/bin/env python
"""Build frozen-model states from the coded respondents, using the v3 package's own
mapping functions (build_persona / build_state / offered_modes) and the DEPLOYED
fixed card order.

Why not call survey_adapter.py's CLI directly: it requires form_id in forms.json and
generates task_order from that form's randomised order. The deployed instrument used a
single fixed order that is not one of the ten randomised forms, so the states are built
here in the true deployed order and task_order records it. No package file is modified
and no mapping rule is re-implemented.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WB / 'src'))

DEPLOYED_ORDER = ['B0', 'W1', 'D1', 'WD1', 'F1', 'P1', 'R1', 'A_WALK', 'A_WAIT', 'A_TRANSFER']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package', type=Path, default=WB / 'docs/plans/shanghai_survey_v3')
    ap.add_argument('--run-dir', type=Path, default=WB / 'outputs/shanghai_sp_v1')
    ap.add_argument('--p04', type=int, nargs='+', default=[1, 2, 4])
    args = ap.parse_args()

    sys.path.insert(0, str(args.package))
    import survey_adapter as sa  # noqa: E402  (package mapping logic, unmodified)

    if set(sa.FORMS) and DEPLOYED_ORDER not in [list(v) for v in sa.FORMS.values()]:
        print('note: deployed fixed order is not any forms.json permutation '
              '(randomisation was not implemented at deploy time)')

    summary = {}
    for p04 in args.p04:
        src = args.run_dir / f'respondents_p04{p04}.csv'
        with src.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        out_path = args.run_dir / f'states_p04{p04}.jsonl'
        avail_counter, n_states, n_people = Counter(), 0, 0
        with out_path.open('w', encoding='utf-8') as out:
            for raw in rows:
                person = sa.build_persona(raw)          # raises if a required code is missing
                availability = sa.offered_modes(raw)
                n_people += 1
                avail_counter['|'.join(k for k, v in availability.items() if v)] += 1
                for pos, cid in enumerate(DEPLOYED_ORDER, 1):
                    state = sa.build_state(person, cid, availability)
                    out.write(json.dumps(dict(respondent_id=raw['respondent_id'], card_id=cid,
                                              task_order=pos, state=state.model_dump(mode='json')),
                                         ensure_ascii=False) + '\n')
                    n_states += 1
        summary[f'p04_{p04}'] = dict(respondents=n_people, states=n_states,
                                     availability_mix=dict(avail_counter),
                                     states_sha256=sha(out_path))
        print(f'p04={p04}: {n_people} respondents -> {n_states} states  availability={dict(avail_counter)}')
    (args.run_dir / 'states_summary.json').write_text(
        json.dumps(dict(card_order=DEPLOYED_ORDER, package=str(args.package), variants=summary),
                   ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
