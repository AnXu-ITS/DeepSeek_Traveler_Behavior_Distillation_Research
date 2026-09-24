#!/usr/bin/env python
"""WJX "按文本" export -> v3 coded survey CSVs (respondents / sp_responses / coverage).

Decisions recorded before looking at any choices:
  * The deployed instrument used ONE FIXED card order (no randomisation) -> form_id is
    written as DEPLOYED_FIXED and task_order is the deployed display position.
  * P04 (household size) was NOT asked in the deployed instrument -> it is injected as an
    explicit reference encoding; three variants (1 / 2 / 4) are emitted so the sensitivity
    is reported next to the primary value (2). No other input is invented.
  * Rows that skip a required field, refuse income, or use an out-of-vocabulary habitual
    mode are RETAINED in the coverage report and marked not_primary_mappable.
  * "都不适合" -> choice_status=unable, chosen_mode empty. Skipped question -> skipped.

Written columns are the symbolic CODES of question_options.json (never booleans), so the
package adapter's own map is the single place where codes become schema values.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import openpyxl

WB = Path(__file__).resolve().parents[2]

DEPLOYED = {1: 'S01', 2: 'S02', 3: 'P01', 4: 'P02', 5: 'P03', 6: 'P05', 7: 'P06', 8: 'P07',
            9: 'P09', 10: 'P11', 11: 'P12', 12: 'P13', 13: 'P14',
            14: 'B0', 15: 'W1', 16: 'D1', 17: 'WD1', 18: 'F1', 19: 'P1', 20: 'R1',
            21: 'A_WALK', 22: 'A_WAIT', 23: 'A_TRANSFER'}
CARD_ORDER = [DEPLOYED[n] for n in range(14, 24)]
SCENARIO_KEYWORD = {14: '晴天', 15: '下雨', 16: '延误', 17: '下雨.*延误', 18: '票价',
                    19: '停车费', 20: '道路', 21: '站点较远', 22: '班次较少', 23: '换乘'}
MODE_BY_LABEL = {'自己开车或打车': 'car', '公交或地铁': 'pt', '普通自行车': 'bike', '步行': 'walk'}
UNABLE_LABEL = '都不适合'
SKIP_LABELS = {'（跳过）', '(跳过)'}
P04_VARIANTS = (1, 2, 4)
P04_PRIMARY = 2
CHANNEL = {'微信': 'wechat', '链接': 'link', '手机': 'mobile', '电脑': 'desktop'}
PERSONA_COLS = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P09', 'P11', 'P12', 'P13', 'P14']


def norm(s):
    s = str(s if s is not None else '').strip()
    for a, b in (('—', '-'), ('–', '-'), ('－', '-'), ('（', '('), ('）', ')')):
        s = s.replace(a, b)
    return re.sub(r'[\s\u3000]+', '', s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workbook', type=Path,
                    default=WB / '上海调查问卷/384387017_按文本_上海出行方式调查_341_341.xlsx')
    ap.add_argument('--package', type=Path, default=WB / 'docs/plans/shanghai_survey_v3')
    ap.add_argument('--output', type=Path, default=WB / 'outputs/shanghai_sp_v1')
    args = ap.parse_args()

    opts = json.loads((args.package / 'question_options.json').read_text(encoding='utf-8'))
    mapping = json.loads((args.package / 'field_mapping.json').read_text(encoding='utf-8'))
    version, missing_codes = mapping['version'], set(mapping['missing_codes'])
    label2code = {q['id']: {norm(o['label']): o['code'] for o in q['options']} for q in opts['questions']}
    text2id = {norm(q['text']): q['id'] for q in opts['questions']}
    persona_rules = mapping['persona_fields']
    qid2field = {v['question']: k for k, v in persona_rules.items()}

    wb = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header, data = rows[0], rows[1:]

    col_of, problems = {}, []
    for i, h in enumerate(header):
        m = re.match(r'^(\d+)[、.]\s*(.*)$', str(h or ''))
        if not m:
            continue
        n, qtext = int(m.group(1)), norm(m.group(2))
        qid = DEPLOYED.get(n)
        if qid is None:
            problems.append(f'Q{n}: no deployed mapping')
            continue
        col_of[qid] = i
        if n <= 13:
            hit = text2id.get(qtext)
            if hit is None:
                cands = [k for k in text2id if k and (k in qtext or qtext in k)]
                hit = cands[0] if len(cands) == 1 else None
            if hit != qid:
                problems.append(f'Q{n}->{qid}: package text mismatch: {qtext!r} -> matched {hit!r}')
        elif not re.search(SCENARIO_KEYWORD[n], qtext):
            problems.append(f'Q{n}->{qid}: scenario keyword {SCENARIO_KEYWORD[n]!r} absent in {qtext!r}')
    for qid in DEPLOYED.values():
        if qid not in col_of and qid != 'P04':
            problems.append(f'{qid}: column missing from export')

    respondents, responses, coverage = [], [], []
    reasons, card_choice = Counter(), Counter()
    card_unable, card_skipped, not_offered = Counter(), Counter(), Counter()
    channels, timestamps = Counter(), 0
    for r_i, row in enumerate(data, start=1):
        rid = f'SHP{r_i:04d}'

        def cell(qid):
            i = col_of.get(qid)
            if i is None or i >= len(row) or row[i] is None:
                return ''
            v = str(row[i]).strip()
            return '' if v in SKIP_LABELS else v

        codes, bad = {}, []
        for qid, field in qid2field.items():
            if qid == 'P04':      # not asked in the deployed instrument; injected below
                continue
            value = norm(cell(qid))
            if value == '':
                bad.append(f'{qid}/{field}:skipped')
                continue
            code = label2code.get(qid, {}).get(value)
            if code is None:
                bad.append(f'{qid}/{field}:unrecognized_label:{value[:16]}')
            elif code in missing_codes:
                bad.append(f'{qid}/{field}:missing_code:{code}')
            elif code in [norm(x) for x in persona_rules[field].get('unsupported', [])]:
                bad.append(f'{qid}/{field}:outside_frozen_vocabulary:{code}')
            else:
                codes[qid] = code
        for qid in ('S01', 'S02'):
            value = norm(cell(qid))
            code = label2code.get(qid, {}).get(value)
            if code != 'yes':
                bad.append(f'{qid}:{code or "unrecognized"}_not_yes')
            else:
                codes[qid] = code
        complete = not bad
        channel = CHANNEL.get(str(row[2]).strip() if len(row) > 2 else '', 'unknown')
        channels[channel] += 1
        try:
            ts = datetime.strptime(str(row[1]).strip(), '%Y/%m/%d %H:%M:%S').isoformat()
            timestamps += 1
        except Exception:
            ts = ''
        if complete:
            for p04 in P04_VARIANTS:
                rec = dict(respondent_id=rid, survey_version=version, form_id='DEPLOYED_FIXED',
                           recruitment_channel=channel, completed_at=ts, **codes, P04=str(p04))
                respondents.append(rec)
        else:
            for b in bad:
                reasons[b] += 1
        coverage.append(dict(respondent_id=rid, input_status='complete' if complete else 'not_primary_mappable',
                             reason=';'.join(bad), p04_injected='1|2|4' if complete else ''))
        car_av = codes.get('P06') == 'yes' and codes.get('P07') == 'yes'
        bike_av = codes.get('P09') == 'yes'
        for n in range(14, 24):
            card, raw = DEPLOYED[n], cell(DEPLOYED[n])
            if raw == '':
                status, mode = 'skipped', ''
                card_skipped[card] += 1
            elif norm(raw) == UNABLE_LABEL:
                status, mode = 'unable', ''
                card_unable[card] += 1
            else:
                mode = MODE_BY_LABEL.get(norm(raw), '')
                status = 'selected' if mode else 'unmapped'
                card_choice[(card, mode or f'UNMAPPED:{norm(raw)}')] += 1
            if mode == 'car' and not car_av:
                not_offered['car'] += 1
            if mode == 'bike' and not bike_av:
                not_offered['bike'] += 1
            responses.append(dict(respondent_id=rid, survey_version=version, form_id='DEPLOYED_FIXED',
                                  card_id=card, task_order=n - 13,
                                  car_available=str(car_av).lower(), bike_available=str(bike_av).lower(),
                                  choice_status=status, chosen_mode=mode))

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cols = ['respondent_id', 'survey_version', 'form_id', 'recruitment_channel', 'completed_at',
            'S01', 'S02'] + PERSONA_COLS
    for p04 in P04_VARIANTS:
        with (out / f'respondents_p04{p04}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows([r for r in respondents if r['P04'] == str(p04)])
    with (out / 'sp_responses.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['respondent_id', 'survey_version', 'form_id', 'card_id', 'task_order',
                                          'car_available', 'bike_available', 'choice_status', 'chosen_mode'])
        w.writeheader()
        w.writerows(responses)
    with (out / 'input_coverage.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['respondent_id', 'input_status', 'reason', 'p04_injected'])
        w.writeheader()
        w.writerows(coverage)

    human = {}
    for (card, mode), n in card_choice.items():
        human.setdefault(card, Counter())[mode] = n
    n_complete = len(respondents) // len(P04_VARIANTS)
    summary = dict(workbook=args.workbook.name, package_version=version, deployed_card_order=CARD_ORDER,
                   form_id='DEPLOYED_FIXED', total_submissions=len(data), complete_respondents=n_complete,
                   not_primary_mappable=len(data) - n_complete,
                   exclusion_reasons=dict(reasons.most_common()), scenario_rows=len(responses),
                   skipped=sum(card_skipped.values()), unable=sum(card_unable.values()),
                   choices_on_unavailable_mode=dict(not_offered), channels=dict(channels),
                   timestamps_parsed=timestamps, verification_problems=problems,
                   human_choice_by_card={c: dict(v) for c, v in sorted(human.items())},
                   p04_reference_encodings=list(P04_VARIANTS), p04_primary=P04_PRIMARY,
                   note='Choices are never used in mapping. P04 was not asked in the deployed instrument; '
                        'reference values 1/2/4 are injected and reported as such.')
    (out / 'conversion_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                                                 encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('total_submissions', 'complete_respondents', 'not_primary_mappable',
                                              'exclusion_reasons', 'scenario_rows', 'skipped', 'unable',
                                              'choices_on_unavailable_mode', 'channels', 'verification_problems')},
                     ensure_ascii=False, indent=2))
    print('\nhuman choice share per card:')
    for card in CARD_ORDER:
        c, tot = human.get(card, Counter()), sum(human.get(card, Counter()).values()) or 1
        print(f'  {card:<11} n={tot:<4} pt={c.get("pt",0)/tot:.3f}  car={c.get("car",0)/tot:.3f}  '
              f'bike={c.get("bike",0)/tot:.3f}  walk={c.get("walk",0)/tot:.3f}')
    print('\nwrote:', out)


if __name__ == '__main__':
    raise SystemExit(main())
