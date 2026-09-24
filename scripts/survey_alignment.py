#!/usr/bin/env python
"""Prepare aligned survey states, run frozen students, export offline teacher requests, and score."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from survey_alignment_lib import prepare, predict_student, teacher_requests, import_teacher, score, audit_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--workbook',type=Path,default=ROOT/'新加披调查问卷/Singapore Urban Mobility and Travel Choice Study _ 新加坡城市出行选择研究（回复）.xlsx')
    a.add_argument('--questionnaire',type=Path,default=ROOT/'新加披调查问卷/新加坡城市出行选择研究_调查问卷.md')
    a.add_argument('--config',type=Path,default=ROOT/'configs/survey_alignment.yaml')
    a.add_argument('--profiles',nargs='+',default=['primary'])
    a.add_argument('--output',type=Path,required=True)
    a = sub.add_parser('audit-inputs',help='Check all states against frozen feature statistics; no predictions')
    a.add_argument('--bundle',type=Path,required=True)
    a.add_argument('--checkpoint',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a = sub.add_parser('predict-student')
    a.add_argument('--bundle',type=Path,required=True)
    a.add_argument('--checkpoint',type=Path,required=True)
    a.add_argument('--profiles',nargs='+',default=['primary'])
    a.add_argument('--limit',type=int,default=0,help='Positive = smoke only; cannot be scored')
    a.add_argument('--output',type=Path,required=True)
    a = sub.add_parser('export-teacher',help='Writes requests only; never makes network calls')
    a.add_argument('--bundle',type=Path,required=True)
    a.add_argument('--profiles',nargs='+',default=['primary'])
    a.add_argument('--model',required=True)
    a.add_argument('--repeats',type=int,default=5)
    a.add_argument('--output',type=Path,required=True)
    a = sub.add_parser('import-teacher')
    a.add_argument('--bundle',type=Path,required=True)
    a.add_argument('--requests-dir',type=Path,required=True)
    a.add_argument('--responses',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a = sub.add_parser('score')
    a.add_argument('--bundle',type=Path,required=True)
    a.add_argument('--predictions',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--resamples',type=int,default=10000)
    a.add_argument('--seed',type=int,default=917)
    a = vars(p.parse_args())
    command = a.pop('command')
    fn = {'prepare':prepare,'predict-student':predict_student,'export-teacher':teacher_requests,
          'import-teacher':import_teacher,'score':score,'audit-inputs':audit_inputs}[command]
    result = fn(**a)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
