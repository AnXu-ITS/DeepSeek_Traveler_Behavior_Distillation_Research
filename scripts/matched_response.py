#!/usr/bin/env python
"""Prepare, train, smoke, evaluate, and compare the controlled response study."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import yaml

from traveler_distillation.matched_response.data import prepare, new_directory, read_json, write_json
from traveler_distillation.matched_response.experiment import VARIANTS, train_run, evaluate_run, validate_config
from traveler_distillation.matched_response.compare import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/matched_response.yaml')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Materialize and hash shared train/val/test pools; no inference')
    p.add_argument('--output',type=Path,required=True)
    for name in ('train','smoke'):
        p=sub.add_parser(name,help='Train+validation only; smoke never evaluates the real test set')
        p.add_argument('--bundle',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--variant',choices=sorted(VARIANTS))
        p.add_argument('--seed',type=int)
        p.add_argument('--device',default='cpu')
    p=sub.add_parser('evaluate',help='Explicitly evaluate a completed run; test is separate from training')
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--device',default='cpu')
    p=sub.add_parser('evaluate-suite',help='Evaluate each completed member of a suite')
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--device',default='cpu')
    p=sub.add_parser('compare',help='Paired differences across evaluated variants and seeds')
    p.add_argument('--evaluations',nargs='+',type=Path,required=True)
    p.add_argument('--reference',choices=sorted(VARIANTS),default='soft_kl')
    p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'));validate_config(cfg)
    if args.command=='prepare':
        result=prepare(ROOT,cfg,args.output)
        print('Prepared',result['bundle_id'])
        for split,m in result['splits'].items():
            print(split,{k:m[k] for k in ('endpoints','pairs','units','personas','source_counts')})
    elif args.command in ('train','smoke'):
        variants=[args.variant] if args.variant else (list(VARIANTS) if args.command=='smoke' else cfg['training']['variants'])
        seeds=[args.seed] if args.seed is not None else ([cfg['training']['seeds'][0]] if args.command=='smoke' else cfg['training']['seeds'])
        if any(s not in cfg['training']['seeds'] for s in seeds):
            parser.error('Seed must be declared in training.seeds')
        output=new_directory(args.output)
        suite=dict(status='running',command=args.command,bundle=str(args.bundle.resolve()),runs=[])
        write_json(output/'suite.json',suite)
        try:
            for seed in seeds:
                expected_initial=None
                for variant in variants:
                    folder=output/f'{variant}_seed{seed}'
                    run=train_run(args.bundle,cfg,folder,variant,seed,args.device,args.command=='smoke')
                    initial=run['metadata']['initial_model_hash']
                    if expected_initial is not None and expected_initial!=initial:
                        raise RuntimeError('Common initialization invariant failed')
                    expected_initial=initial
                    suite['runs'].append(dict(path=folder.name,variant=variant,seed=seed,status=run['status']))
                    write_json(output/'suite.json',suite)
                    print(variant,seed,run['status'],flush=True)
                    if args.command=='smoke':
                        evaluate_run(args.bundle,folder,folder/'validation_check','val',args.device)
            suite['status']='complete';write_json(output/'suite.json',suite)
        except BaseException as exc:
            suite['status']='failed';suite['error']=repr(exc);write_json(output/'suite.json',suite)
            raise
    elif args.command=='evaluate':
        r=evaluate_run(args.bundle,args.run,args.output,args.split,args.device)
        print(r['variant'],r['seed'],r['split'],r['metrics']['state']['macro_source_kl'])
    elif args.command=='evaluate-suite':
        suite=read_json(args.suite/'suite.json')
        if suite['status']!='complete':
            parser.error('Only a complete suite can be evaluated')
        if suite['command']=='smoke' and args.split=='test':
            parser.error('Smoke suite cannot evaluate real test data')
        output=new_directory(args.output)
        for run in suite['runs']:
            r=evaluate_run(args.bundle,args.suite/run['path'],output/run['path'],args.split,args.device)
            print(r['variant'],r['seed'],r['split'],flush=True)
    else:
        r=compare(args.evaluations,args.reference,args.output,cfg['evaluation']['bootstrap_resamples'],cfg['evaluation']['bootstrap_seed'])
        print('Compared',list(r['comparisons']),'training seeds',r['training_seeds'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
