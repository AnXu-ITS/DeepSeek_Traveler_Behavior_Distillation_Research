"""Compare identical evaluation units and separate seed spread from test CI."""
from collections import defaultdict
from pathlib import Path
import numpy as np

from .data import read_json, rows, write_json, new_directory, file_hash
from .metrics import paired_cluster_difference


def compare(evaluations, reference, output, n_boot=10000, seed=917):
    groups=defaultdict(dict)
    protocol=None;split=None;initial={};smoke=None
    for folder in evaluations:
        folder=Path(folder);meta=read_json(folder/'metrics.json')
        if protocol is None:
            protocol=meta['protocol_id'];split=meta['split'];smoke=meta['smoke']
        if (meta['protocol_id'],meta['split'],meta['smoke'])!=(protocol,split,smoke):
            raise ValueError('Cannot compare different protocols, splits, or smoke/formal runs')
        variant=meta['variant'];s=meta['seed']
        if s in groups[variant]:
            raise ValueError('Duplicate variant/seed')
        if s in initial and initial[s]!=meta['initial_model_hash']:
            raise ValueError('Variants did not start from identical weights for this seed')
        initial[s]=meta['initial_model_hash']
        datasets={}
        for name in ('predictions','pairs','interactions'):
            if file_hash(folder/f'{name}.jsonl')!=meta['raw_hashes'][name]:
                raise ValueError('Evaluation rows changed after scoring')
            datasets[name]=list(rows(folder/f'{name}.jsonl'))
        groups[variant][s]=datasets
    if reference not in groups or len(groups)<2:
        raise ValueError('Need the reference and at least one comparator')
    seeds=set(groups[reference])
    if any(set(g)!=seeds for g in groups.values()):
        raise ValueError('Variants must have exactly the same training seeds')
    result=dict(reference=reference,protocol_id=protocol,split=split,smoke=smoke,
        training_seeds=sorted(seeds),n_training_seeds=len(seeds),comparisons={},
        bootstrap=dict(resamples=n_boot,seed=seed,unit='persona',method='percentile'),
        interpretation='Mean metric differences across matched training seeds. CI resamples personas after averaging per-unit metric differences across these fixed seeds; seed SD is separate. Not a joint seed-and-population CI.')
    targets={'predictions':['kl','probability_l1','mode_accuracy','departure_mae','pt_mae','fvr','infeasible_pt_mass'],
             'pairs':['response_gap'],'interactions':['interaction_gap']}
    def average_rows(variant,kind,key):
        seedrows=[{r['id']:r for r in groups[variant][s][kind]} for s in sorted(seeds)]
        ids=set(seedrows[0])
        if any(set(r)!=ids for r in seedrows):
            raise ValueError('Unit identities changed across seeds')
        averaged=[]
        for rid in sorted(ids):
            rs=[r[rid] for r in seedrows]
            if any(r['persona']!=rs[0]['persona'] for r in rs):
                raise ValueError('Persona identity changed across seeds')
            xs=[r[key] for r in rs]
            if any(x is None for x in xs) and not all(x is None for x in xs):
                raise ValueError('Metric denominator changed across seeds')
            averaged.append(dict(id=rid,persona=rs[0]['persona'],**{key:None if xs[0] is None else float(np.mean(xs))}))
        return averaged
    for variant in sorted(set(groups)-{reference}):
        output_metrics={}
        for kind,keys in targets.items():
            for key in keys:
                a=average_rows(variant,kind,key);b=average_rows(reference,kind,key)
                comparison=paired_cluster_difference(a,b,key,n_boot,seed)
                differences={}
                for s in sorted(seeds):
                    one=paired_cluster_difference(groups[variant][s][kind],groups[reference][s][kind],key,1,seed)
                    differences[str(s)]=one['mean']
                numbers=[x for x in differences.values() if x is not None]
                comparison['per_training_seed']=differences
                comparison['training_seed_sd']=float(np.std(numbers,ddof=1)) if len(numbers)>1 else None
                output_metrics[kind+'.'+key]=comparison
        result['comparisons'][variant+' minus '+reference]=output_metrics
    output=new_directory(output)
    write_json(output/'comparison.json',result)
    return result
