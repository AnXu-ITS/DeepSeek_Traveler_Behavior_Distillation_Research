"""Transparent endpoint/pair metrics; raw units retained for paired analysis."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch

from ..student.dataset import collate_batch
from .data import MODES


def encode(e, extractor):
    from ..schemas.state import UniversalTravelerState
    f = extractor.encode(UniversalTravelerState.model_validate(e['state']))
    return dict(global_cat=torch.tensor(f['global_cat'], dtype=torch.long),
                global_num=torch.tensor(f['global_num'], dtype=torch.float32),
                alt_mode_idx=torch.tensor(f['alt_mode_idx'], dtype=torch.long),
                alt_num=torch.tensor(f['alt_num'], dtype=torch.float32),
                alt_mask=torch.tensor(f['alt_available'], dtype=torch.float32),
                target_probs=torch.tensor(e['teacher_probs'], dtype=torch.float32),
                target_mode_idx=torch.tensor(np.argmax(e['teacher_probs']), dtype=torch.long),
                target_departure=torch.tensor(e['teacher_departure'], dtype=torch.float32))


def response_mask(a, b, rule):
    if rule == 'union':
        return (a + b).clamp(0, 1)
    if rule == 'intersection':
        return a * b
    if rule == 'perturbed':
        return b
    raise ValueError(f'Unknown mask rule: {rule}')


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None


@torch.no_grad()
def predict(model, endpoints, encoded, device, batch_size):
    model.eval()
    result = []
    for start in range(0, len(endpoints), batch_size):
        es = endpoints[start:start+batch_size]
        b = {k: v.to(device) for k, v in collate_batch([encoded[e['id']] for e in es]).items()}
        out = model(b)
        for e, p, dep in zip(es, out['mode_probabilities'].cpu().tolist(),
                             out['departure_time_shift_min'].cpu().tolist()):
            t = e['teacher_probs']
            if not all(math.isfinite(x) for x in p + [dep]):
                raise ValueError('Non-finite model prediction')
            kl = sum(x*math.log(x/max(y, 1e-8)) for x,y in zip(t,p) if x>0)
            active = [a['available'] for a in e['state']['alternatives']]
            inf = e.get('accessibility_class') == 'E_infeasible'
            result.append(dict(id=e['id'], persona=e['persona'], trip=e['trip'],
                input_hash=e['input_hash'], source=e['source'], bucket=e['bucket'],
                split=e['split'], teacher=t, student=p, mask=active,
                teacher_departure=e['teacher_departure'], student_departure=dep,
                kl=kl, probability_l1=sum(abs(x-y) for x,y in zip(t,p)),
                mode_accuracy=float(np.argmax(t)==np.argmax(p)),
                departure_mae=abs(dep-e['teacher_departure']),
                pt_mae=abs(t[1]-p[1]), fvr=float(np.argmax(p)==1) if inf else None,
                infeasible_pt_mass=p[1] if inf else None))
    return result


def pair_scores(predictions, pairs, thresholds, mask_rule):
    by_id = {r['id']: r for r in predictions}
    out = []
    for pair in pairs:
        a,b = by_id[pair['base']],by_id[pair['cf']]
        t = np.array(b['teacher']) - np.array(a['teacher'])
        s = np.array(b['student']) - np.array(a['student'])
        mask = response_mask(torch.tensor(a['mask']).float(), torch.tensor(b['mask']).float(), mask_rule).numpy().astype(bool)
        if not mask.any():
            raise ValueError('No comparable modes in a response pair')
        result = dict(pair, teacher_delta=t.tolist(), student_delta=s.tolist(),
                      response_gap=float(np.abs(t-s)[mask].mean()), n_available=int(mask.sum()))
        result['thresholds'] = {}
        for eps in thresholds:
            informative = mask & (np.abs(t) >= eps)
            quiet = mask & ~informative
            agree = informative & (np.sign(t)==np.sign(s))
            result['thresholds'][str(eps)] = dict(
                informative_n=int(informative.sum()), agreeing_n=int(agree.sum()),
                quiet_n=int(quiet.sum()), quiet_abs_sum=float(np.abs(s)[quiet].sum()),
                informative_sign=float(agree.sum()/informative.sum()) if informative.any() else None,
                informative_fraction=float(informative.sum()/mask.sum()),
                false_response=float(np.abs(s)[quiet].mean()) if quiet.any() else None)
        out.append(result)
    return out


def interaction_scores(predictions, interactions):
    by_id = {r['id']: r for r in predictions}
    result = []
    for item in interactions:
        a,i,j,ij = [by_id[e] for e in item['endpoints']]
        t = np.array(ij['teacher'])-np.array(i['teacher'])-np.array(j['teacher'])+np.array(a['teacher'])
        s = np.array(ij['student'])-np.array(i['student'])-np.array(j['student'])+np.array(a['student'])
        mask = np.any([r['mask'] for r in (a,i,j,ij)],axis=0)
        result.append(dict(item, interaction_gap=float(np.abs(t-s)[mask].mean()),
                           teacher_interaction=t.tolist(), student_interaction=s.tolist()))
    return result


def summarize(predictions, pair_rows, interaction_rows):
    def state_summary(rs):
        return dict(n=len(rs), personas=len({r['persona'] for r in rs}),
                    **{key:mean(r[key] for r in rs) for key in
                       ('kl','probability_l1','mode_accuracy','departure_mae','pt_mae','fvr','infeasible_pt_mass')},
                    n_infeasible=sum(r['fvr'] is not None for r in rs),
                    departure_p90=float(np.quantile([r['departure_mae'] for r in rs],.9)) if rs else None)
    sources = sorted({r['source'] for r in predictions})
    state = state_summary(predictions)
    state['by_source'] = {s:state_summary([r for r in predictions if r['source']==s]) for s in sources}
    state['macro_source_kl'] = mean(v['kl'] for v in state['by_source'].values())
    response = dict(n=len(pair_rows), personas=len({r['persona'] for r in pair_rows}),
                    groups=len({r['group'] for r in pair_rows}),
                    response_gap=mean(r['response_gap'] for r in pair_rows), thresholds={})
    for eps in (pair_rows[0]['thresholds'] if pair_rows else []):
        vals=[r['thresholds'][eps] for r in pair_rows]
        ni=sum(v['informative_n'] for v in vals); nq=sum(v['quiet_n'] for v in vals)
        response['thresholds'][eps] = dict(informative_n=ni, quiet_n=nq,
            informative_sign=sum(v['agreeing_n'] for v in vals)/ni if ni else None,
            informative_fraction=ni/(ni+nq) if ni+nq else None,
            false_response=sum(v['quiet_abs_sum'] for v in vals)/nq if nq else None)
    buckets = sorted({r['bucket'] for r in predictions})
    return dict(state=state, response=response,
        interaction=dict(n=len(interaction_rows), error=mean(r['interaction_gap'] for r in interaction_rows)),
        by_bucket={b:dict(state=state_summary([r for r in predictions if r['bucket']==b]),
            pairs=sum(r['bucket']==b for r in pair_rows),
            response_gap=mean(r['response_gap'] for r in pair_rows if r['bucket']==b),
            interaction_gap=mean(r['interaction_gap'] for r in interaction_rows if r['bucket']==b)) for b in buckets})


def paired_cluster_difference(left, right, key, n_boot, seed, id_key='id'):
    """State/pair-weighted difference, resampling whole personas jointly.

    Conditional on these training seeds and observed tasks. This does not
    estimate uncertainty across new training runs or new OD populations.
    """
    a={r[id_key]:r for r in left}; b={r[id_key]:r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):
        raise ValueError('Paired comparison requires unique identical unit IDs')
    groups=defaultdict(list)
    for rid in sorted(a):
        x,y=a[rid],b[rid]
        if x['persona']!=y['persona']:
            raise ValueError('Mismatched paired persona')
        if (x[key] is None)!=(y[key] is None):
            raise ValueError('Mismatched metric denominator')
        if x[key] is not None:
            groups[x['persona']].append(x[key]-y[key])
    if not groups:
        return dict(mean=None, ci=None, n=0, n_personas=0)
    vals=list(groups.values()); sums=np.array([sum(v) for v in vals]); ns=np.array([len(v) for v in vals])
    result=dict(mean=float(sums.sum()/ns.sum()),n=int(ns.sum()),n_personas=len(vals),
                bootstrap_unit='persona, all associated tasks retained',
                uncertainty='conditional on fixed training seeds and existing tasks',
                small_cluster_warning=len(vals)<20)
    if len(vals)<2:
        result['ci']=None
        return result
    rng=np.random.default_rng(seed)
    idx=rng.integers(len(vals),size=(n_boot,len(vals)))
    distribution=sums[idx].sum(axis=1)/ns[idx].sum(axis=1)
    result['ci']=np.quantile(distribution,[.025,.975]).tolist()
    return result
