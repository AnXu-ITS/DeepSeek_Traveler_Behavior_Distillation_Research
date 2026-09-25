"""Input-only intensity coverage audit; no API results or outcome selection."""
from synthetic_teacher import DEST,CARDS,BUNDLE
from controlled import OUT,read_json,write_json,file_hash
from offline_audits import rows
import numpy as np

def features(state):
    pt=next(a for a in state['alternatives'] if a['mode']=='pt');ctx=state['context']
    return {'context_delay_min':ctx['transit_delay_min'],'fare_multiplier':ctx['fare_multiplier'],
        'pt_cost':pt['monetary_cost'],'pt_access_min':pt['access_time_min'],
        'pt_total_min':pt['travel_time_min'],'pt_wait_min':pt['wait_time_min'],
        'pt_feasible':pt['pt_feasible'],'weather_intensity':ctx['weather']['intensity']}

def main():
    states=rows(DEST/'states.jsonl');report=[];sources={}
    for family,bundle in [('full',BUNDLE),('delay_holdout',OUT.parent/'delay_family/bundle')]:
        pools={split:rows(bundle/f'{split}_endpoints.jsonl') for split in ['train','val']}
        for split,es in pools.items():sources[(bundle/f'{split}_endpoints.jsonl').as_posix()]=file_hash(bundle/f'{split}_endpoints.jsonl')
        for card in CARDS:
            ss=[features(s['state']) for s in states if s['card']==card]
            for feature in ss[0]:
                # Feature values are invariant across new personas in this design.
                values=sorted({s[feature] for s in ss});assert len(values)==1;x=values[0]
                vals=np.array([features(e['state'])[feature] for e in pools['train']]);val=np.array([features(e['state'])[feature] for e in pools['val']])
                exact=int(np.isclose(vals,x,rtol=0,atol=1e-8).sum())
                kind='exact marginal value seen' if exact else 'between training extrema' if vals.min()<=x<=vals.max() else 'outside training range'
                report.append(dict(family=family,card=card,feature=feature,value=x,training_min=float(vals.min()),training_max=float(vals.max()),training_exact_count=exact,validation_exact_count=int(np.isclose(val,x,rtol=0,atol=1e-8).sum()),marginal_support=kind))
    write_json(DEST/'intensity_support_audit.json',dict(source_sha256=sources,manifest_sha256=file_hash(DEST/'states.jsonl'),rows=report,
        interpretation='Marginal feature ranges only, not proof of joint-state or conditional support. Different levels alone do not establish intervention-disjoint training. Only the explicitly refitted delay-family holdout isolates an unseen family; full-data experiments test new personas and declared intensity sensitivities.'))
    print('Input-only support audit saved; '+str(len(report))+' feature-card-family checks')

if __name__=='__main__':main()
