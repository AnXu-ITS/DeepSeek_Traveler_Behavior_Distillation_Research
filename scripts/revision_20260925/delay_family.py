"""Strict annotated PT-delay-family removal before fitting six new Students.

The historical test is a retrospective diagnostic. Fresh synthetic reference
evaluation is reported separately if the provider is available.
"""
import copy,sys
import controlled as c
from offline_audits import rows
from traveler_distillation.matched_response.data import make_units
from traveler_distillation.student.features import FeatureExtractor
from traveler_distillation.schemas.state import UniversalTravelerState

def main():
    original=c.BUNDLE;dest=c.OUT.parent/'delay_family';bundle=dest/'bundle';bundle.mkdir(parents=True,exist_ok=True)
    if not (bundle/'manifest.json').exists():
        exclusions=[];counts={};allparts={}
        # Mechanism delay quadruplets include broken/mediator endpoints with
        # zero context delay, so remove the whole annotated group as well.
        for split in ['train','val','test']:
            data,_=c.load_split(original,split)
            def belongs(e):return e['state']['context']['transit_delay_min']>0 or ('delay' in e.get('audit_group','').lower())
            reject={e['id'] for e in data['endpoints'] if belongs(e)} if split!='test' else set()
            es=[e for e in data['endpoints'] if e['id'] not in reject];ids={e['id'] for e in es}
            ps=[p for p in data['pairs'] if p['base'] in ids and p['cf'] in ids];ints=[r for r in data['interactions'] if set(r['endpoints'])<=ids];us=make_units(es,ps)
            assert split=='test' or not any(belongs(e) for e in es)
            exclusions.extend(dict(split=split,id=i) for i in sorted(reject));counts[split]=dict(endpoints=len(es),pairs=len(ps),units=len(us),removed=len(reject));allparts[split]=es
            for kind,rs in [('endpoints',es),('pairs',ps),('interactions',ints),('units',us)]:c.write_rows(bundle/f'{split}_{kind}.jsonl',rs)
        states=[UniversalTravelerState.model_validate(e['state']) for e in allparts['train']]
        ext=c.S8FeatureExtractor.from_s7_and_train(FeatureExtractor().fit(states).state_dict(),states)
        c.write_json(bundle/'extractor.json',dict(state=ext.state_dict(),spec=ext.spec))
        files={p.name:c.file_hash(p) for p in bundle.iterdir() if p.is_file()}
        c.write_json(bundle/'manifest.json',dict(bundle_id=c.digest(files),files=files,exclusions=exclusions,splits=counts,note='Delay intervention labels and entire delay mechanism quadruplets removed from train and val; extractor freshly fitted on retained train inputs; test unchanged and retrospective. No pretrained checkpoint or auxiliary losses.'))
        c.write_json(dest/'protocol.json',dict(family='positive context transit delay plus every delay mechanism quadruplet',variants=['soft_kl','signed_l1'],weight=1.,seeds=[42,2026,7],epochs=120,selection=['static','response'],source_bundle_sha256=c.file_hash(original/'manifest.json'),driver_sha256=c.file_hash(__file__),test_is_new=False))
    c.BUNDLE=bundle;c.OUT=dest/'train';c.GRID=[('soft_kl',0.),('signed_l1',1.)]
    for v,w in c.GRID:
        for seed in c.SEEDS:c.train(v,w,seed)
    c.evaluate()

if __name__=='__main__':main()
