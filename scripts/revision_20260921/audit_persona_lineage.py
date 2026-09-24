"""Bounded read-only audit of historical and controlled persona holdouts.

Writes a new evidence report only; does not modify data, weights or manuscript.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,sys
import torch
from prepare_survey import ROOT,sha,rows,readj,writej
from traveler_distillation.config import load_yaml
from traveler_distillation.generators import resolve_combination

DEST=ROOT/'outputs/revision_20260921/persona_lineage_audit'
SPLITS=['train','val','test']
def main():
    DEST.mkdir(exist_ok=True);sources=[]
    def j(name):p=ROOT/name;sources.append(p);return readj(p)
    def rr(name):p=ROOT/name;sources.append(p);return rows(p)
    legacy=j('outputs/student_v0_3_s3_c/split_manifest.json')
    s5=j('outputs/student_s5_joint_m2/split_manifest.json')
    mechm=j('data/student_s7_mechanism/split_manifest.json')
    access=j('releases/s9_supply_aware_v2/data_manifest/s9_split_manifest.json')
    current_access=j('data/singapore_accessibility/split_manifest.json')
    release_access=j('releases/s9_supply_aware_v2/data_manifest/s9_accessibility_dataset.json')
    release_s7=j('releases/s7_w3_generic_core_v1/data_manifest/data_manifest.json')
    psets={k:set(legacy['personas'][k]) for k in SPLITS}
    assert [len(psets[k]) for k in SPLITS]==[28,6,6]
    overlaps={a+'_'+b:len(psets[a]&psets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(overlaps.values())
    for other in [s5['personas'],mechm['personas'],access['persona_split'],current_access['persona_split']]:
        assert all(set(other[k])==psets[k] for k in SPLITS)
    verified_release_sources=[]
    for name,d in release_access['files'].items():
        p=ROOT/d['path'];sources.append(p);assert sha(p)==d['sha256'];verified_release_sources.append(str(p.relative_to(ROOT)))
    for d in release_s7['datasets']:
        p=ROOT/d['path'];sources.append(p);assert sha(p)==d['sha256'];verified_release_sources.append(str(p.relative_to(ROOT)))
        if 'k5_view' in d:
            p=ROOT/d['k5_view']['path'];sources.append(p);assert sha(p)==d['k5_view']['sha256']
    base=rr('data/student_v0_3_s3/aggregated_teacher_dataset.jsonl')
    joint=rr('data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl')
    qs=rr('data/student_s7_mechanism/quadruplets.jsonl')
    acc=rr('data/singapore_accessibility/states_with_teacher.jsonl')
    cfg=ROOT/'configs/joint_sampling.yaml';sources.append(cfg);combos=load_yaml(cfg)['joint_combinations']
    seen_joint=[r for r in joint if (resolve_combination(r['perturbation'].get('joint_axes'),combos) or {}).get('seen_in_training',True)]
    def pid(r):return r.get('persona_group_id') or r['state']['persona']['persona_id']
    source_sets={};counts={}
    for name,rs in [('legacy',base),('seen_joint',seen_joint),('accessibility',acc)]:
        source_sets[name]={};counts[name]={}
        for split in SPLITS:
            part=[r for r in rs if pid(r) in psets[split]]
            ids={pid(r) for r in part};actual={r['state']['persona']['persona_id'] for r in part}
            assert ids==actual==psets[split]
            source_sets[name][split]=ids;counts[name][split]=dict(states=len(part),personas=len(ids))
    source_sets['mechanism']={};counts['mechanism']={}
    for split in SPLITS:
        part=[r for r in qs if r['split']==split];ids={r['persona_id'] for r in part}
        assert ids<=psets[split]
        actual={m['state']['persona']['persona_id'] for r in part for m in r['members'].values()}
        assert ids==actual
        source_sets['mechanism'][split]=ids;counts['mechanism'][split]=dict(quadruplets=len(part),states=4*len(part),personas=len(ids))
    train_union=set().union(*(sets['train'] for sets in source_sets.values()))
    assert not train_union&psets['test']
    checkpoints={};chain=[('S3-C','outputs/student_v0_3_s3_c/checkpoints/best.pt',None),
      ('S5-M2','outputs/student_s5_joint_m2/checkpoints/best.pt','outputs/student_v0_3_s3_c/checkpoints/best.pt'),
      ('S7-W3','releases/s7_w3_generic_core_v1/checkpoint/model.pt','outputs/student_s5_joint_m2/checkpoints/best.pt'),
      ('S9','releases/s9_supply_aware_v2/checkpoint/model.pt','releases/s7_w3_generic_core_v1/checkpoint/model.pt')]
    for label,name,parent in chain:
        p=ROOT/name;sources.append(p);d=torch.load(p,map_location='cpu',weights_only=False)
        init=d.get('init_checkpoint')
        if parent is None:assert init is None
        else:
            q=Path(init);q=q if q.is_absolute() else ROOT/q
            assert q.resolve()==(ROOT/parent).resolve()
        checkpoints[label]=dict(checkpoint=name,sha256=sha(p),recorded_parent=parent)
    # Verify the S3-C entry point constructs weights afresh, rather than inferring
    # that fact solely from absence of an init_checkpoint metadata key.
    initcode=ROOT/'scripts/train_student_v0_2_c.py';sources.append(initcode)
    text=initcode.read_text(encoding='utf-8');assert 'torch.load(' not in text and 'model.load_state_dict(ckpt' not in text
    assert 'model = TravelerStudent(s_cfg, extractor.spec)' in text
    assert 'FeatureExtractor().fit([s.state for s in train])' in text
    for n in ['scripts/train_student_s5_joint.py','scripts/train_student_s7.py','scripts/train_student_s8.py','scripts/build_s7_mechanism_dataset.py','configs/student_v0_3_c.yaml','outputs/student_v0_3_s3_c/student_v0_2_c_report.md']:
        sources.append(ROOT/n)
    bundle=ROOT/'outputs/matched_response_v1/bundle';manifest=j('outputs/matched_response_v1/bundle/manifest.json')
    controlled={};controlled_sets={};odsets={};burden={}
    for split in SPLITS:
        rs=rr(f'outputs/matched_response_v1/bundle/{split}_endpoints.jsonl');ids={r['persona'] for r in rs};actual={r['state']['persona']['persona_id'] for r in rs}
        assert ids==actual==psets[split]
        assert Counter(r['source'] for r in rs)==manifest['splits'][split]['source_counts']
        endpoints={r['id']:r for r in rs};assert len(endpoints)==len(rs)
        unit_counts={}
        for kind in ['units','pairs','interactions']:
            records=rr(f'outputs/matched_response_v1/bundle/{split}_{kind}.jsonl')
            for r in records:
                if 'ids' in r:assert all(i in endpoints for i in r['ids'])
            unit_counts[kind]=len(records)
        controlled[split]=dict(n_endpoints=len(rs),n_personas=len(ids),source_counts=dict(Counter(r['source'] for r in rs)),**unit_counts)
        controlled_sets[split]=ids
        ar=[r for r in rs if r['source']=='accessibility'];odsets[split]={r['od'] for r in ar}
        assert odsets[split]==set(access['od_split'][split])
        pts=[next(a for a in r['state']['alternatives'] if a['mode']=='pt') for r in ar]
        feasible=[a for a in pts if a['pt_feasible']>0]
        walks=[a['access_time_min']+a['egress_time_min'] for a in feasible]
        burden[split]=dict(n_feasible=len(feasible),feasible_walk_burden_ge15=sum(a>=15 for a in walks),all_walk_burden_ge15=sum(a['access_time_min']+a['egress_time_min']>=15 for a in pts),min_feasible_walk_burden=min(walks),max_feasible_walk_burden=max(walks))
    od_overlaps={a+'_'+b:len(odsets[a]&odsets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(od_overlaps.values())
    for key,path in manifest['source_paths'].items():assert sha(Path(path))==manifest['source_hashes'][key]
    # Resolve the adjacent holdout wording using actual archived states.
    # The S8-only threshold statement survived in S9's manifest prose.
    archived_s8=rr('data/singapore_accessibility_s8_legacy/records.jsonl')
    raw_s9=rr('data/singapore_accessibility/records.jsonl');oldburden={}
    for split in SPLITS:
        old=[r['accessibility'] for r in archived_s8 if r['split']==split and r['accessibility']['pt_feasible']]
        walks=[a['access_time_min']+a['egress_time_min'] for a in old]
        oldburden[split]=dict(n_feasible=len(old),feasible_walk_burden_ge15=sum(a>=15 for a in walks),max_feasible_walk_burden=max(walks))
        new=[r['accessibility'] for r in raw_s9 if r['split']==split and r['accessibility']['pt_feasible']]
        assert len(new)==burden[split]['n_feasible']
        assert sum(a['access_time_min']+a['egress_time_min']>=15 for a in new)==burden[split]['feasible_walk_burden_ge15']
    result=dict(status='passed',claim='Test persona identities are excluded from parameter-fitting splits throughout the documented S3-C → S5-M2 → S7-W3 → S9 Student lineage, not only accessibility adaptation.',
      persona_counts={k:len(v) for k,v in psets.items()},persona_overlap_counts=overlaps,all_recorded_persona_manifests_equal=True,
      historical_pool_counts=counts,accessibility_test_personas_in_any_recorded_training_pool=len(train_union&psets['test']),checkpoint_lineage=checkpoints,
      S3C_initialization='Fresh TravelerStudent construction; extractor fitted only on training states; no checkpoint-loading call in recorded entry point',
      controlled=controlled,controlled_persona_sets_exactly_equal_historical_holdout=True,controlled_all_endpoint_persona_ids_match_declared_group=True,
      od_counts={k:len(v) for k,v in odsets.items()},od_overlap_counts=od_overlaps,walking_burden=burden,archived_S8_walking_burden=oldburden,
      required_claim_correction='Delete the claim that >=15 min walking burden occurs only in the S9 test partition. It held in archived S8 (0/0/7 qualifying states) but not corrected S9 (11/8/19). Frozen S9 metadata retained the old description; raw S9 states and controlled endpoint copies agree.',
      limits=['Evidence concerns recorded Student fitting/replay and persona identity, not absence of these examples from prior evaluation or research design decisions.',
              'Does not establish unseen persona attributes, a newly collected blind benchmark, or lack of exposure in remote Teacher pretraining.',
              'Checkpoint metadata names the parent artifacts; this audit checks the currently archived chain and matching source hashes, not an immutable log of every historical process.'],
      source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources))},code_sha256=sha(__file__))
    writej(DEST/'verification.json',result)
    report=['# Persona holdout and recorded Student lineage audit','',
      'The stronger, bounded claim is supported: the six accessibility-test persona identities are excluded from fitting in the recorded S3-C → S5-M2 → S7-W3 → S9 lineage. The holdout is not confined to accessibility adaptation. This does not imply a new blind test, absence from historical evaluation/development, or absence from the remote Teacher\'s pretraining.','',
      'Suggested main-text wording:','',
      '> Within the accessibility dataset, the training, validation and test OD pools are disjoint. The same six test persona identities are held out from parameter fitting throughout the documented S3-C, S5-M2, S7-W3 and S9 Student training/replay lineage; this is a reused research benchmark, not a newly collected blind test.','',
      '**Required adjacent correction:** Delete the current claim that access-plus-egress walking burden of at least 15 minutes occurs only in the S9 test partition. Directly inspected S9 states contain 11 such feasible-PT states in train, 8 in validation and 19 in test. The respective feasible-state denominators are 178, 39 and 39. Maximum feasible walking burdens are 24.047, 17.475 and 34.907 minutes. The raw S9 accessibility records and the controlled endpoint copies agree.','',
      'The threshold holdout statement is true of the archived S8 records (0/0/7 qualifying states in train/val/test), whose walking/cycling supply calculations were subsequently corrected. It is stale prose in the S9 manifest, not a valid property of the final encoded S9 dataset. This audit does not change either dataset or infer that the persona/OD partitions leaked.','',
      'Controlled comparison wording (unchanged claim, now checked against actual rows):','',
      '> The matched-response bundle contains 1,839 training, 365 validation and 437 test endpoints from mutually disjoint sets of 28, 6 and 6 persona identities, respectively, shared across all four data sources.','',
      'Evidence chain:','',
      '- S9 release accessibility split manifest, original S3-C split manifest, S5-M2 split manifest and mechanism split manifest have identical 28/6/6 identity sets and zero pairwise intersections.',
      '- Actual legacy, seen-joint and accessibility rows reproduce those identity sets; actual mechanism training quadruplets contain a subset of training personas only. No accessibility-test identity appears in any recorded training/replay source.',
      '- Checkpoint init_checkpoint metadata links S9 to frozen S7-W3, S7-W3 to S5-M2, and S5-M2 to S3-C. The S3-C entry point creates a new TravelerStudent and fits normalization only on train states (scripts/train_student_v0_2_c.py:305,315,352).',
      '- S5 filters merged inputs by the reused manifest (scripts/train_student_s5_joint.py:197–205); S7 filters legacy/joint and mechanism splits (scripts/train_student_s7.py:297–324); S9 uses the S8 entry point with split-filtered replay and accessibility pools (scripts/train_student_s8.py:428–474).',
      '- Frozen S7 and S9 data manifests match the current bulk-data SHA-256 values. Controlled endpoint rows and training-unit references were inspected directly, not inferred from a repeated prose statement.',
      '',
      '| Split | Controlled endpoints | Persona identities | Accessibility ODs | Feasible PT states with walk burden >=15 min |',
      '|---|---:|---:|---:|---:|']
    for split in SPLITS:report.append(f"| {split} | {controlled[split]['n_endpoints']} | {controlled[split]['n_personas']} | {len(odsets[split])} | {burden[split]['feasible_walk_burden_ge15']} |")
    report+=['','Detailed per-source counts, parent-checkpoint hashes, exact source hashes and caveats are in verification.json. This audit did not modify data, checkpoints or the manuscript.']
    (DEST/'PERSONA_LINEAGE_AUDIT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','persona_counts','persona_overlap_counts','historical_pool_counts','accessibility_test_personas_in_any_recorded_training_pool','controlled','od_counts','walking_burden']},ensure_ascii=False))

if __name__=='__main__':main()
