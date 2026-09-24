"""Build a common, auditable endpoint and relation pool from existing labels.

Preparation may inspect split metadata/labels to materialize the split files.
Training only loads the train and validation files; test is a separate command.
No Teacher calls, checkpoint inheritance, or supply recomputation occur here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import yaml

from ..accessibility.accessibility_features import S8FeatureExtractor
from ..dataset.aggregation import AggregatedTeacherTarget
from ..schemas.state import UniversalTravelerState
from ..student.features import FeatureExtractor
from ..student.release_guard import assert_not_frozen_output

SPLITS = ('train', 'val', 'test')
MODES = ('car', 'pt', 'bike', 'walk')
ROLES = ('baseline', 'natural', 'broken', 'mediator')
RANK = {'A_excellent': 0, 'B_good': 1, 'C_moderate': 2, 'D_poor': 3, 'E_infeasible': 4}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_rows(path, values):
    with Path(path).open('w', encoding='utf-8') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def new_directory(path):
    path = assert_not_frozen_output(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def split_index(parts):
    out = {}
    for split in SPLITS:
        for pid in parts[split]:
            if pid in out:
                raise ValueError(f'Overlapping split identity: {pid}')
            out[pid] = split
    return out


def endpoint(eid, source, split, state, probs, departure, **meta):
    state = UniversalTravelerState.model_validate(state).model_dump(mode='json')
    alternatives = {a['mode']: a for a in state['alternatives']}
    if len(alternatives) != len(state['alternatives']) or set(alternatives) != set(MODES):
        raise ValueError(f'{eid}: matched protocol requires the four explicit alternatives')
    state['alternatives'] = [alternatives[m] for m in MODES]
    mask = [bool(alternatives[m]['available']) for m in MODES]
    raw = [float(probs.get(m, 0)) for m in MODES]
    if set(probs) - set(MODES) or min(raw) < 0 or not all(math.isfinite(p) for p in raw):
        raise ValueError(f'{eid}: invalid Teacher probabilities')
    if not any(mask) or any(p > 1e-5 for p, ok in zip(raw, mask) if not ok):
        raise ValueError(f'{eid}: invalid probability mass on unavailable modes')
    ps = [p if ok else 0. for p, ok in zip(raw, mask)]
    total = sum(ps)
    if abs(total - 1) > 1e-5:
        raise ValueError(f'{eid}: probabilities do not sum to one: {total}')
    ps = [p / total for p in ps]  # only six-decimal aggregation roundoff
    feature_state = json.loads(json.dumps(state))
    feature_state['persona'].pop('persona_id', None)
    feature_state['trip'].pop('trip_id', None)
    feature_state['context'].pop('context_id', None)
    return dict(id=eid, source=source, split=split, state=state, teacher_probs=ps,
                teacher_departure=float(departure), raw_probability_sum=total,
                persona=state['persona']['persona_id'], trip=state['trip']['trip_id'],
                input_hash=digest(feature_state), **meta)


def validate_pool(endpoints, pairs, interactions):
    by_id = {}
    personas, hashes = {}, {}
    for e in endpoints:
        if e['id'] in by_id:
            raise ValueError(f'Duplicate endpoint ID: {e["id"]}')
        if e['split'] not in SPLITS:
            raise ValueError('Unknown split')
        for identity, index in [(e['persona'], personas), (e['input_hash'], hashes)]:
            if identity in index and index[identity] != e['split']:
                raise ValueError(f'Cross-split persona or model-input overlap: {e["id"]}')
            index[identity] = e['split']
        by_id[e['id']] = e
    for relation in pairs + interactions:
        ids = relation.get('endpoints') or [relation['base'], relation['cf']]
        es = [by_id[i] for i in ids]
        if len({e['split'] for e in es}) != 1 or es[0]['split'] != relation['split']:
            raise ValueError(f'Cross-split relation: {relation["id"]}')
        if len({(e['persona'], e['trip']) for e in es}) != 1:
            raise ValueError(f'Relation changes persona/trip: {relation["id"]}')
    for split in SPLITS:
        if not any(e['split'] == split for e in endpoints):
            raise ValueError(f'Empty {split} split')


def make_units(endpoints, pairs):
    """Identical exposure for every variant: pair endpoints PLUS unpaired states.

    A shared baseline is repeated once per incident pair. This weighting is
    explicit, fixed, and common to pointwise and relational objectives.
    """
    units = [dict(id=p['id'], source=p['source'], ids=[p['base'], p['cf']],
                  pair=True, group=p['group']) for p in pairs]
    covered = {i for u in units for i in u['ids']}
    units += [dict(id='singleton:' + e['id'], source=e['source'], ids=[e['id']],
                   pair=False, group=e['persona'] + '::' + e['trip'])
              for e in endpoints if e['id'] not in covered]
    return sorted(units, key=lambda u: u['id'])


def collect(root, cfg):
    paths = {k: (Path(root) / v).resolve() for k, v in cfg['data'].items()}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    original_hashes={k:file_hash(p) for k,p in paths.items()}
    legacy = split_index(read_json(paths['legacy_split'])['personas'])
    acc_manifest = read_json(paths['accessibility_split'])
    acc_split = split_index(acc_manifest['persona_split'])
    if legacy != acc_split:
        raise ValueError('Legacy/accessibility persona partitions differ; audit before use')
    combos = yaml.safe_load(paths['joint_config'].read_text(encoding='utf-8'))['joint_combinations']
    combo_index = {tuple(sorted(c['axes'])): c for c in combos}
    ep, pairs, interactions, exclusions = [], [], [], []
    samples = {}
    for source in ('legacy', 'joint', 'accessibility'):
        samples[source] = []
        for raw in rows(paths[source]):
            s = AggregatedTeacherTarget.model_validate(raw)
            if s.status != 'complete':
                raise ValueError(f'Incomplete label: {s.sample_id}')
            pid = s.persona_group_id or s.state.persona.persona_id
            if pid != s.state.persona.persona_id or pid not in legacy:
                raise ValueError(f'Unmapped persona: {s.sample_id}')
            split = legacy[pid]
            bucket = source
            if source == 'joint':
                axes = tuple(sorted(a['axis'] for a in s.perturbation.joint_axes))
                if axes not in combo_index:
                    raise ValueError(f'Unknown joint combination: {axes}')
                seen = combo_index[axes]['seen_in_training']
                if not seen and split != 'test':
                    exclusions.append(dict(id='joint:' + s.sample_id, split=split,
                                           reason='unseen combination excluded from train and validation'))
                    continue
                bucket = 'joint_seen' if seen else 'joint_unseen'
            e = endpoint(source + ':' + s.sample_id, source, split, s.state.model_dump(),
                         s.teacher_aggregate.mode_probabilities,
                         s.teacher_aggregate.departure_time_shift_min,
                         bucket=bucket, original_id=s.sample_id, teacher_k=s.aggregation_metadata.k)
            ep.append(e)
            samples[source].append((s, e))
    index = {e['id']: e for e in ep}

    def add_pair(a, b, source, label):
        if a not in index or b not in index:
            raise ValueError(f'Missing relation endpoint: {a}, {b}')
        e = index[a]
        pairs.append(dict(id=f'{source}:{label}', base=a, cf=b, source=source,
                          split=e['split'], group=e['persona'] + '::' + e['trip'],
                          persona=e['persona'], bucket=index[b]['bucket']))

    for source in ('legacy', 'joint'):
        for s, e in samples[source]:
            if s.perturbation.axis == 'baseline':
                continue
            bid = 'legacy:' + str(s.baseline_sample_id)
            if bid not in index:
                raise ValueError(f'Missing baseline label for {e["id"]}: {bid}')
            add_pair(bid, e['id'], source, e['original_id'])
    rec = {r['sample_id']: r for r in rows(paths['accessibility_records'])}
    curves = defaultdict(list)
    for s, e in samples['accessibility']:
        r = rec[s.sample_id]
        if r['split'] != e['split'] or r['persona_id'] != e['persona']:
            raise ValueError(f'Accessibility metadata mismatch: {s.sample_id}')
        if r['od_index'] not in acc_manifest['od_split'][e['split']]:
            raise ValueError(f'OD holdout mismatch: {s.sample_id}')
        e.update(accessibility_class=r['accessibility_class'], od=r['od_index'],
                 curve_group=r['curve_group'])
        curves[r['curve_group']].append(e)
    for gid, members in sorted(curves.items()):
        members.sort(key=lambda e: RANK[e['accessibility_class']])
        # Corrected supply can map several states to the same class. Preserve
        # every endpoint; compare all members of adjacent DISTINCT classes.
        # These are supply-profile contrasts, NOT fixed-OD counterfactuals.
        classes=defaultdict(list)
        for e in members:
            classes[RANK[e['accessibility_class']]].append(e)
        ranks=sorted(classes)
        for low,high in zip(ranks,ranks[1:]):
            for a in classes[low]:
                for b in classes[high]:
                    add_pair(a['id'], b['id'], 'accessibility', a['id'] + '->' + b['id'])
                    pairs[-1].update(contrast_type='supply_profile',fixed_od=a['od']==b['od'])

    for q in rows(paths['mechanism']):
        pid, split = q['persona_id'], q['split']
        if legacy.get(pid) != split:
            raise ValueError(f'Mechanism split mismatch: {q["audit_group_id"]}')
        ids = {}
        for role in ROLES:
            m = q['members'][role]
            eid = 'mechanism:' + q['audit_group_id'] + ':' + role
            e = endpoint(eid, 'mechanism', split, m['state'], m['teacher_probs'],
                         m['teacher_departure'], bucket='mechanism', teacher_k=m['teacher_k'],
                         audit_group=q['audit_group_id'], role=role)
            ep.append(e)
            index[eid] = e
            ids[role] = eid
        for role in ROLES[1:]:
            add_pair(ids['baseline'], ids[role], 'mechanism', q['audit_group_id'] + ':' + role)

    singles = defaultdict(list)
    for s, e in samples['legacy']:
        singles[(e['persona'], e['trip'])].append((s, e))
    for s, e in samples['joint']:
        ids = ['legacy:' + s.baseline_sample_id]
        missing=[]
        for axis in s.perturbation.joint_axes:
            matches = [v['id'] for x, v in singles[(e['persona'], e['trip'])]
                       if x.perturbation.axis == axis['axis']
                       and abs(float(x.perturbation.level) - float(axis['level'])) < 1e-6]
            if len(matches)>1:
                raise ValueError(f'Ambiguous interaction single endpoint: {e["id"]}, {axis}')
            if not matches:
                missing.append(axis)
            ids += matches
        if missing:
            exclusions.append(dict(id='interaction:'+e['id'],split=e['split'],
                reason='missing single-axis Teacher endpoint; state and baseline-response pair retained',
                missing_axes=missing))
            continue
        if len(ids) != 3:
            raise ValueError('Interaction protocol requires exactly two axes')
        interactions.append(dict(id='interaction:' + e['id'], endpoints=ids + [e['id']],
                                 split=e['split'], persona=e['persona'], bucket=e['bucket'],
                                 group=e['persona'] + '::' + e['trip']))
    ep.sort(key=lambda e: e['id'])
    pairs.sort(key=lambda p: p['id'])
    validate_pool(ep, pairs, interactions)
    if original_hashes!={k:file_hash(p) for k,p in paths.items()}:
        raise ValueError('Raw input changed during preparation')
    return ep, pairs, interactions, exclusions, paths


def prepare(root, cfg, output):
    ep, pairs, interactions, exclusions, sources = collect(root, cfg)
    train = [e for e in ep if e['split'] == 'train']
    # One deterministic fit on TRAIN endpoints, never inherited from S7/S9.
    states = [UniversalTravelerState.model_validate(e['state']) for e in train]
    base = FeatureExtractor().fit(states)
    extractor = S8FeatureExtractor.from_s7_and_train(base.state_dict(), states)
    # from_s7_and_train here receives freshly fitted TRAIN stats, NOT a checkpoint.
    output = new_directory(output)
    manifest = dict(schema_version=1, source_hashes={k: file_hash(p) for k, p in sources.items()},
                    source_paths={k: str(p) for k, p in sources.items()}, data_config=cfg['data'],
                    exclusions=exclusions, splits={}, files={},
                    note='Existing historical holdouts; no claim of a newly blinded test set. Accessibility pairs compare adjacent distinct class profiles and may change OD; they are not fixed-OD interventions.',
                    target_policy='Available modes; normalize aggregation rounding only, tolerance 1e-5.')
    write_json(output/'extractor.json', dict(state=extractor.state_dict(), spec=extractor.spec))
    for split in SPLITS:
        es = [e for e in ep if e['split'] == split]
        ps = [p for p in pairs if p['split'] == split]
        it = [v for v in interactions if v['split'] == split]
        us = make_units(es, ps)
        for name, values in [('endpoints', es), ('pairs', ps), ('interactions', it), ('units', us)]:
            write_rows(output/f'{split}_{name}.jsonl', values)
        counts = Counter(i for u in us for i in u['ids'])
        manifest['splits'][split] = dict(endpoints=len(es), pairs=len(ps), units=len(us),
            interactions=len(it), personas=len({e['persona'] for e in es}),
            source_counts=dict(Counter(e['source'] for e in es)),
            endpoint_exposures_per_epoch=dict(sorted(counts.items())), unit_pool_hash=digest(us),
            unique_model_inputs=len({e['input_hash'] for e in es}))
    for p in sorted(output.iterdir()):
        manifest['files'][p.name] = file_hash(p)
    manifest['bundle_id'] = digest(manifest['files'])
    write_json(output/'manifest.json', manifest)
    return manifest


def load_split(bundle, split):
    if split not in SPLITS:
        raise ValueError(split)
    bundle = Path(bundle)
    manifest = read_json(bundle/'manifest.json')
    if manifest['bundle_id'] != digest(manifest['files']):
        raise ValueError('Bundle manifest identity mismatch')
    data = {}
    for kind in ('endpoints', 'pairs', 'interactions', 'units'):
        filename = f'{split}_{kind}.jsonl'
        if file_hash(bundle/filename) != manifest['files'][filename]:
            raise ValueError(f'Prepared input changed: {filename}')
        data[kind] = list(rows(bundle/filename))
    if file_hash(bundle/'extractor.json') != manifest['files']['extractor.json']:
        raise ValueError('Prepared extractor changed')
    return data, manifest
