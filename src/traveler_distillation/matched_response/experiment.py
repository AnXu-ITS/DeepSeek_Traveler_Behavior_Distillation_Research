"""Matched training: fixed full epochs, common validation selection, no test access."""
from __future__ import annotations

from collections import Counter
import copy
import importlib.metadata
import platform
import random
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
import torch.nn.functional as F

from ..accessibility.accessibility_features import S8FeatureExtractor, TravelerStudentS8
from ..student.dataset import collate_batch
from ..student.losses import kl_divergence, elasticity_direction_loss, elasticity_magnitude_loss
from .data import digest, file_hash, load_split, new_directory, read_json, write_json, write_rows
from .metrics import encode, predict, pair_scores, interaction_scores, response_mask, summarize

VARIANTS = {
    'soft_kl': dict(ce=0., signed_l1=0., signed_huber=0., direction=0., magnitude=0.),
    'ce_kl': dict(ce=1., signed_l1=0., signed_huber=0., direction=0., magnitude=0.),
    'signed_l1': dict(ce=0., signed_l1=1., signed_huber=0., direction=0., magnitude=0.),
    'signed_huber': dict(ce=0., signed_l1=0., signed_huber=1., direction=0., magnitude=0.),
    'direction': dict(ce=0., signed_l1=0., signed_huber=0., direction=1., magnitude=0.),
    'magnitude': dict(ce=0., signed_l1=0., signed_huber=0., direction=0., magnitude=1.),
    'direction_magnitude': dict(ce=0., signed_l1=0., signed_huber=0., direction=1., magnitude=1.),
}


def validate_config(cfg):
    if set(cfg) != {'protocol','data','student','training','objective','evaluation','smoke'}:
        raise ValueError('Unknown/missing top-level configuration field')
    if cfg['protocol']['initialization']!='random' or cfg['protocol']['selection']!='macro_source_kl':
        raise ValueError('Only neutral random initialization and common macro_source_kl selection supported')
    fields={
        'protocol':{'name','initialization','selection'},
        'data':{'legacy','joint','joint_config','mechanism','accessibility','accessibility_records','legacy_split','accessibility_split'},
        'student':{'arch_version','n_alt_num','global_hidden_dim','alternative_hidden_dim','cat_embedding_dim','mode_embedding_dim','scorer_hidden_dim','dropout'},
        'training':{'variants','seeds','epochs','batch_units','eval_batch_size','learning_rate','weight_decay','threads'},
        'evaluation':{'sign_thresholds','bootstrap_resamples','bootstrap_seed'},
        'smoke':{'epochs','units_per_source'},
    }
    for section,names in fields.items():
        if set(cfg[section])!=names:
            raise ValueError(f'Unknown/missing {section} field')
    if cfg['student']['n_alt_num']!=12 or cfg['student']['arch_version']!='student_s8_v1':
        raise ValueError('This protocol requires the 12-field S9-family architecture')
    t=cfg['training']; o=cfg['objective']; e=cfg['evaluation']
    if set(o)!={'departure_weight','departure_huber_delta','response_weight','response_mask','direction_eps','signed_huber_beta'}:
        raise ValueError('Unknown objective field: hidden relational loss is not allowed')
    if len(set(t['variants']))!=len(t['variants']) or not t['variants'] or set(t['variants'])-VARIANTS.keys():
        raise ValueError('Invalid variant list')
    if not t['seeds'] or len(set(t['seeds']))!=len(t['seeds']):
        raise ValueError('Training seeds must be nonempty and unique')
    if any(type(s) is not int or s<0 for s in t['seeds']):
        raise ValueError('Seeds must be non-negative integers')
    for key in ('epochs','batch_units','eval_batch_size','threads'):
        if not isinstance(t[key],int) or t[key]<1:
            raise ValueError(f'Invalid {key}')
    if o['response_mask'] not in ('union','perturbed','intersection'):
        raise ValueError('Invalid response mask')
    if any(o[k]<0 for k in ('departure_weight','response_weight')) or any(o[k]<=0 for k in ('departure_huber_delta','direction_eps','signed_huber_beta')):
        raise ValueError('Invalid loss parameters')
    if t['learning_rate']<=0 or t['weight_decay']<0 or e['bootstrap_resamples']<1:
        raise ValueError('Invalid optimization/evaluation parameters')
    if not e['sign_thresholds'] or any(x<=0 for x in e['sign_thresholds']):
        raise ValueError('Sign thresholds must be positive')
    if any(type(x) is not int or x<1 for x in cfg['smoke'].values()):
        raise ValueError('Invalid smoke budget')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_digest(model):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def code_fingerprint():
    root=Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*.py'))})


def epoch_order(n,seed,epoch):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch,9127])).permutation(n).tolist()


def loss_for_units(model, units, encoded, device, variant, objective):
    """All variants use the SAME forward rows, dropout draws and endpoint terms.

    Sum endpoint losses and response losses within a unit, mean over units.
    Singletons have no response term. No mechanism/persona/accessibility
    auxiliary updater is called. All supplied pair kinds use this one formula.
    """
    coefficients=VARIANTS[variant]
    ids=[eid for u in units for eid in u['ids']]
    batch={k:v.to(device) for k,v in collate_batch([encoded[i] for i in ids]).items()}
    out=model(batch);p=out['mode_probabilities'];t=batch['target_probs']
    kl=kl_divergence(t,p)
    ce=-torch.log(p.clamp_min(1e-8)).gather(1,batch['target_mode_idx'][:,None]).squeeze(1)
    dep=F.huber_loss(out['departure_time_shift_min'],batch['target_departure'],
                     delta=objective['departure_huber_delta'],reduction='none')
    base=[];cf=[];offset=0
    for unit in units:
        if unit['pair']:
            base.append(offset);cf.append(offset+1)
        offset+=len(unit['ids'])
    terms={k:p.sum()*0. for k in ('signed_l1','signed_huber','direction','magnitude')}
    if base:
        td=t[cf]-t[base];sd=p[cf]-p[base]
        mask=response_mask(batch['alt_mask'][base],batch['alt_mask'][cf],objective['response_mask'])
        denom=mask.sum(-1).clamp_min(1)
        terms['signed_l1']=(((td-sd).abs()*mask).sum(-1)/denom).sum()
        # SmoothL1 with beta: linear-tail scale comparable to signed L1.
        terms['signed_huber']=((F.smooth_l1_loss(sd,td,beta=objective['signed_huber_beta'],reduction='none')*mask).sum(-1)/denom).sum()
        terms['direction']=elasticity_direction_loss(td,sd,mask,margin=0.,eps=objective['direction_eps']).sum()
        terms['magnitude']=elasticity_magnitude_loss(td,sd,mask).sum()
    response=sum(coefficients[k]*v for k,v in terms.items())
    total=(kl.sum()+coefficients['ce']*ce.sum()+objective['departure_weight']*dep.sum()
           +objective['response_weight']*response)/len(units)
    return total, dict(kl=float(kl.sum().detach()),ce=float(ce.sum().detach()),
                       departure=float(dep.sum().detach()),
                       response=float(response.detach()),n_endpoints=len(ids),n_units=len(units))


def _encoded(split,ext):
    return {e['id']:encode(e,ext) for e in split['endpoints']}


def _smoke_subset(data,units_per_source):
    units=[]
    for source in sorted({u['source'] for u in data['units']}):
        units += [u for u in data['units'] if u['source']==source][:units_per_source]
    ids={i for u in units for i in u['ids']}
    return dict(units=units,endpoints=[e for e in data['endpoints'] if e['id'] in ids],
                pairs=[p for p in data['pairs'] if p['base'] in ids and p['cf'] in ids],
                interactions=[i for i in data['interactions'] if set(i['endpoints'])<=ids])


def train_run(bundle,cfg,output,variant,seed,device='cpu',smoke=False):
    validate_config(cfg)
    if variant not in VARIANTS or seed not in cfg['training']['seeds']:
        raise ValueError('Unknown variant/seed')
    train,manifest=load_split(bundle,'train');val,_=load_split(bundle,'val')
    if manifest.get('data_config')!=cfg['data']:
        raise ValueError('Config data sources differ from prepared bundle; prepare the intended bundle first')
    if smoke:
        train=_smoke_subset(train,cfg['smoke']['units_per_source'])
        val=_smoke_subset(val,cfg['smoke']['units_per_source'])
    ext=S8FeatureExtractor.from_state_dict(read_json(Path(bundle)/'extractor.json')['state'])
    train_encoded=_encoded(train,ext); val_encoded=_encoded(val,ext)
    t=cfg['training'];epochs=cfg['smoke']['epochs'] if smoke else t['epochs']
    torch.set_num_threads(t['threads'])
    torch.use_deterministic_algorithms(True)
    seed_all(seed)
    model=TravelerStudentS8(cfg['student'],ext.spec).to(device)
    initial_hash=model_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    code_id=code_fingerprint()
    comparison_cfg=copy.deepcopy(cfg)
    # Inventory of runs does not alter any individual scientific comparison.
    comparison_cfg['training'].pop('variants')
    comparison_cfg['training'].pop('seeds')
    protocol=digest(dict(config=comparison_cfg,bundle=manifest['bundle_id'],code=code_id,smoke=smoke,device=device))
    output=new_directory(output)
    metadata=dict(protocol_id=protocol,bundle_id=manifest['bundle_id'],variant=variant,seed=seed,
        smoke=smoke,device=device,initialization='random; no inherited checkpoint',
        initial_model_hash=initial_hash,extractor_hash=digest(ext.state_dict()),
        parameter_count=model.count_parameters(),config=cfg,code_id=code_id,
        loss_coefficients=VARIANTS[variant],selection='lowest validation macro_source_kl; earliest tie',
        training_budget=dict(epochs=epochs,units_per_epoch=len(train['units']),
                            endpoint_exposures_per_epoch=dict(Counter(i for u in train['units'] for i in u['ids']))),
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
            pydantic=importlib.metadata.version('pydantic'),pyyaml=importlib.metadata.version('PyYAML')))
    write_json(output/'run.json',metadata)
    write_json(output/'status.json',dict(status='running',smoke=smoke))
    repo=Path(__file__).resolve().parents[3]
    with ZipFile(output/'code_snapshot.zip','w',compression=ZIP_DEFLATED) as archive:
        for path in sorted((repo/'src/traveler_distillation').rglob('*.py')):
            archive.write(path,str(path.relative_to(repo)))
        archive.write(repo/'scripts/matched_response.py','scripts/matched_response.py')
    metadata['code_snapshot_sha256']=file_hash(output/'code_snapshot.zip')
    write_json(output/'run.json',metadata)
    torch.save(dict(model_state=model.state_dict(),config=cfg['student'],feature_spec=ext.spec,
                    extractor_state=ext.state_dict(),initial_model_hash=initial_hash),output/'initial.pt')
    history=[];best=float('inf');start=time.monotonic(); steps=0
    try:
        for epoch in range(epochs):
            order=epoch_order(len(train['units']),seed,epoch)
            write_rows(output/f'epoch_{epoch+1:03d}_units.jsonl',
                       [dict(unit=train['units'][i]['id']) for i in order])
            model.train();loss_sum=0.;seen=0
            for offset in range(0,len(order),t['batch_units']):
                units=[train['units'][i] for i in order[offset:offset+t['batch_units']]]
                optimizer.zero_grad(set_to_none=True)
                loss,parts=loss_for_units(model,units,train_encoded,device,variant,cfg['objective'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Non-finite gradient')
                optimizer.step();steps+=1
                loss_sum+=float(loss.detach())*len(units);seen+=len(units)
            predictions=predict(model,val['endpoints'],val_encoded,device,t['eval_batch_size'])
            metrics=summarize(predictions,[],[])
            score=metrics['state']['macro_source_kl']
            if score is None or not np.isfinite(score):
                raise FloatingPointError('Invalid validation criterion')
            record=dict(epoch=epoch+1,optimizer_steps=steps,train_loss=loss_sum/seen,
                        validation=metrics['state'],schedule_hash=digest(order))
            history.append(record)
            write_rows(output/'history.jsonl',history)
            if score<best:
                best=score
                checkpoint=dict(model_state=copy.deepcopy(model.state_dict()),config=cfg['student'],
                    feature_spec=ext.spec,extractor_state=ext.state_dict(),epoch=epoch+1,
                    selection_score=score,protocol_id=protocol,bundle_id=manifest['bundle_id'],
                    initial_model_hash=initial_hash,variant=variant,seed=seed,smoke=smoke)
                torch.save(checkpoint,output/'best.pt')
        torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                        epoch=epochs,protocol_id=protocol),output/'last_training_state.pt')
        final=dict(status='complete',smoke=smoke,epochs=epochs,optimizer_steps=steps,
                   best_epoch=checkpoint['epoch'],best_validation=best,
                   elapsed_s=time.monotonic()-start,best_checkpoint_sha256=file_hash(output/'best.pt'))
        write_json(output/'status.json',final)
        return dict(metadata=metadata,status=final)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='failed',smoke=smoke,error=repr(exc),
                   optimizer_steps=steps,completed_epochs=len(history)))
        raise


def evaluate_run(bundle,run,output,split,device='cpu'):
    run=Path(run);meta=read_json(run/'run.json');status=read_json(run/'status.json')
    if status['status']!='complete':
        raise ValueError('Training has not completed')
    if meta['smoke'] and split=='test':
        raise ValueError('Smoke checkpoints cannot evaluate the real test set')
    if meta['code_id']!=code_fingerprint():
        raise ValueError('Code differs from training; use the recorded code version')
    if file_hash(run/'best.pt')!=status['best_checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    data,manifest=load_split(bundle,split)
    if manifest['bundle_id']!=meta['bundle_id']:
        raise ValueError('Different evaluation bundle')
    if meta['smoke']:
        data=_smoke_subset(data,meta['config']['smoke']['units_per_source'])
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    ext=S8FeatureExtractor.from_state_dict(ck['extractor_state'])
    model=TravelerStudentS8(ck['config'],ck['feature_spec']).to(device)
    model.load_state_dict(ck['model_state']);torch.set_num_threads(meta['config']['training']['threads'])
    predictions=predict(model,data['endpoints'],_encoded(data,ext),device,
                        meta['config']['training']['eval_batch_size'])
    ps=pair_scores(predictions,data['pairs'],meta['config']['evaluation']['sign_thresholds'],
                   meta['config']['objective']['response_mask'])
    interactions=interaction_scores(predictions,data['interactions'])
    output=new_directory(output)
    write_rows(output/'predictions.jsonl',predictions)
    write_rows(output/'pairs.jsonl',ps)
    write_rows(output/'interactions.jsonl',interactions)
    report=dict(protocol_id=meta['protocol_id'],bundle_id=meta['bundle_id'],variant=meta['variant'],
                seed=meta['seed'],smoke=meta['smoke'],split=split,
                initial_model_hash=meta['initial_model_hash'],extractor_hash=meta['extractor_hash'],
                best_epoch=ck['epoch'],checkpoint_sha256=file_hash(run/'best.pt'),
                metrics=summarize(predictions,ps,interactions),
                raw_hashes={name:file_hash(output/f'{name}.jsonl') for name in ('predictions','pairs','interactions')})
    write_json(output/'metrics.json',report)
    return report
