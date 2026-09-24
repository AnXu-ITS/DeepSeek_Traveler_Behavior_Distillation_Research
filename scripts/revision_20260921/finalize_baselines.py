"""Read-only checks and final tables for executed AIT baselines and loss audits."""
from pathlib import Path
import sys,copy,itertools
sys.path.insert(0,str(Path(__file__).resolve().parent))
from departure_baselines import *
from traveler_distillation.matched_response.metrics import paired_cluster_difference

def main():
    torch.set_num_threads(2)
    kinds=['bounded_linear','independent_neural']+MAIN_VARIANTS
    checks=[];source_rows=[];old_hashes={};timeline=[]
    for kind,seed in itertools.product(kinds,SEEDS):
        folder=OUT/f'train/{kind}_seed{seed}';st=read(folder/'status.json');ck=torch.load(folder/'best_departure.pt',weights_only=False,map_location='cpu');hist=rows(folder/'history.jsonl')
        assert st['status']=='complete' and len(hist)==120 and st['optimizer_steps']==6360
        assert ck['epoch']==min(hist,key=lambda r:r['validation_departure_mae'])['epoch']
        pred=rows(OUT/f'test/{kind}_seed{seed}/predictions.jsonl');assert len(pred)==437
        assert all(abs(r['student_departure'])<=60 for r in pred)
        if kind not in MAIN_VARIANTS:
            orig=rows(ROOT/'outputs/reviewer_closure_20260920/baseline/predictions.jsonl');assert all(p['student']==o['student'] for p,o in zip(pred,orig))
        m=read(OUT/f'test/{kind}_seed{seed}/metrics.json')
        for source,s in m['state']['by_source'].items():
            source_rows.append(dict(model=kind,seed=seed,source=source,n=s['n'],departure_mae=s['departure_mae'],departure_median=s['departure_median'],departure_p90=s['departure_p90'],kl=s['kl']))
        timeline.append(dict(model=kind,seed=seed,selected_epoch=ck['epoch'],val_departure_mae=ck['selection_score'],seconds=st['elapsed_s'],kl_replay_max_weight_difference=st.get('kl_replay',{}).get('max_absolute_weight_difference')))
        checks.append(dict(model=kind,seed=seed,training_complete=True,selection_matches_validation_minimum=True,outputs_bounded=True,n_test_predictions=len(pred),checkpoint_sha256=sha(folder/'best_departure.pt')))
    for kind,seed in itertools.product(MAIN_VARIANTS,SEEDS):
        p=OLD/f'train/{kind}_seed{seed}/best.pt';old_hashes[str(p.relative_to(ROOT))]=sha(p);assert sha(p)==read(p.parent/'status.json')['best_checkpoint_sha256']
    csvout(OUT/'departure_by_source.csv',source_rows);csvout(OUT/'checkpoint_selection.csv',timeline)
    paired={}
    def avg(kind,field,original=False):
        paths=[(OLD if original else OUT)/f'test/{kind}_seed{s}/predictions.jsonl' for s in SEEDS]
        rs=[rows(p) for p in paths];by=[{r['id']:r for r in z} for z in rs]
        return [dict(r,**{field:float(np.mean([b[r['id']][field] for b in by]))}) for r in rs[0]]
    for kind in MAIN_VARIANTS:
        paired[kind]={}
        for original in [True,False]:
            paired[kind]['original_KL_selection' if original else 'new_MAE_selection']=paired_cluster_difference(avg('independent_neural','departure_mae'),avg(kind,'departure_mae',original),'departure_mae',10000,917)
    write(OUT/'hybrid_minus_neural_departure.json',paired)
    write(OUT/'verification.json',dict(status='passed',runs=checks,original_KL_checkpoint_hashes_unchanged=old_hashes,all_model_choice_predictions_identical_to_frozen_MNL_S=True,source_preservation='Original raw data and all original checkpoints read-only; only revision paths written.',runtime_note='MAE-selection reruns retain model initializations, data, exposure schedule, epochs and optimizer settings; two CPU threads used instead of original four. KL replay weight differences are explicitly recorded, so reruns are not represented as bitwise recovery of original trajectories.'))
    print('VERIFIED 18 completed runs, matching validation selection, 437 predictions each, six identical MNL choice branches; original 12 checkpoints unchanged.')
if __name__=='__main__':main()
