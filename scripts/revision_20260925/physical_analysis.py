"""Independent per-person audit of finished four-arm execution ledgers."""
from physical_supply import *

def main():
    reconstruction=read_json(SUMMARY/'baseline_reconstruction_audit.json')
    assert reconstruction['n_exact']==1000 and reconstruction['same_semantics']
    inventory=read_json(SUMMARY/'run_inventory.json');assert len(inventory)==40
    ledgers={};summary=[];checks=[]
    supplyaudit=read_json(DEST/'supply/audit.json');kept=set(supplyaudit['kept_trip_ids']);removed=set(supplyaudit['removed_trip_ids'])
    for r in inventory:
        source=Path(r['source']);result=read_json(source/'result.json');ledger=rows(source/'person_ledger.jsonl');key=r['model'],r['arm'],r['seed'];ledgers[key]=ledger
        assert len(ledger)==1000 and len({x['person_id'] for x in ledger})==1000
        assert file_hash(source/'person_ledger.jsonl')==r['ledger_sha256']
        cfg=(source/'config.xml').read_text(encoding='utf-8')
        expected_schedule=BASESUPPLY['schedule'] if r['arm'] in ARMS[:2] else DEST/'supply/transitSchedule.xml'
        cfg_tree=ET.parse(source/'config.xml');params={p.get('name'):p.get('value') for m in cfg_tree.getroot().findall('module') if m.get('name')=='transit' for p in m.findall('param')}
        named=params.get('transitScheduleFile');assert named and Path(named).resolve()==expected_schedule.resolve(),(r,named)
        for x in ledger:
            raw=np.array([x['original_probability'][m] for m in old.MODES]);p=raw*np.array([x['planner_mask'].get(m,False) for m in old.MODES]);p/=p.sum()
            assert max(abs(p[j]-x['filtered_probability'][m]) for j,m in enumerate(old.MODES))<1e-8
            u=old.uniform(x['person_id'],r['seed']);assert abs(u-x['uniform'])<1e-15
            assert x['assigned_mode']==old.MODES[min(3,int(np.searchsorted(p.cumsum(),u)))]
        if r['arm'] in ARMS[2:]:
            trips={route[k] for x in ledger for route in [x['outbound_route'],x['return_route']] for k in ['trip_id','trip_id_2'] if route.get(k)}
            assert not trips&removed and trips<=kept
        vals=[x['events'] for x in ledger];times=[x['outbound_trip_min'] for x in vals if x['outbound_trip_min'] is not None]
        summary.append(dict(model=r['model'],arm=r['arm'],seed=r['seed'],n_initial=1000,raw_pt=np.mean([x['original_probability']['pt'] for x in ledger]),adjusted_pt=np.mean([x['filtered_probability']['pt'] for x in ledger]),assigned_pt=sum(x['assigned_mode']=='pt' for x in ledger),routed_pt=sum(x['routed_mode']=='pt' for x in ledger),boarded_pt=sum(x['outbound_pt_boardings']>0 for x in vals),completed=sum(x['outbound_arrived'] for x in vals),return_completed=sum(x['return_arrived'] for x in vals),completed_journey_mean_min=float(np.mean(times)) if times else None,n_time_observed=len(times)))
        checks.append(dict(model=r['model'],arm=r['arm'],seed=r['seed'],initial_n=True,policy_transformation=True,common_random_numbers=True,supply_path=True,scheduled_trip_ids=True,reused=r['reused']))
    contrasts=[]
    for model in MODELS:
        for base,arm in [('baseline',a) for a in ARMS[1:]]+[('frozen_disrupted','recomputed_disrupted')]:
            samples=[]
            for seed in SEEDS:
                b=ledgers[model,base,seed];d=ledgers[model,arm,seed];assert [x['person_id'] for x in b]==[x['person_id'] for x in d]
                out=[]
                for x,y in zip(b,d):
                    ex,ey=x['events'],y['events']
                    if arm=='frozen_disrupted':
                        assert max(abs(x['original_probability'][m]-y['original_probability'][m]) for m in old.MODES)<1e-6 and abs(x['departure_shift_min']-y['departure_shift_min'])<1e-5
                    horizon_x=ex['outbound_time_to_completion_or_horizon_min'];horizon_y=ey['outbound_time_to_completion_or_horizon_min']
                    assert horizon_x is not None and horizon_y is not None,'Do not silently drop non-departed people'
                    out.append([100*(y['original_probability']['pt']-x['original_probability']['pt']),100*((ey['outbound_pt_boardings']>0)-(ex['outbound_pt_boardings']>0)),100*(int(ey['outbound_arrived'])-int(ex['outbound_arrived'])),horizon_y-horizon_x])
                samples.append(out)
            arr=np.array(samples);avg=arr.mean(0);rng=np.random.default_rng(20260925);ix=rng.integers(1000,size=(4000,1000));boot=avg[ix].mean(1)
            contrasts.append(dict(model=model,baseline=base,arm=arm,n_initial=1000,paired_seeds=5,metrics={k:dict(mean=float(avg[:,j].mean()),ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),seed_sd=float(arr[:,:,j].mean(1).std(ddof=1))) for j,k in enumerate(['raw_pt_pp','boarded_pt_pp','completion_pp','completion_or_horizon_minutes'])}))
    write_json(SUMMARY/'run_metrics.json',summary);write_json(SUMMARY/'verified_contrasts.json',contrasts)
    inputs=[*BASESUPPLY.values(),old.checkpoint('s9'),RAW/'outputs/reviewer_closure_20260920/baseline/model.npz',RAW/'outputs/revision_20260921/baselines/train/independent_neural_seed42/best_departure.pt']
    write_json(SUMMARY/'verification.json',dict(expected=40,verified=len(checks),checks=checks,input_sha256={str(p):file_hash(p) for p in inputs},probability_reuse_tolerance=1e-6,departure_reuse_tolerance_min=1e-5,event_hashes={r['source']:read_json(Path(r['source'])/'result.json')['event_sha256'] for r in inventory},scope='Event hashes were produced by the simulation runner on complete parsed events; this audit recomputes ledger hashes and policy/paired/supply invariants. Horizon time is a descriptive administrative failure penalty, not mean survival time.'))
    print('Verified 40 runs, all paired assignments and supply paths; time and adaptation contrasts saved',flush=True)

if __name__=='__main__':main()
