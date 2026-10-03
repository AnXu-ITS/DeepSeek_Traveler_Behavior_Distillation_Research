"""Verify supplied summary arithmetic, without API calls or simulation."""
from pathlib import Path
import json, csv
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data_tables/revision20260925'
a=json.loads((DATA/'formal_analysis.json').read_text())
x=np.array([r['signed_minus_softkl'] for r in a['primary']])
assert len(x)==a['n']==30
boot=x[np.random.default_rng(20260925).integers(len(x),size=(10000,len(x)))].mean(1)
ci=np.quantile(boot,[.025,.975])
assert np.isclose(x.mean(),a['mean'],rtol=0,atol=1e-12)
assert np.allclose(ci,a['ci95'],rtol=0,atol=1e-12)
runs=json.loads((DATA/'physical_run_metrics.json').read_text())
assert len(runs)==40
assert all(r['n_initial']==r['completed']==r['return_completed']==1000 for r in runs)
assert all(r['assigned_pt']==r['routed_pt']==r['boarded_pt'] for r in runs)
idx={(r['model'],r['arm'],r['seed']):r for r in runs}
out=[]
for model in sorted({r['model'] for r in runs}):
    for arm in sorted({r['arm'] for r in runs}-{'baseline'}):
        diffs=[]
        for r in runs:
            if (r['model'],r['arm'])!=(model,arm): continue
            b=idx[model,'baseline',r['seed']]
            diffs.append([(r[k]-b[k])*100 for k in ['raw_pt','adjusted_pt']]+[(r[k]-b[k])/10 for k in ['assigned_pt','routed_pt','boarded_pt']])
        v=np.array(diffs); assert len(v)==5
        means=v.mean(0); increments=np.diff(means)
        assert np.isclose(means[0]+increments.sum(),means[-1])
        out.append(dict(model=model,arm=arm,predicted_pp=means[0],increments_pp=increments.tolist(),boarding_pp=means[-1],seed_sd_pp=v[:,-1].std(ddof=1)))
result=dict(scope='Retained summaries only; no acquisition, training, respondent bootstrap or network rerun.',
            personas=30,primary_mean=float(x.mean()),primary_ci95=ci.tolist(),runs=40,stages=out)
print(json.dumps(result,indent=2))
