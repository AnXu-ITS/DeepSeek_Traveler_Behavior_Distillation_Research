"""Verify that the LOS rebuild uses the historical baseline feature semantics."""
from physical_supply import *

def main():
    old.OUT=DEST/'baseline_reconstruction_audit';torch.set_num_threads(2)
    ex=Experiment(1000);baseline=rows(RAW/'outputs/revision_20260921/execution/states/C0_baseline.jsonl');details=[]
    for i,r in enumerate(baseline):
        st=UniversalTravelerState.model_validate(r['state'])
        acc=plan_accessibility(ex.idx,r['origin_node'],r['destination_node'],st.trip.desired_departure_min*60)
        alternatives=build_real_alternatives(st.persona,st.trip,st.context,ex.idx,r['origin_node'],r['destination_node'],acc)
        orig={a.mode:a.model_dump(mode='json') for a in st.alternatives};new={a.mode:a.model_dump(mode='json') for a in alternatives}
        diffs={}
        for mode in orig:
            for key,value in orig[mode].items():
                if isinstance(value,(int,float)) and not isinstance(value,bool):
                    if abs(value-new[mode][key])>1e-6:diffs[mode+'.'+key]=dict(historical=value,recomputed=new[mode][key])
                elif value!=new[mode][key]:diffs[mode+'.'+key]=dict(historical=value,recomputed=new[mode][key])
        details.append(dict(person_id=r['person_id'],differences=diffs))
        if (i+1)%200==0:print('baseline reconstruction',i+1,flush=True)
    report=dict(n=1000,n_exact=sum(not r['differences'] for r in details),mismatch_fields=dict(collections.Counter(k for r in details for k in r['differences'])),same_semantics=all(not r['differences'] for r in details),details=details)
    write_json(SUMMARY/'baseline_reconstruction_audit.json',report)
    print({k:v for k,v in report.items() if k!='details'},flush=True)

if __name__=='__main__':main()
