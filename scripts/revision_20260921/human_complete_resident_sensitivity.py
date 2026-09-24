"""Human-only nested-cohort sensitivity; no model mapping or model inference.

Resample the 333 complete respondents eligible by recent Shanghai travel once per bootstrap draw and use
that draw for both the full mean and its original 321-person subset mean.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from collections import Counter
import csv,json,hashlib
import numpy as np
from prepare_survey import ROOT,OUT,ORDER,MODES,readj,sha,writej

DEST=OUT/'human_complete_resident_sensitivity'
PAPER=ROOT.parent/'蒸馏出行意图paper'
NAMES={'rain':'Rain','delay':'PT delay','rain_delay_interaction':r'Rain $\times$ delay','fare':'PT fare','parking':'Parking','road':'Road penalty','walk_vs_wait':'Walk versus wait','transfer_vs_wait':'Transfer versus wait'}

def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def csvout(p,data):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    DEST.mkdir(exist_ok=True)
    sources=[ROOT/'outputs/shanghai_sp_v1/sp_responses.csv',ROOT/'outputs/shanghai_sp_v1/input_coverage.csv',OUT/'sample_flow_audit/shanghai_excluded_record_flow.csv',OUT/'sample_flow_audit/sample_flow_audit.json',OUT/'contrasts.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    answers=readcsv(sources[0]);coverage=readcsv(sources[1]);excluded=readcsv(sources[2]);by={(r['respondent_id'],r['card_id']):r for r in answers}
    selected=set(r['respondent_id'] for r in coverage if r['input_status']=='complete')
    reason={r['respondent_id']:r['exclusion_category'] for r in excluded}
    extra=set(r for r,c in reason.items() if c in ['income_refusal','explicit_unsupported_categories'])
    ids=sorted(selected|extra);assert len(ids)==333 and len(selected)==321 and len(extra)==12
    original=np.array([rid in selected for rid in ids]);Y=np.array([[MODES.index(by[rid,t]['chosen_mode']) if by[rid,t]['choice_status']=='selected' else -1 for t in ORDER] for rid in ids])
    assert (Y>=0).sum()==3328 and (Y[original]>=0).sum()==3208 and (Y[~original]>=0).all()
    assert all(by[rid,t]['choice_status'] in ['selected','unable'] for rid in ids for t in ORDER)
    # The whole 333-person panel is the bootstrap unit; subgroup membership is
    # carried with each selected person, not independently re-sampled in each mean.
    draws=np.random.default_rng(917).integers(len(ids),size=(10000,len(ids)),dtype=np.int32)
    W=np.stack([np.bincount(row,minlength=len(ids)) for row in draws]).astype(float)
    spec=readj(OUT/'contrasts.json')['Shanghai'];rows=[];unitcols=[];bootcols=[]
    frozen=readj(OUT/'family_responses.json')['Shanghai']['ownership']['s9']['responses']
    for contrast,s in spec.items():
        tt=[ORDER.index(t) for t in s['weights']];weight=np.array(list(s['weights'].values()));mm=[MODES.index(m) for m in s['modes']]
        good=(Y[:,tt]>=0).all(1);sub=good&original
        values=(np.isin(Y[:,tt],mm)*weight).sum(1).astype(float)
        fullmean=float(values[good].mean());submean=float(values[sub].mean());extramean=float(values[good&~original].mean())
        assert abs(submean-frozen[contrast]['human'])<1e-12 and int(sub.sum())==frozen[contrast]['n']
        bfull=(W@np.where(good,values,0))/(W@good.astype(float));bsub=(W@np.where(sub,values,0))/(W@sub.astype(float));bdiff=bfull-bsub
        diff=fullmean-submean;assert abs(diff-(good&~original).sum()/good.sum()*(extramean-submean))<1e-12
        def q(a,x):return float(np.quantile(a,x))*100
        rows.append(dict(contrast=contrast,modes='|'.join(s['modes']),n_original=int(sub.sum()),n_complete_residents=int(good.sum()),n_additional=int((good&~original).sum()),
            original_human_response_pp=100*submean,complete_resident_human_response_pp=100*fullmean,additional12_human_response_pp=100*extramean,
            full_minus_original_response_pp=100*diff,
            original_ci95_low=q(bsub,.025),original_ci95_high=q(bsub,.975),complete_ci95_low=q(bfull,.025),complete_ci95_high=q(bfull,.975),
            difference_ci95_low=q(bdiff,.025),difference_ci95_high=q(bdiff,.975),difference_ci_family8_low=q(bdiff,.05/16),difference_ci_family8_high=q(bdiff,1-.05/16)))
        unitcols.append(np.where(good,values,np.nan));bootcols.append(bdiff)
    csvout(DEST/'human_responses_333_vs_321.csv',rows)
    np.savez_compressed(DEST/'paired_nested_bootstrap.npz',indices=draws,respondents=np.array(ids),original_subset=original,contrast=np.array(list(spec)),response_units=np.stack(unitcols,axis=1),bootstrap_full_minus_original=np.stack(bootcols,axis=1))
    dist=[];taskdist=[]
    groups={'all_additional12':extra,'unsupported_category11':{rid for rid in extra if reason[rid]=='explicit_unsupported_categories'},'income_refusal1':{rid for rid in extra if reason[rid]=='income_refusal'}}
    for name,group in groups.items():
        counts=Counter(by[rid,t]['chosen_mode'] for rid in group for t in ORDER);n=len(group)*10
        for mode in MODES:dist.append(dict(group=name,n_respondents=len(group),n_task_choices=n,mode=mode,count=counts[mode],choice_share=counts[mode]/n))
        for task in ORDER:
            counts=Counter(by[rid,task]['chosen_mode'] for rid in group)
            for mode in MODES:taskdist.append(dict(group=name,task=task,n_respondents=len(group),mode=mode,count=counts[mode],choice_share=counts[mode]/len(group)))
    csvout(DEST/'additional_respondents_mode_distribution.csv',dist);csvout(DEST/'additional_respondents_task_distribution.csv',taskdist)
    text=[r'\subsection{Sample flow and human-only coverage sensitivity}',r'\label{sec:sample_flow}',
      'All 341 Shanghai submissions recorded consent. Eight respondents reported no travel in Shanghai during the previous 30 days and skipped the subsequent persona and choice questions. Of the remaining 333 eligible respondents who answered all ten tasks, 321 met the frozen input-mapping rules. Eleven were omitted from model evaluation because their stated habitual mode and/or absence of a fixed departure schedule lay outside that mapping, and one because income was refused. These were mapping exclusions, not twelve incomplete questionnaires. Their 120 explicit task choices remain available. The model-evaluated cohort contains 3,208 explicit mode choices and two none-suitable responses; the complete eligible cohort contains 3,328 explicit choices and the same two none-suitable responses.',
      '',
      "The Singapore export contained 334 consented submissions. Two respondents selected ``Tourist / Visitor (Not living in Singapore)'' and were screened out, leaving 332 eligible respondents and 3,320 complete task choices. Its 23 missing or undisclosed incomes were retained using a declared medium reference and low/high sensitivities. The original Shanghai rule instead omitted its one income refusal. We report this difference in mapping policy rather than attributing it to questionnaire noncompletion.",
      '',
      'We evaluate the effect of the Shanghai mapping restriction on the human response target without assigning model inputs to the omitted respondents. Each bootstrap draw samples 333 eligible respondents with replacement and carries all ten answers and the original mapping-subset indicator together. The complete-cohort mean and the original-subset mean are then computed from that same draw, excluding only respondents whose constituent contrast answers include a none-suitable response. This nested respondent bootstrap preserves the overlap between cohorts; it does not treat their estimates as independent. We use 10,000 draws with seed 917. The resulting difference describes sensitivity to the recorded mapping restriction, not its causal effect or a city-representative population comparison.',
      '',r'\begin{table}[pos=htbp]',r'\centering\footnotesize',
      r'\caption{Shanghai human responses with and without the frozen input-mapping restriction. Changes and differences are percentage points; the last column is complete eligible respondents minus the original model-evaluated cohort with a nominal 95\% nested respondent-bootstrap interval.}',
      r'\label{tab:sample_flow_response}',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.13}',
      r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrl@{}}',r'\toprule',
      r'Contrast & $N_{321}$ & $N_{333}$ & Original & Complete & Difference [95\% interval] \\',r'\midrule']
    for r in rows:
        text.append(' & '.join([NAMES[r['contrast']],str(r['n_original']),str(r['n_complete_residents']),f"{r['original_human_response_pp']:.2f}",f"{r['complete_resident_human_response_pp']:.2f}",f"{r['full_minus_original_response_pp']:+.2f} [{r['difference_ci95_low']:.2f}, {r['difference_ci95_high']:.2f}]"])+r' \\')
    text += [r'\bottomrule',r'\end{tabular*}',r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize',
      'PT is the outcome except parking and road disruption, which use the recorded driving/taxi composite. All additional 12 respondents contribute explicit answers to every contrast. The two none-suitable responses reduce the interaction and walk-versus-wait counts from 321/333 to 320/332. These are complete-case contrast denominators, not additional respondent exclusions. Family-adjusted eight-contrast intervals and per-cohort intervals are retained in the result CSV. An interval spanning zero does not demonstrate equivalence.',r'\end{minipage}',r'\end{table}','']
    union={r['mode']:r['count'] for r in dist if r['group']=='all_additional12'}
    maxchange=max(rows,key=lambda r:abs(r['full_minus_original_response_pp']))
    sign_preserved=all(np.sign(r['original_human_response_pp'])==np.sign(r['complete_resident_human_response_pp']) for r in rows)
    text.append(f"Across the omitted 12 respondents' 120 task choices, {union['car']} were driving/taxi, {union['pt']} PT, {union['bike']} bicycle and {union['walk']} walking. These repeated task counts are descriptive, not 120 independent respondents. The largest absolute change in an aggregate human response is {abs(maxchange['full_minus_original_response_pp']):.2f} points ({NAMES[maxchange['contrast']]}). "+('All eight point-estimate response directions are retained. ' if sign_preserved else 'Some point-estimate response directions change. ')+ 'This comparison leaves the frozen model cohort and its Teacher sample unchanged, and does not test model predictions for the 12 omitted people.')
    (PAPER/'Elsevier_template/generated/revision20260921/supp_sample_flow.tex').write_text('\n'.join(text)+'\n',encoding='utf-8')
    methods='Shanghai model comparisons retain the 321-person frozen mapping cohort; a separate human-only sensitivity compares it with all 333 consented respondents eligible by recent Shanghai travel using a shared, nested respondent bootstrap.'
    limits='Twelve complete Shanghai respondents fall outside the original input-mapping rule, including one income refusal; the human-only coverage comparison quantifies their influence on aggregate responses but cannot establish predictive validity for those omitted participants.'
    (DEST/'main_text_suggestions.txt').write_text('Methods: '+methods+'\n\nLimitations: '+limits+'\n',encoding='utf-8')
    writej(DEST/'verification.json',dict(status='passed',n_residents=333,n_original=321,n_additional=12,original_explicit_choices=3208,all_resident_explicit_choices=3328,additional_explicit_choices=120,none_suitable=2,
        all_original_human_responses_reproduce_frozen_results=True,all_difference_mixture_identities_exact=True,point_estimate_directions_preserved=sign_preserved,
        max_absolute_response_change_pp=abs(maxchange['full_minus_original_response_pp']),largest_change_contrast=maxchange['contrast'],
        bootstrap=dict(n_resamples=10000,seed=917,unit='complete recent-travel-eligible respondent, with all tasks and fixed subset membership retained',comparison='same draw for original subset and full cohort; complete-case exclusion within contrast',nominal_interval=.95,family='Bonferroni adjustment within the eight contrasts; no independent-cohort assumption'),
        legacy_schema_note='Existing path and field names containing resident are retained for compatibility; S02 screens travel in Shanghai in the previous 30 days, not residence. Cohort and all numerical statistics unchanged.',source_sha256=hashes,sources_unchanged=all(sha(ROOT/p)==h for p,h in hashes.items()),no_student_mapping_or_prediction=True,no_teacher_api_calls=True,code_sha256=sha(__file__)))
    report=['# Shanghai complete-eligible-cohort human-only sensitivity','',methods,'',limits,'',
      f'All 333 eligible respondents answered ten tasks; 321 are the frozen model subset, 12 additional people supply 120 explicit choices. The same two none-suitable answers remain unscored. Largest absolute response change: {abs(maxchange["full_minus_original_response_pp"]):.4f} pp ({maxchange["contrast"]}). Directions preserved: {sign_preserved}.','',
      'The bootstrap samples the 333-person empirical distribution and carries the subset membership indicator, preserving overlap; this also allows its empirical mixture proportion to vary between draws. It is a coverage sensitivity, not a causal estimate. Frozen models, their cohort and all Teacher files are unchanged.','',
      'Files: human_responses_333_vs_321.csv (point estimates, counts and shared-bootstrap intervals), additional_respondents_mode_distribution.csv (12-person union plus category11/income1), additional_respondents_task_distribution.csv, paired_nested_bootstrap.npz, verification.json. The generated supp_sample_flow.tex is ready for the root coordinator to insert.']
    (DEST/'HUMAN_COVERAGE_SENSITIVITY.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    writej(DEST/'manifest.json',dict(files={p.name:sha(p) for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'},
        supplement_fragment_sha256=sha(PAPER/'Elsevier_template/generated/revision20260921/supp_sample_flow.tex'),
        legacy_path_note='Directory/CSV fields containing resident are historical names; actual S02 screens recent Shanghai travel, not residence.'))
    print(json.dumps(dict(status='complete',response_comparison=rows,additional_distribution=dist),ensure_ascii=False))

if __name__=='__main__':main()
