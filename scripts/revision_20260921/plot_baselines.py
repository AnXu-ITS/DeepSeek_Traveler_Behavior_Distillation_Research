"""175-mm vector Figure 3 and reproducible audit diagnostics, from frozen results."""
from pathlib import Path
import sys,json,itertools,csv
sys.path.insert(0,str(Path(__file__).resolve().parent))
from departure_baselines import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def save(fig,name):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ['pdf','svg']:fig.savefig(folder/f'{name}.{ext}',facecolor='white')
    fig.savefig(folder/f'{name}.png',dpi=240,facecolor='white');plt.close(fig)

def main():
    assert len(read(OUT/'verification.json')['runs'])==18
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8.5,'ytick.labelsize':9,'legend.fontsize':8,'axes.linewidth':.7,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    labels=['Soft KL','CE + KL','Signed','Dir. + mag.','MNL-S']
    old={v:[read(OLD/f'test/{v}_seed{s}/metrics.json')['metrics'] for s in SEEDS] for v in MAIN_VARIANTS}
    new=read(OUT/'departure_summary.json');ridge=read(OUT/'ridge_diagnostic.json')['original']
    fig,axs=plt.subplots(1,2,figsize=(175/25.4,90/25.4),gridspec_kw={'width_ratios':[1,1.16]})
    fig.subplots_adjust(left=.115,right=.97,top=.87,bottom=.28,wspace=.61)
    ax=axs[0]
    vals=[[m['response']['response_gap'] for m in old[v]] for v in MAIN_VARIANTS]+[[ridge['response']['response_gap']]]
    for i,x in enumerate(vals):
        col=['#ADC1D1','#ADC1D1','#5484A6','#29658C','#D29251'][i]
        ax.barh(i,np.mean(x),height=.58,color=col,edgecolor='black',linewidth=.55)
        if len(x)>1:
            ax.errorbar(np.mean(x),i,xerr=np.std(x,ddof=1),fmt='none',color='black',capsize=2,lw=.7)
            ax.scatter(x,np.array([i]*3)+np.array([-.12,0,.12]),s=12,facecolors='white',edgecolors='black',linewidths=.5,zorder=4)
    ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlim(0,.095);ax.set_xticks([0,.03,.06,.09]);ax.set_xlabel('Mean response error');ax.set_title('(a) Probability responses',loc='left',pad=9,fontweight='bold')
    ax=axs[1]
    kinds=MAIN_VARIANTS+['bounded_linear','independent_neural','ridge']
    labs=['Soft KL','CE + KL','Signed','Dir. + mag.','MNL + bounded','MNL + neural','MNL + ridge']
    for i,k in enumerate(kinds):
        if k=='ridge':
            ax.plot(ridge['state']['departure_mae'],i,marker='D',ms=4,color='#A6682D');continue
        n=new[k]['departure_mae'];pos=i+.12 if k in MAIN_VARIANTS else i
        ax.errorbar(n['mean'],pos,xerr=n['sd'],fmt='o',ms=4,color='#276E9D' if k in MAIN_VARIANTS else '#A6682D',capsize=2,elinewidth=.7)
        if k in MAIN_VARIANTS:
            xx=[m['state']['departure_mae'] for m in old[k]]
            ax.errorbar(np.mean(xx),i-.12,xerr=np.std(xx,ddof=1),fmt='o',ms=4,mfc='white',mec='#444444',color='#777777',capsize=2,elinewidth=.7)
    ax.set_yticks(range(7),labs);ax.set_ylim(6.6,-.6);ax.set_xlim(5.8,13.7);ax.set_xticks([6,8,10,12]);ax.set_xlabel('Departure MAE (min)');ax.set_title('(b) Departure branch comparison',loc='left',pad=9,fontweight='bold')
    for a in axs:
        for sp in a.spines.values():sp.set_visible(True);sp.set_color('black')
        a.tick_params(direction='out',length=3);a.grid(axis='x',alpha=.13,zorder=0);a.set_axisbelow(True)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='none',mfc='white',mec='#444444',label='Original KL selection'),Line2D([0],[0],marker='o',color='none',mfc='#276E9D',mec='#276E9D',label='Common departure-MAE selection'),Line2D([0],[0],marker='o',color='none',mfc='#A6682D',mec='#A6682D',label='MNL timing branches')],loc='lower center',bbox_to_anchor=(.51,.005),ncol=2,frameon=False,handlelength=1,columnspacing=1.2)
    save(fig,'figure3_response_departure')
    # Compact supplemental mathematical/robustness visual; all intervals are diagnostics.
    fig,axs=plt.subplots(1,2,figsize=(175/25.4,77/25.4));fig.subplots_adjust(left=.105,right=.97,bottom=.23,top=.86,wspace=.36)
    x=np.linspace(-.3,.3,401);t=.2
    axs[0].plot(x,np.maximum(-x,0)+abs(t-abs(x)),color='#276E9D',label='Direction + magnitude')
    axs[0].plot(x,abs(t-x),color='#A6682D',ls='--',label='Signed absolute error');axs[0].axvspan(-t,0,color='#276E9D',alpha=.1)
    axs[0].set(xlabel='Student response s',ylabel='Scalar loss',title='(a) Equal-normalization example')
    rs=list(csv.DictReader((OUT/'leave_one_persona_out.csv').open(encoding='utf-8-sig')))
    ps=sorted({r['excluded_persona'] for r in rs})
    for j,k in enumerate(['signed_l1','direction_magnitude']):
        y=[np.mean([float(r['response_gap_difference']) for r in rs if r['model']==k and r['excluded_persona']==p]) for p in ps]
        axs[1].plot(range(6),y,'o-',ms=4,color=['#A6682D','#276E9D'][j],label=['Signed','Dir. + mag.'][j])
    axs[1].axhline(0,color='black',lw=.7);axs[1].set_xticks(range(6),[p.replace('P0000','P') for p in ps]);axs[1].set(xlabel='Excluded test persona',ylabel='Response error difference vs soft KL',title='(b) Leave-one-persona-out sensitivity')
    for ax in axs:
        for sp in ax.spines.values():sp.set_color('black')
        ax.legend(frameon=False,fontsize=8,loc='best')
    save(fig,'loss_and_persona_diagnostics')
    (OUT/'figures/CAPTIONS.md').write_text('''Figure 3. Response preservation and departure prediction. (a) Original validation-KL-selected controlled models, with identical architecture, training data, endpoint exposure and three random seeds. Bars show means, whiskers show training-seed standard deviations, and open symbols show the three runs. MNL-S is a separately fitted linear choice model and has a deterministic probability row. (b) Departure error for the original KL-selected neural models and the newly trained, validation-departure-MAE-selected models. The MNL-S choice coefficients are held fixed while its timing branch is replaced by a bounded linear or independently initialized neural predictor. Error bars show training-seed standard deviations, not confidence intervals. The ridge reference has no training-seed variability. All evaluations use 437 held-out endpoints; response error uses 398 pairs and interaction error (reported in the table) uses 94 contrasts. Active total parameter counts are 24,562 for each joint neural model, 555 for MNL-S plus bounded linear or ridge timing, and 18,733 for MNL-S plus independent neural timing.\n\nSupplemental diagnostic. (a) The direction-and-magnitude loss has a flat opposite-sign interval when its effective direction and magnitude normalizations coincide. The illustration uses a scalar target of 0.2; it does not imply identical gradients under all real masks. (b) Mean over the three training seeds after removing each of six test personas; no model is retrained. Negative values favor explicit pair supervision relative to soft KL.\n''',encoding='utf-8')
    print('175-mm vector Figure3 and supplemental diagnostics created.')
if __name__=='__main__':main()
