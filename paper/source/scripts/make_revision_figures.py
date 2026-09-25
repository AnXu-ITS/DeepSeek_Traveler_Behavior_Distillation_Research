"""Regenerate the narrative figures from frozen, packaged CSV/JSON evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data_tables'/'revision20260925'
OUT=ROOT/'figures'/'revision20260925';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,
 'axes.spines.right':False,'axes.edgecolor':'#8A959E','axes.linewidth':.6,
 'grid.color':'#E2E7EA','grid.linewidth':.5,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white'})
INK='#243746';BLUE='#2874A6';TEAL='#14817B';ORANGE='#CB7034';PURPLE='#8263A3';GREY='#788692'
def save(fig,name):
 for ext in ('pdf','svg','png'):fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',pad_inches=.08,dpi=220)
 plt.close(fig)
def title(ax,letter,text):ax.set_title(f'{letter}  {text}',loc='left',fontweight='bold',pad=12,color=INK)
def clean(ax):ax.set_axisbelow(True);ax.grid(axis='x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)

def concept():
 fig=plt.figure(figsize=(7.15,3.1));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 ax.text(.02,.96,'One traveler, one trip, two conditions',fontsize=12,weight='bold',color=INK)
 for x,head,body,col in [(0.02,'Baseline','Traveler + trip\nCurrent travel conditions',BLUE),(.53,'Changed condition','Same traveler + trip\nDelay, fare, access or rain',ORANGE)]:
  ax.add_patch(FancyBboxPatch((x,.65),.43,.23,boxstyle='round,pad=.012',facecolor=col+'12',edgecolor=col,lw=.9))
  ax.text(x+.02,.83,head,weight='bold',color=col);ax.text(x+.02,.785,body,va='top',linespacing=1.2)
 ax.annotate('',(.515,.775),(.465,.775),arrowprops={'arrowstyle':'->','color':INK})
 ax.text(.5,.585,'Response = change in the choice distribution',ha='center',fontsize=11,weight='bold',color=INK)
 cols=[(.02,'1  Learn the change','Does the Student preserve\nthe Teacher response?',BLUE),(.355,'2  Compare with people','Does that response agree\nwith stated choices?',TEAL),(.69,'3  Follow the response','Do predicted changes survive\nassignment and boarding?',ORANGE)]
 for x,h,b,c in cols:
  ax.add_patch(FancyBboxPatch((x,.19),.285,.30,boxstyle='round,pad=.012',facecolor='#F5F7F8',edgecolor='#D5DEE4'))
  ax.text(x+.007,.43,h,fontsize=9.1,weight='bold',color=c);ax.text(x+.007,.345,b,fontsize=9,va='top',linespacing=1.6)
 ax.text(.5,.075,'Each comparison answers a different question; none substitutes for the others.',ha='center',fontsize=9,color=GREY)
 save(fig,'figure1_questions')

def learning():
 grid=pd.read_csv(D/'weight_selection_summary.csv');formal=pd.read_csv(D/'formal_paired_contrasts.csv')
 old=pd.read_csv(ROOT/'data_tables'/'figure3_response_timing.csv').iloc[:5]
 fig,axs=plt.subplots(1,2,figsize=(7.15,3.15),gridspec_kw={'width_ratios':[.94,1.3]})
 ax=axs[0];labels=['Soft KL','CE + KL','Signed response','Direction + magnitude','MNL-S']
 for i,row in old.reset_index(drop=True).iterrows():
  ax.barh(i,row.response_error,height=.62,xerr=row.response_error_sd,color=[GREY,'#B0B8BE',BLUE,ORANGE,PURPLE][i],error_kw={'ecolor':INK,'elinewidth':.8,'capsize':2},zorder=3)
 ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlabel('Response error');ax.set_xlim(0,.09);clean(ax);title(ax,'a','Archived Teacher')
 ax.text(.02,-.28,'Six test personas; whiskers: seed SD',transform=ax.transAxes,fontsize=8,color=GREY)
 ax=axs[1];cards=['ALL','delay_8','delay_45','fare_delay'];labels=['All contrasts','Delay: 8 min','Delay: 45 min','Fare + delay']
 for family,col,mark,off,lab in [('full',BLUE,'o',-.18,'Full training'),('delay_holdout',ORANGE,'s',.18,'Delay family withheld')]:
  for i,card in enumerate(cards):
   q=formal.query('family==@family and selection=="static" and card==@card').iloc[0]
   ax.barh(i+off,q['mean'],height=.30,xerr=[[q['mean']-q.ci_low],[q.ci_high-q['mean']]],color=col,error_kw={'ecolor':INK,'elinewidth':.8,'capsize':2},label=lab if i==0 else None,zorder=3)
 ax.axvline(0,color=INK,lw=.8);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlabel('Signed-response error minus soft-KL error');ax.set_xlim(-.019,.017);clean(ax);title(ax,'b','New synthetic personas')
 ax.legend(loc='lower left',bbox_to_anchor=(-.03,-.51),frameon=False,fontsize=8)
 ax.text(.02,-.27,'← signed response better    worse →',transform=ax.transAxes,fontsize=8,color=GREY)
 fig.subplots_adjust(left=.195,right=.98,bottom=.30,top=.88,wspace=.76);save(fig,'figure2_learning')

def generalization():
 d=pd.read_csv(D/'formal_paired_contrasts.csv');a=json.loads((D/'formal_analysis.json').read_text())
 fig,axs=plt.subplots(1,2,figsize=(7.15,3.35),gridspec_kw={'width_ratios':[1.05,1]})
 vals=np.sort([r['signed_minus_softkl'] for r in a['primary']]);ax=axs[0]
 ax.barh(range(len(vals)),vals,height=.72,color=np.where(vals<0,BLUE,ORANGE),zorder=3)
 ax.axvline(0,color=INK,lw=.7);ax.set(yticks=[],xlabel='Signed-response error minus soft-KL error',ylabel='Thirty new personas, ordered by difference');clean(ax);title(ax,'a','Person-level paired differences')
 ax=axs[1];cards=['fare_1.25','fare_2.5','access_5','access_20','fare_access','pt_infeasible','rain_0.25','rain_0.9']
 # Use the frozen card names, with an explicit assertion to prevent silent omissions.
 actual=set(d.card);cards=['fare_1_25','fare_2_5','access_5','access_20','fare_access','pt_infeasible','weather_0_25','weather_0_9']
 assert set(cards)==actual-{'ALL','delay_8','delay_45','fare_delay'}
 labels={'fare_1_25':'Fare × 1.25','fare_2_5':'Fare × 2.5','access_5':'Access + 5 min','access_20':'Access + 20 min','fare_access':'Fare + access','pt_infeasible':'No PT connection','weather_0_25':'Rain: 0.25','weather_0_9':'Rain: 0.9'}
 for i,c in enumerate(cards):
  q=d.query('family=="full" and selection=="static" and card==@c').iloc[0]
  ax.barh(i,q['mean'],height=.60,xerr=[[q['mean']-q.ci_low],[q.ci_high-q['mean']]],color=BLUE,error_kw={'ecolor':INK,'elinewidth':.8,'capsize':2},zorder=3)
 ax.axvline(0,color=INK,lw=.7);ax.set_yticks(range(len(cards)),[labels.get(c,c.replace('_',' ')) for c in cards]);ax.invert_yaxis();clean(ax);ax.set_xlabel('Paired error difference');title(ax,'b','Other retained contrasts')
 fig.subplots_adjust(left=.07,right=.98,bottom=.19,top=.88,wspace=.80);save(fig,'figureA_generalization')

def human():
 d=pd.read_csv(ROOT/'data_tables'/'figure4_response_bars.csv')
 models=list(d.model.drop_duplicates());mapping={'S9':'SA-Student','MNL-S':'MNL-S','Soft KL':'Soft KL','CE+KL':'CE + KL','Signed response':'Signed','Direction+magnitude':'Direction + mag.'}
 colors=[TEAL,PURPLE,GREY,'#AAB4BD',BLUE,ORANGE]
 fig=plt.figure(figsize=(7.15,6.3));gs=fig.add_gridspec(2,2,height_ratios=[1,1.35],hspace=.60,wspace=.66)
 for k,(city,contrast,head) in enumerate([('Singapore','PT access (PT)','Poorer PT access'),('Shanghai','Delay (PT)','PT delay')]):
  ax=fig.add_subplot(gs[0,k]);q=d.query('city==@city and contrast==@contrast').set_index('model')
  for i,m in enumerate(models):
   rr=q.loc[m];ax.barh(i,rr.bias,height=.60,xerr=[[rr.bias-rr.lower],[rr.upper-rr.bias]],color=colors[i],error_kw={'ecolor':INK,'elinewidth':.8,'capsize':2},zorder=3)
  ax.axvline(0,color=INK,lw=.7);ax.set_yticks(range(len(models)),[mapping.get(m,m) for m in models]);ax.invert_yaxis();ax.set_xlim(-30,55);ax.set_xlabel('Response bias (percentage points)');clean(ax);title(ax,'ab'[k],city+': '+head)
 for k,city in enumerate(['Singapore','Shanghai']):
  ax=fig.add_subplot(gs[1,k]);q=d[d.city==city];contrasts=list(q.contrast.drop_duplicates());arr=np.array([[float(q[(q.contrast==c)&(q.model==m)].iloc[0].bias) for m in models] for c in contrasts]);im=ax.pcolormesh(np.arange(len(models)+1)-.5,np.arange(len(contrasts)+1)-.5,arr,cmap='PuOr_r',norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=40),rasterized=False);ax.invert_yaxis()
  for i,c in enumerate(contrasts):
   for j,m in enumerate(models):
    rr=q[(q.contrast==c)&(q.model==m)].iloc[0]
    if rr.lower>0 or rr.upper<0:ax.text(j,i,'•',ha='center',va='center',color='white' if abs(rr.bias)>22 else INK,fontsize=12)
  short=[c.replace(' (PT)','').replace(' (car)','').replace(' (bike + walk)','').replace('Rain $\\times$ delay','Rain × delay').replace('PT access','PT access').replace('Transfer vs wait','Transfer / wait').replace('Walk vs wait','Walk / wait').replace('Road penalty','Roadworks') for c in contrasts]
  ax.set_yticks(range(len(contrasts)),short);ax.set_xticks(range(len(models)),[mapping.get(m,m) for m in models],rotation=55,ha='right',fontsize=7.5);ax.tick_params(length=0);title(ax,'cd'[k],city+': all contrasts')
 cax=fig.add_axes([.34,.035,.40,.017]);cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.solids.set_rasterized(False);cb.set_label('More negative  ←  response bias (pp)  →  more positive',fontsize=8);cb.set_ticks([-40,-20,0,20,40])
 fig.subplots_adjust(left=.15,right=.99,bottom=.22,top=.95);save(fig,'figure3_human')

def decomposition():
 d=pd.read_csv(ROOT/'data_tables'/'teacher_decomposition_published_precision.csv');d=d[d.model=='s9']
 examples=[('Singapore','Poorer PT access'),('Shanghai','PT delay'),('Shanghai','Walk vs wait')]
 # Match descriptive names in the source explicitly below.
 print('Decomposition contrasts:',list(d.contrast.drop_duplicates()))
 examples=[('Singapore',list(d[d.city=='Singapore'].contrast.drop_duplicates())[0]),('Shanghai',next(c for c in d[d.city=='Shanghai'].contrast.unique() if 'delay' in c.lower() and 'rain' not in c.lower())),('Shanghai',next(c for c in d[d.city=='Shanghai'].contrast.unique() if 'walk' in c.lower()))]
 fig,axs=plt.subplots(1,3,figsize=(7.15,2.6),sharex=True)
 metrics=['teacher_minus_human','student_minus_teacher','student_minus_human'];labels=['Reference − human','Student − reference','Student − human']
 for k,(city,c) in enumerate(examples):
  ax=axs[k];q=d[(d.city==city)&(d.contrast==c)].set_index('metric')
  for i,mt in enumerate(metrics):
   rr=q.loc[mt];ax.barh(i,rr.estimate,height=.56,xerr=[[rr.estimate-rr.nominal_lower],[rr.nominal_upper-rr.estimate]],color=[BLUE,ORANGE,GREY][i],error_kw={'ecolor':INK,'elinewidth':.8,'capsize':2},zorder=3)
  ax.axvline(0,color=GREY,lw=.7);ax.set_yticks(range(3),labels if k==0 else []);ax.invert_yaxis();ax.set_xlim(-45,50);ax.set_xlabel('Difference (pp)');clean(ax);title(ax,'abc'[k],['SG: PT access','SH: delay','SH: walk / wait'][k])
 fig.subplots_adjust(left=.195,right=.99,bottom=.25,top=.81,wspace=.16);save(fig,'figure4_decomposition')

def supply():
 runs=pd.DataFrame(json.loads((D/'physical_run_metrics.json').read_text()));cs=json.loads((D/'physical_verified_contrasts.json').read_text())
 fig,axs=plt.subplots(2,2,figsize=(7.15,5.25),gridspec_kw={'height_ratios':[.95,1.15]})
 arms=['baseline','perceived_delay','frozen_disrupted','recomputed_disrupted'];labs=['Baseline','Perceived\ndelay','Fewer trips,\nfixed behavior','Fewer trips,\nupdated inputs'];cols=[GREY,ORANGE,PURPLE,TEAL];marks=['o','s','^','D']
 for k,model in enumerate(['s9','independent_neural_seed42']):
  ax=axs[0,k];x=np.arange(4)
  for j,arm in enumerate(arms):
   q=runs[(runs.model==model)&(runs.arm==arm)];v=[q.raw_pt.mean()*100,q.adjusted_pt.mean()*100,q.assigned_pt.mean()/10,q.boarded_pt.mean()/10]
   ax.plot(x,v,marker=marks[j],color=cols[j],label=labs[j].replace('\n',' '),lw=1.2,ms=4)
  ax.set_xticks(x,['Predicted','Feasible','Assigned','Boarded'],rotation=20);ax.set_ylim(0,60);ax.set_ylabel('PT share (%)');ax.grid(axis='y');title(ax,'ab'[k],['SA-Student','MNL-S + neural timing'][k])
  ax=axs[1,k]
  for i,arm in enumerate(arms[1:]):
   row=next(c for c in cs if c['model']==model and c['arm']==arm and c['baseline']=='baseline')
   for mt,off,col,mark,label in [('raw_pt_pp',-.18,GREY,'o','Predicted'),('boarded_pt_pp',.18,TEAL,'s','Boarded')]:
    q=row['metrics'][mt];v=q['mean'];lo,hi=q['ci95'];ax.barh(i+off,v,height=.30,xerr=[[v-lo],[hi-v]],color=col,error_kw={'ecolor':INK,'elinewidth':.8,'capsize':2},label=label if i==0 else None,zorder=3)
  ax.axvline(0,color=INK,lw=.7);ax.set_yticks(range(3),['Perceived delay','Fewer trips, fixed','Fewer trips, updated']);ax.invert_yaxis();ax.set_xlim(-17,2);ax.set_xlabel('Change from baseline (pp)');clean(ax);title(ax,'cd'[k],'Response to each change')
 axs[0,0].legend(loc='lower left',bbox_to_anchor=(-.04,-.63),ncol=2,frameon=False,fontsize=7.5,columnspacing=.9)
 axs[1,1].legend(loc='lower left',frameon=False,fontsize=8)
 fig.subplots_adjust(left=.18,right=.99,top=.93,bottom=.10,hspace=1.1,wspace=.65);save(fig,'figure5_supply')

def optimization():
 d=pd.read_csv(D/'weight_selection_summary.csv');fig,axs=plt.subplots(1,2,figsize=(7.15,2.7),sharey=True)
 for k,sel in enumerate(['static','response']):
  ax=axs[k]
  for v,c,m,lab in [('signed_l1',BLUE,'o','Signed response'),('direction_magnitude',ORANGE,'s','Direction + magnitude')]:
   q=d.query('variant==@v and selection==@sel').sort_values('weight');ax.plot(q.weight,q.response_gap,marker=m,color=c,label=lab,lw=1.2,ms=4)
  base=d.query('variant=="soft_kl" and selection==@sel').iloc[0].response_gap;ax.axhline(base,color=GREY,ls='--',label='Soft KL');ax.set_xticks([.25,.5,1,2]);ax.set_xlabel('Response-loss weight');ax.grid(axis='y');title(ax,'ab'[k],['Select by validation KL','Select by validation response'][k])
 axs[0].set_ylabel('Archived response error');axs[1].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'figureA_weights')

if __name__=='__main__':
 concept();learning();generalization();human();decomposition();supply();optimization()
