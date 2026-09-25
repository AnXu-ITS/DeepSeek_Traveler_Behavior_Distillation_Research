"""Draw Figure 4 from the published full-cohort response tables.

Each city is drawn as a separate matplotlib figure, then the vector panels are
assembled vertically. Error bars are the source's Bonferroni-adjusted intervals;
no respondent-level data or new uncertainty estimates are constructed.
"""
from pathlib import Path
import csv
import re
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/narrative'
MODELS = ['S9','MNL-S','Soft KL','CE+KL','Signed response','Direction+magnitude']
LEGEND = ['SA-Student','MNL-S','Soft KL','CE+KL','Signed response','Direction + magnitude']
# Read the supplied summary file; do not reconstruct participant data or
# overwrite archival identifiers to match display names.
with (ROOT/'data_tables/figure4_response_bars.csv').open(newline='',encoding='utf-8-sig') as f:
    ROWS=list(csv.DictReader(f))
for row in ROWS:
    row['n']=int(row['n'])
    for key in ('human','model_response','bias','lower','upper'):
        row[key]=float(row[key])
assert len(ROWS)==78, len(ROWS)

plt.rcParams.update({'font.family':'Liberation Sans','font.size':9,
                     'axes.labelsize':9.5,'axes.titlesize':11,
                     'xtick.labelsize':9,'ytick.labelsize':9,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7})
labels={
    'Singapore':['PT access\n(PT)','Rain\n(bike + walk)','Fare\n(PT)','Delay\n(PT)','Roadworks\n(car)'],
    'Shanghai':['Rain\n(PT)','Delay\n(PT)','Rain ×\ndelay\n(PT)','Fare\n(PT)',
                'Parking\n(car)','Road\npenalty\n(car)','Walk vs\nwait\n(PT)','Transfer vs\nwait\n(PT)']}
panel_paths=[]
for city,letter,n,ylim in [('Singapore','a',332,(-13,50)),('Shanghai','b',321,(-44,20))]:
    subset=[r for r in ROWS if r['city']==city]
    contrasts=list(dict.fromkeys(r['contrast'] for r in subset))
    # One chart per matplotlib figure; shared ordering across all six models.
    fig=plt.figure(figsize=(7.20,3.00))
    ax=fig.add_axes([0.086,0.20,0.903,0.64])
    x=np.arange(len(contrasts),dtype=float);w=.118
    for mi,(model,lab) in enumerate(zip(MODELS,LEGEND)):
        rs=[next(r for r in subset if r['model']==model and r['contrast']==c) for c in contrasts]
        y=np.array([r['bias'] for r in rs])
        err=np.array([[r['bias']-r['lower'] for r in rs],[r['upper']-r['bias'] for r in rs]])
        if np.any(err<0): raise ValueError('Interval does not enclose its point estimate')
        ax.bar(x+(mi-2.5)*w,y,width=w*.89,label=lab,yerr=err,capsize=1.35,
               error_kw={'elinewidth':.60,'capthick':.60},alpha=.85,zorder=3)
    ax.axhline(0,linewidth=.8,linestyle='-',zorder=2)
    ax.set_ylim(*ylim);ax.set_xlim(-.54,len(contrasts)-.46)
    ax.set_xticks(x,labels[city]);ax.tick_params(axis='x',length=0,pad=5)
    ax.set_ylabel('Model − human response (pp)')
    ax.set_title(f'({letter}) {city}  |  Full survey sample, n = {n}',loc='left',pad=12,fontweight='bold')
    # Keep all four frame edges; uncertainty is distinguished from Figure 5 by
    # vertical grouped bars and the use of family-adjusted intervals.
    for sp in ax.spines.values():sp.set_visible(True)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,1.02),
               ncol=6,frameon=False,fontsize=7.9,handlelength=1.25,handletextpad=.4,columnspacing=.95)
    path=OUT/f'figure4_{city.lower()}_bars.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.svg'))
    fig.savefig(path.with_suffix('.png'),dpi=180)
    plt.close(fig);panel_paths.append(path)

combined=fitz.open();width=7.2*72;height=6*72
page=combined.new_page(width=width,height=height)
for i,path in enumerate(panel_paths):
    d=fitz.open(path);page.show_pdf_page(fitz.Rect(0,i*216,width,(i+1)*216),d,0)
combined.save(OUT/'figure4_human_response.pdf',garbage=4,deflate=True)
page.get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(OUT/'figure4_human_response.png')
combined.close()
# Preserve editable vector text in the combined SVG as well as the PDFs.
NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS)
root=ET.Element(f'{{{NS}}}svg',{'width':'518.4pt','height':'432pt','viewBox':'0 0 518.4 432','version':'1.1'})
for i,path in enumerate(panel_paths):
    child=ET.parse(path.with_suffix('.svg')).getroot()
    child.set('x','0');child.set('y',str(216*i));child.set('width','518.4');child.set('height','216')
    # Prefix ids and references so the two exported matplotlib SVGs stay distinct.
    prefix=f'panel{i}_'
    for el in child.iter():
        if el.get('id'):el.set('id',prefix+el.get('id'))
        for key,val in list(el.attrib.items()):
            if 'url(#' in val:el.set(key,re.sub(r'url\(#([^)]*)\)',lambda m:'url(#'+prefix+m.group(1)+')',val))
            elif key.endswith('href') and val.startswith('#'):el.set(key,'#'+prefix+val[1:])
    root.append(child)
ET.ElementTree(root).write(OUT/'figure4_human_response.svg',encoding='utf-8',xml_declaration=True)
print('Figure 4: two grouped-bar panels, six models and all 13 survey contrasts.')
print('Source intervals retained; data_tables/figure4_response_bars.csv is read-only.')
