"""Rebuild the article's two-panel Figure 3 from the supplied summary CSV.

The original ZIP contained a scatter-plot script/SVG but a two-panel publication
PDF. This script retains the publication's response-bars / timing-bars design,
using only its supplied summary estimates. The 4.97% annotation and 7.43--7.71
reference range are the values reported in the manuscript, not recalculated
from rounded CSV entries. No new model fits or uncertainty estimates are made.
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
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures/narrative'
with (ROOT/'data_tables/figure3_response_timing.csv').open(newline='',encoding='utf-8-sig') as f:
    rows={r['model']:r for r in csv.DictReader(f)}
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8.5,'axes.labelsize':8.5,
    'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65})
H=243.779526;W=510.236206;W0=276;W1=W-W0
paths=[]
# One chart per figure: the two chart PDFs are assembled into the publication panel.
fig=plt.figure(figsize=(W0/72,H/72));ax=fig.add_axes([.29,.19,.69,.64])
models=['Soft KL','CE+KL','Signed response','Direction+magnitude','MNL-S + ridge']
labels=['Soft KL','CE+KL','Signed response','Dir. + mag.','MNL-S']
vals=[float(rows[x]['response_error']) for x in models]
sds=[float(rows[x]['response_error_sd']) for x in models]
colors=['#BFD8E3','#CFDEE6','#7BAAC3','#386B89','#2A8079']
ax.barh(np.arange(5),vals,color=colors,height=.57)
ax.errorbar(vals[:4],np.arange(4),xerr=sds[:4],fmt='none',ecolor='black',capsize=2,elinewidth=.75,capthick=.75)
ax.set_yticks(np.arange(5),labels);ax.invert_yaxis();ax.set_xlim(0,.104)
ax.set_xticks([0,.02,.04,.06,.08,.10]);ax.set_xlabel('Probability-response error')
ax.axvline(vals[0],linestyle='--',linewidth=.7,color='#6F6F6F')
for y,v in enumerate(vals):ax.text(v+sds[y]+.002,y-.02,f'{v:.4f}',va='center',fontsize=7.4)
ax.text(.040,3,'−4.97%',color='white',va='center',fontsize=8.3,fontweight='bold')
ax.grid(axis='x',alpha=.19,linewidth=.5);ax.set_axisbelow(True)
fig.text(.025,.945,'(a) Response supervision',fontsize=10,fontweight='bold')
fig.text(.025,.889,'Models selected by validation KL',fontsize=8)
p=OUT/'figure3_response_panel.pdf';fig.savefig(p);fig.savefig(p.with_suffix('.svg'));plt.close(fig);paths.append(p)
fig=plt.figure(figsize=(W1/72,H/72));ax=fig.add_axes([.20,.19,.77,.64])
models=['MNL-S + ridge','MNL-S + bounded linear','MNL-S + independent neural timing']
vals=[float(rows[x]['departure_mae']) for x in models];sds=[float(rows[x]['departure_mae_sd']) for x in models]
ax.axhspan(7.43,7.71,color='#999999',alpha=.22,zorder=1)
ax.bar(np.arange(3),vals,width=.52,color=['#B9DAD3','#73B5AA','#287A70'],zorder=3)
ax.errorbar([1,2],vals[1:],yerr=sds[1:],fmt='none',ecolor='black',capsize=2,elinewidth=.75,capthick=.75,zorder=4)
ax.set_xticks(np.arange(3),['Ridge','Bounded\nlinear','Neural']);ax.set_ylim(0,15)
ax.set_yticks(np.arange(0,15,2));ax.set_ylabel('Departure MAE (min)')
for x,v in enumerate(vals):ax.text(x,v+sds[x]+.35,f'{v:.2f}',ha='center',fontsize=8)
ax.grid(axis='y',alpha=.19,linewidth=.5);ax.set_axisbelow(True)
ax.text(.98,.97,'Joint neural means:\n7.43–7.71 min',transform=ax.transAxes,ha='right',va='top',fontsize=7.6)
fig.text(.035,.945,'(b) Modular departure prediction',fontsize=10,fontweight='bold')
fig.text(.035,.889,'Fixed MNL-S choice probabilities',fontsize=8)
p=OUT/'figure3_timing_panel.pdf';fig.savefig(p);fig.savefig(p.with_suffix('.svg'));plt.close(fig);paths.append(p)
doc=fitz.open();page=doc.new_page(width=W,height=H)
for i,(p,x,w) in enumerate(zip(paths,[0,W0],[W0,W1])):
    src=fitz.open(p);page.show_pdf_page(fitz.Rect(x,0,x+w,H),src,0)
doc.save(OUT/'figure3_response_timing.pdf',garbage=4,deflate=True)
NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS)
root=ET.Element(f'{{{NS}}}svg',{'width':f'{W}pt','height':f'{H}pt','viewBox':f'0 0 {W} {H}','version':'1.1'})
for i,(p,x,w) in enumerate(zip(paths,[0,W0],[W0,W1])):
    child=ET.parse(p.with_suffix('.svg')).getroot()
    child.set('x',str(x));child.set('y','0');child.set('width',str(w));child.set('height',str(H))
    prefix=f'p{i}_'
    for el in child.iter():
        if el.get('id'):el.set('id',prefix+el.get('id'))
        for k,v in list(el.attrib.items()):
            if 'url(#' in v:el.set(k,re.sub(r'url\(#([^)]*)\)',lambda m:'url(#'+prefix+m.group(1)+')',v))
            elif k.endswith('href') and v.startswith('#'):el.set(k,'#'+prefix+v[1:])
    root.append(child)
ET.ElementTree(root).write(OUT/'figure3_response_timing.svg',encoding='utf-8',xml_declaration=True)
print('Figure 3: publication two-panel design rebuilt from unchanged summary estimates.')
