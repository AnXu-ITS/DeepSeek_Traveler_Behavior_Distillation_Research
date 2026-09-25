"""Revise figure wording without changing source estimates or interval geometry.

Uses the preserved vector sources. Figure 1 receives a text-only correction;
Figures 5 and 6 retain every non-text SVG element. Figure 4 is regenerated
separately from the unchanged published-precision CSV.
"""
from pathlib import Path
import math
import re
import fitz
import cairosvg
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/narrative'
BASE = ROOT / 'figures/template/publication_sources'
BASE.mkdir(parents=True, exist_ok=True)

# The original Figure 1 text conflicts with the paired-comparison definition.
# Remove text only; raster artwork and vector paths remain intact.
doc = fitz.open(ROOT / 'figures/redesign920/figure1_unchanged.pdf')
page = doc[0]
spans = [s for b in page.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l['spans']
         if s['text'] in ('Same resident', '+ different trip')]
assert len(spans) == 2
for span in spans:
    page.add_redact_annot(fitz.Rect(span['bbox']), fill=None, cross_out=False)
page.apply_redactions(images=0, graphics=0, text=0)
from matplotlib.font_manager import FontProperties, findfont
fontfile = findfont(FontProperties(family='Liberation Sans', weight='bold'))
font = fitz.Font(fontfile=fontfile)
page.insert_font(fontname='RevisionBold', fontfile=fontfile)
for span, label in zip(spans, ('Same traveler', '+ same trip')):
    center = (span['bbox'][0]+span['bbox'][2])/2
    width = font.text_length(label, fontsize=span['size'])
    c = span['color']; rgb = ((c>>16&255)/255, (c>>8&255)/255, (c&255)/255)
    page.insert_text((center-width/2, span['origin'][1]), label,
                     fontname='RevisionBold', fontsize=span['size'], color=rgb)
doc.save(OUT/'figure1_response_concept.pdf', garbage=4, deflate=True)
doc.close()

NS = '{http://www.w3.org/2000/svg}'
font = fitz.Font('helv')
for name in ('figure5_teacher_comparison.svg', 'figure6_source_edited.svg'):
    baseline = BASE/name
    if not baseline.exists():
        baseline.write_bytes((OUT/name).read_bytes())
    tree = etree.parse(str(baseline))
    for node in tree.findall('.//'+NS+'text'):
        old = ''.join(node.itertext())
        replacements = {
            'S9': 'SA-Student',
            'S9 − human': 'SA-Student − human',
            'Raw': 'Predicted', 'Screened': 'Adjusted', 'Observed': 'Simulated',
            '(a) S9 baseline PT share': '(a) SA-Student baseline PT use',
            '(b) S9 delay response': '(b) SA-Student delay response',
            'Argmax + fallback': 'Deterministic',
            'Sampling + fallback': 'Stochastic',
            'Screened sampling': 'Constrained stochastic',
            'Executed − raw response (pp)': 'Simulated − predicted response (pp)',
            'Link-speed execution': 'Network-derived speeds',
            'Mode speed caps': 'Mode-specific speed caps',
        }
        new = replacements.get(old, old)
        if new == old: continue
        assert len(node) == 0, 'Unexpected multi-span label: '+old
        node.text = new
        # Recenter the rotated stage labels above the same tick locations.
        trans = node.get('transform','')
        match = re.fullmatch(r'translate\(([-.\d]+) ([-.\d]+)\) rotate\(-24\)', trans)
        if match:
            dw = font.text_length(new, fontsize=9) - font.text_length(old, fontsize=9)
            x,y=map(float,match.groups())
            node.set('transform',f'translate({x-dw*math.cos(math.radians(24))/2:.6f} '
                                 f'{y+dw*math.sin(math.radians(24))/2:.6f}) rotate(-24)')
    tree.write(str(OUT/name), encoding='utf-8', xml_declaration=True)
    if name.startswith('figure5'):
        cairosvg.svg2pdf(url=str(OUT/name), write_to=str((OUT/name).with_suffix('.pdf')))
print('Figure 1 paired-state label and Figure 5/6 display labels revised; data unchanged.')
