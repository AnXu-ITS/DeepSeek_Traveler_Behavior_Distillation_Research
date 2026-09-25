"""Rebuild edited publication figures; no training, API calls or simulation.

Compilation does not require this optional script. LibreOffice and the fonts
noted in README.md are required for Figure 2 export.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
for name in ('edit_publication_figures.py','make_figure3_response_timing.py',
             'make_figure4_bars.py','assemble_execution_figure.py','edit_figure2_template.py'):
    subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,check=True)
lo=shutil.which('soffice') or shutil.which('libreoffice')
if not lo:
    raise SystemExit('Figure 2 PPTX was rebuilt. Export it as PDF in PowerPoint/LibreOffice, '
                     'saving figures/narrative/figure2_method.pdf.')
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    subprocess.run([lo,'-env:UserInstallation='+(tmp/'profile').as_uri(),
                    '--headless','--convert-to','pdf','--outdir',str(tmp),
                    str(ROOT/'figures/narrative/figure2_method_template.pptx')],check=True)
    result=tmp/'figure2_method_template.pdf'
    if not result.exists(): raise RuntimeError('LibreOffice did not produce the Figure 2 PDF.')
    shutil.copy2(result,ROOT/'figures/narrative/figure2_method.pdf')
print('All edited publication figures regenerated from supplied sources and summaries.')
