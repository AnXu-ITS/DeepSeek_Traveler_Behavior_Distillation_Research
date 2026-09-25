"""Build the article and supplement, including their cross-document references."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DOCUMENTS = ('supplement', 'cas-sc-template')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', action='store_true', help='Regenerate all auxiliary labels and bibliography files.')
    args = parser.parse_args()
    latex = shutil.which('pdflatex')
    bibtex = shutil.which('bibtex') or shutil.which('bibtex.original') or shutil.which('bibtex8')
    if not latex or not bibtex:
        print('Install TeX Live or MiKTeX with pdfLaTeX and BibTeX.', file=sys.stderr)
        return 1
    if args.clean:
        for name in DOCUMENTS:
            for ext in ('aux', 'bbl', 'blg', 'log', 'out', 'abs', 'fls', 'fdb_latexmk', 'lof', 'lot'):
                (ROOT / f'{name}.{ext}').unlink(missing_ok=True)
    def run(command: list[str]) -> None:
        subprocess.run(command, cwd=ROOT, check=True)
    def compile_document(name: str) -> None:
        run([latex, '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', name + '.tex'])
    try:
        for name in DOCUMENTS:
            compile_document(name)
        run([bibtex, 'cas-sc-template'])
        for _ in range(3):
            for name in DOCUMENTS:
                compile_document(name)
    except subprocess.CalledProcessError as exc:
        print(f'Build failed (exit {exc.returncode}); see the document log.', file=sys.stderr)
        return exc.returncode or 1
    issues = []
    for name in DOCUMENTS:
        log = (ROOT / f'{name}.log').read_text(errors='replace')
        for marker in ('There were undefined references', 'Citation `', 'multiply defined',
                       'Overfull ', 'Float too large', 'destination with the same identifier'):
            if marker in log:
                issues.append(f'{name}: {marker}')
    if issues:
        print('\n'.join(issues), file=sys.stderr)
        return 2
    print('Build complete: cas-sc-template.pdf and supplement.pdf.')
    print('No unresolved references/citations, overfull boxes or oversized floats.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
