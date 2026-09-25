# AIT manuscript: narrative and evidence revision, 25 September 2026

The active manuscript is `cas-sc-template.tex` / `cas-sc-template.pdf`; the article now includes Appendices A–D. `supplement.tex` / `supplement.pdf` retains detailed reproduction material. This revision incorporates the completed controlled, new-synthetic-reference and physical-supply experiments pinned to research commit `ba7c5d304b46b46f8f74c451cda2825a3916bf6d`.

## Rebuild

```bash
python scripts/make_revision_figures.py
python build.py --clean
```

Figure generation requires numpy, pandas and matplotlib. The TeX build requires pdfLaTeX and BibTeX. New plots are provided as vector PDF and editable-text SVG, with PNG previews. Their inputs and source hashes are in `data_tables/revision20260925/`; older survey aggregate values retain their published precision. Figure regeneration makes no API calls.

## Reading order

- Main article: problem, method and independent units, response-learning boundary, human disagreements, and physical versus perceived service changes.
- Appendix A: questionnaire tasks, sample flow, input interpretation and sensitivity.
- Appendix B: model roles, exact objectives, loss geometry and optimization sensitivity.
- Appendix C: supporting comparison tables, assignment interpretation and speed assumptions.
- Appendix D: new-persona protocol, service metadata and bounded cost accounting.
- Supplement: field dictionary, detailed source/seed tables, repeated-query diagnostics and reproduction inventory.

The manuscript preserves the author-supplied consent statement and states that institutional approval was not obtained. It distinguishes historical official Pro targets, the later official Pro survey reference and the new OpenCode Go Flash reference. No new human repair validation is claimed. API values are usage-based estimates, not invoices.

The source before this narrative revision is backed up locally at `../revision_response_20260925/before_narrative_revision_20260925.zip`. Earlier editorial notes are historical records. The mutation helper `scripts/revise_narrative.py` was used once against that backup and is not a build dependency; routine rebuilding must use the commands above.

See `editorial_notes/REVISION_NARRATIVE_CN.md` for the delivery report and `editorial_notes/NARRATIVE_REVISION_PLAN_CN.md` for the evidence-placement decisions. The research repository is private; participant-level records and full network dependencies have additional access requirements. The original `AIT` directory has not been modified.

Figure 1 and its caption are preserved from the author-provided manuscript. The unused alternative conceptual diagram is retained only as an editorial asset.
