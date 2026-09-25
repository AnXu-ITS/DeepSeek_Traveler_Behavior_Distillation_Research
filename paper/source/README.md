# AIT response-fidelity revision — 25 September 2026

**Title:** Beyond Static Imitation: Auditing LLM-Derived Traveler Responses for Transport Simulation

This is an evidence-limited editorial revision of the author-supplied `AIT_Response_Fidelity_Revision.zip`. Existing scientific results, figure assets and original numerical result tables are preserved. No new training, Teacher requests, survey collection or MATSim simulations were run.

The revision inserts the author's ethics/consent statement, identifies official DeepSeek V4 Pro supply, corrects the superseded routing assumption using official API documentation, records the existing limited compositional holdout, and adds a retrospective API cost table plus a conditional lifecycle break-even formulation.

## Files

- `cas-sc-template.pdf` and `supplement.pdf`: compiled revised article and supplement.
- `sections/`, `generated/`, `figures/`, `references.bib`: editable article sources and figure assets.
- `reproduction/cost_audit/`: aggregate-only token ledger, reproducible offline cost script and planning scenarios. Estimates are not provider invoices.
- `editorial_notes/revision_20260925.patch`: changes from the supplied ZIP, where present in the full local delivery.

Earlier editorial notes in the full local package are historical inputs, not the status of this revision. The original experimental repository base remains `4e67286ce72680a9e3a916a3fb7e97b1548b53f7`. New experiments remain proposals requiring an author decision; no future result is used as current evidence.

## Build

```bash
python build.py --clean
```

This uses pdfLaTeX and BibTeX and resolves cross-document references. Source files and both compiled PDFs are supplied. The limited revisions are complete; the broader stronger-paper roadmap and submission readiness remain open.

The compiled PDFs for this repository source tree are in [the versioned PDF directory](../revision_20260925/). Full local editorial notes are delivered separately from this scientific source subset.
