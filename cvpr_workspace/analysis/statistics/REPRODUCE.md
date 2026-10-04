# Historical closure analysis: reproduction scope

This directory retains the statistical implementations used during the September
2026 analysis. Some functions are still imported by later analyses. The public
release does not contain the complete input and validation environment needed to
replay the old closure workflow.

`finalize_closure.py` and `seal_closure.py` are retired entry points. They stop with
exit code 2 before importing analysis code, writing outputs or claiming a result
has passed validation. Their exact original source is retained in the
[author-workflow archive](../../../archive/development_history/author_workflows/README.md).
The archive is historical provenance, not an executable reproduction guide.

Use the current [reproduction guide](../../../docs/REPRODUCIBILITY.md) to choose a
supported task and check its data/runtime requirements. For a read-only check of
documentation links and retained artifact hashes, run from the repository root:

```bash
python scripts/check_repository_docs.py
```

This command does not rerun the historical closure statistics or certify their
original author-workspace validation. Research Teacher prompts, numerical
results and checkpoint files are unchanged by the entry-point maintenance.
