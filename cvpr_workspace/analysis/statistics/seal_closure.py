"""Retired author-workspace closure entry point.

The historical implementation is retained under
archive/development_history/author_workflows/cvpr_workspace/analysis/statistics/.
It required validation tools, tests and manuscript files outside this release.
"""
import sys


def main():
    print(
        "This historical closure entry point is retired: its complete validation "
        "environment is not included in this release. No results were generated "
        "or validated. See cvpr_workspace/analysis/statistics/REPRODUCE.md and "
        "docs/REPRODUCIBILITY.md. For the supported read-only repository check, "
        "run: python scripts/check_repository_docs.py",
        file=sys.stderr,
    )
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
