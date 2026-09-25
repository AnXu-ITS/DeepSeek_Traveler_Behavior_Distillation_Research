"""Verify documentation links, declared language scope and copied evidence."""
from pathlib import Path
import hashlib
import json
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CHINESE_AUTHOR_REPORTS = {
    "evidence/experiments_20260925/EXPERIMENT_REPORT_CN.md",
    "outputs/revision_20260925/EXPERIMENT_REPORT_CN.md",
}

def main():
    errors = []
    docs = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts and ".venv" not in p.parts]
    for path in docs:
        text = path.read_text(encoding="utf-8-sig")
        # The author explicitly requested a Chinese experiment report. This
        # exemption is exact-path and language-only; links/fences stay checked.
        if path.relative_to(ROOT).as_posix() not in CHINESE_AUTHOR_REPORTS and re.search(r"[\u3400-\u9fff]", text):
            errors.append(f"Untranslated Chinese: {path.relative_to(ROOT)}")
        if len(re.findall(r"^\s*```", text, re.M)) % 2:
            errors.append(f"Unbalanced code fence: {path.relative_to(ROOT)}")
        for match in re.finditer(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text):
            link = match.group(1).strip().strip("<>").split(' "')[0]
            if link.startswith(("http:", "https:", "mailto:", "#", "data:")):
                continue
            target = unquote(link.split("#")[0])
            if not (path.parent / target).exists():
                errors.append(f"Missing link in {path.relative_to(ROOT)}: {link}")
    manifest = json.loads((ROOT / "evidence/paper_20260924/artifact_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = ROOT / artifact["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            errors.append(f"Artifact checksum mismatch: {artifact['path']}")
    print(json.dumps({"markdown_files": len(docs), "declared_chinese_reports": sorted(CHINESE_AUTHOR_REPORTS), "artifacts": len(manifest["artifacts"]),
                      "passed": not errors, "errors": errors}, indent=2))
    return int(bool(errors))

if __name__ == "__main__":
    sys.exit(main())
