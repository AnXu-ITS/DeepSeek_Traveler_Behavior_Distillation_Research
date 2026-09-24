"""Evaluate a controlled checkpoint using its exact archived training source.

The archive is extracted into a temporary directory; the working source tree,
released checkpoint and prepared bundle are left unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=ROOT / "outputs/matched_response_v1/bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    run, bundle, output = args.run.resolve(), args.bundle.resolve(), args.output.resolve()
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    archive = run / "code_snapshot.zip"
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != meta["code_snapshot_sha256"]:
        raise ValueError("Archived source SHA-256 differs from the training record")
    with tempfile.TemporaryDirectory(prefix="traveler-source-") as tmp:
        target = Path(tmp).resolve()
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                path = (target / member.filename).resolve()
                if not path.is_relative_to(target) or member.is_dir():
                    if member.is_dir() and path.is_relative_to(target):
                        continue
                    raise ValueError("Unsafe archive path")
                if not (member.filename.startswith("src/") or member.filename == "scripts/matched_response.py"):
                    raise ValueError("Unexpected file in source archive")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(handle.read(member))
        config = target / "training_config.yaml"
        config.write_text(yaml.safe_dump(meta["config"]), encoding="utf-8")
        cmd = [sys.executable, str(target / "scripts/matched_response.py"), "--config", str(config),
               "evaluate", "--bundle", str(bundle), "--run", str(run), "--split", args.split,
               "--output", str(output)]
        return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())
