#!/usr/bin/env python
"""Phase B.5C step 3: bracket the operating point with flowCapacityFactor.

The population is identical across sample-factor variants (same seed), so the
scenario is REUSED: copy the baseline_sample scenario, rewrite the qsim
flow/storage capacity factors, rerun MATSim and collect metrics.

Usage:
    python scripts/singapore/run_b5c_factor_sweep.py --factors 0.2,0.3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traveler_distillation.singapore.scenario_metrics import load_link_attrs, collect_metrics

_TMP = "C:/Users/xuan1/AppData/Local/Temp"
_CP = f"{_TMP}/matsim_run;{_TMP}/matsim_rel/matsim-2026.0.jar;{_TMP}/matsim_rel/libs/*"
_NETWORK = "data/singapore/transit/network_with_transit.xml"


def _rewrite_factors(sdir: Path, flow: float, storage: float) -> None:
    cfg = (sdir / "config.xml").read_text(encoding="utf-8")
    import re
    cfg = re.sub(r'(<param name="flowCapacityFactor" value=")[\d.]+(")', rf"\g<1>{flow}\g<2>", cfg)
    cfg = re.sub(r'(<param name="storageCapacityFactor" value=")[\d.]+(")', rf"\g<1>{storage}\g<2>", cfg)
    (sdir / "config.xml").write_text(cfg, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default="0.2,0.3")
    ap.add_argument("--src", default="outputs/singapore_phase_b5/calibration/baseline_sample")
    ap.add_argument("--out", default="outputs/singapore_phase_b5/calibration")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    links = load_link_attrs(_NETWORK)
    results = json.loads((out / "calibration_results.json").read_text(encoding="utf-8"))

    for fstr in args.factors.split(","):
        f = float(fstr.strip())
        name = f"sample{fstr.strip().replace('.', '_')}"
        sdir = out / f"baseline_{name}"
        if sdir.exists() and (sdir / "java_run.log").exists():
            print(f"[skip] {name} exists")
            continue
        if sdir.exists():
            shutil.rmtree(sdir)
        shutil.copytree(src, sdir, ignore=shutil.ignore_patterns("output", "java_run.log"))
        _rewrite_factors(sdir, f, f)
        print(f"===== factor {f} =====", flush=True)
        with open(sdir / "java_run.log", "w", encoding="utf-8", errors="replace") as fh:
            t0 = time.time()
            proc = subprocess.run(
                ["java", "-Xmx8g", "-cp", _CP, "RunMatsimPreloaded", "config.xml"],
                cwd=str(sdir), stdout=fh, stderr=subprocess.STDOUT, timeout=10800,
            )
            wall = time.time() - t0
        record = {
            "scheme": name, "flow_capacity_factor": f, "storage_capacity_factor": f,
            "matsim_exit_code": proc.returncode, "runtime_seconds": round(wall, 1),
            "metrics": collect_metrics(sdir / "output" / "ITERS" / "it.0" / "0.events.xml.zst", links),
        }
        results[name] = record
        (out / "calibration_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        m = record["metrics"]
        print(json.dumps({"factor": f, "exit": proc.returncode,
                          "slow": m.get("network_congestion_slow_share"),
                          "links15s": m.get("share_links_delay_gt_15s"),
                          "carT": m.get("car_mean_travel_time_min"),
                          "failed": m.get("failed_trips"), "stuck": m.get("stuck_and_abort")},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
