#!/usr/bin/env python
"""E1 finalize — regenerate the E1 report (with T4) and refresh the archive.

Run AFTER scripts/singapore/run_phase_c_mnl.py completes:
    python scripts/finalize_e1.py [--paper-repo <path>]

Does: make_e1_report.py -> E1_REPORT.md; copies E1_REPORT.md,
PHASE_C_MNL_REPORT.md, phase_c_mnl_records.json into the paper-repo archive
<paper-repo>/data_report/10_E1_MNL/; prints gate summary + key T4 rows.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER = Path(r"C:\Users\xuan1\Desktop\UOL学习\蒸馏出行意图paper")
SCENARIOS = ["C0_baseline", "C1_heavy_rain", "C2_fare_increase",
             "C3_transit_delay", "C4_road_disruption", "C5_joint_rain_delay"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-repo", type=str, default=str(DEFAULT_PAPER))
    args = ap.parse_args()
    paper = Path(args.paper_repo)
    archive = paper / "data_report" / "10_E1_MNL"

    records_path = _ROOT / "outputs" / "singapore_phase_c_mnl" / "phase_c_mnl_records.json"
    if not records_path.exists():
        print("FAIL: Phase C MNL records not found — S4 has not completed.")
        return 1

    # 1) regenerate report
    r = subprocess.run([sys.executable, str(_ROOT / "scripts" / "make_e1_report.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout.strip())
    if r.returncode != 0:
        print("make_e1_report failed:", r.stderr[-2000:])
        return r.returncode

    # 2) archive refresh
    archive.mkdir(parents=True, exist_ok=True)
    for f in ("E1_REPORT.md",):
        shutil.copy2(_ROOT / "outputs" / "e1_mnl" / f, archive / f)
    for f in ("PHASE_C_MNL_REPORT.md", "phase_c_mnl_records.json"):
        src = _ROOT / "outputs" / "singapore_phase_c_mnl" / f
        if src.exists():
            shutil.copy2(src, archive / f)
    print(f"archive refreshed: {archive}")

    # 3) gate summary + key rows for manual verification
    records = {r["scenario"]: r for r in json.loads(records_path.read_text(encoding="utf-8"))}
    print("\n=== gate summary ===")
    for s in SCENARIOS:
        r = records[s]
        print(f"  {s}: gate={r['gate']['pass']} exit={r.get('matsim_exit_code')} "
              f"stuck={ (r.get('metrics') or {}).get('stuck_persons', '—') }")
    print("\n=== student mode shares (MNL) ===")
    for s in SCENARIOS:
        r = records[s]
        d = r["decisions"]["student_mode_counts"]
        n = r["n_agents"]
        print(f"  {s}: " + " ".join(f"{m}={100 * d.get(m, 0) / n:.1f}%" for m in ("car", "pt", "bike", "walk")))
    print("\n=== C1-C5 vs C0 (MNL) ===")
    r0 = records["C0_baseline"]
    for s in SCENARIOS[1:]:
        r = records[s]
        d0, d = r0["decisions"]["student_mode_counts"], r["decisions"]["student_mode_counts"]
        m0, m = r0["metrics"], r["metrics"]
        dpt = 100 * (d.get("pt", 0) / r["n_agents"] - d0.get("pt", 0) / r0["n_agents"])
        dcar = 100 * (d.get("car", 0) / r["n_agents"] - d0.get("car", 0) / r0["n_agents"])
        print(f"  {s}: dpt={dpt:+.1f}pp dcar={dcar:+.1f}pp "
              f"dboardings={m['pt_boardings'] - m0['pt_boardings']:+d} dVKT={m['car_vkt_km'] - m0['car_vkt_km']:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
