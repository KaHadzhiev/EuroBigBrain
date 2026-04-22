#!/usr/bin/env python3
"""EBB 6yr full pipeline runner — runs QA FIRST, then steps 1-5.

If QA fails, exits immediately (no wasted compute).
If any step fails, halts and reports.

Usage:
  python run_full_pipeline.py               # run all
  python run_full_pipeline.py --qa-only     # just check readiness
  python run_full_pipeline.py --skip-qa     # bypass QA (DEBUG ONLY)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / "python"

STEPS = [
    ("Step 1 — Merge Dukascopy ticks",      "merge_dukascopy_eurusd.py"),
    ("Step 2 — Aggregate to M5 bars",       "dukascopy_to_m5_bars.py"),
    ("Step 3 — Build 23-feature parquet",   "build_eurusd_features_6yr.py"),
    ("Step 4 — Train LightGBM 4yr/2yr",     "train_eurusd_lightgbm_6yr.py"),
    ("Step 5 — Python sim threshold sweep", "sim_eurusd_6yr.py"),
]


def run(script, label):
    print(f"\n--- {label} ---")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(PY / script)], cwd=REPO)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"  [FAIL] {script} exited {r.returncode} after {dt:.1f}s -- halting pipeline")
        sys.exit(r.returncode)
    print(f"  [OK] {script} done in {dt:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-only", action="store_true", help="Only run QA")
    ap.add_argument("--skip-qa", action="store_true", help="Skip QA (DEBUG)")
    args = ap.parse_args()

    total_t0 = time.time()

    if not args.skip_qa:
        print("=== RUNNING QA FIRST ===")
        r = subprocess.run([sys.executable, str(PY / "pipeline_qa.py")], cwd=REPO)
        if r.returncode != 0:
            print("\n[RED] QA FAILED. Pipeline will NOT run. Fix configs and retry.")
            sys.exit(r.returncode)
        print()

    if args.qa_only:
        print("QA-only mode — exiting.")
        sys.exit(0)

    for label, script in STEPS:
        run(script, label)

    total_dt = time.time() - total_t0
    print(f"\n[OK] PIPELINE COMPLETE in {total_dt:.1f}s")
    print("\nNext steps (manual):")
    print("  - Deploy new ONNX to MT5 (models/eur_tb_h10_6yr.onnx -> MQL5/Files/)")
    print("  - Recompile EBB_TripleBarrier EA")
    print("  - Run run_mt5_thr_sweep.py for MT5 every-tick baseline")
    print("  - Run calibrate_sim_vs_mt5.py for isotonic calibration")


if __name__ == "__main__":
    main()
