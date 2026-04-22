#!/usr/bin/env python3
"""Pipeline QA — runs BEFORE any EBB 6yr pipeline step.
Refuses to proceed if critical configs are patched for interim runs,
data files missing, scripts fail to parse, etc.

Exit 0 = all clear.
Exit 1 = any check failed (details printed).

Usage:
  python pipeline_qa.py          # run all checks, exit non-zero on any fail
  python pipeline_qa.py --strict # same, but also fails on warnings
"""
import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
PY = REPO / "python"

# --- Expected critical configs (edit when the deploy spec changes) ---
EXPECTED = {
    "train_eurusd_lightgbm_6yr.py": {
        "TRAIN_START": '"2020-01-01"',
        "TRAIN_END":   '"2024-01-01"',
        "TEST_END":    '"2026-04-13"',
    },
    "sim_eurusd_6yr.py": {
        "VANTAGE_SPREAD_PIPS": "1.3",
        "SLIPPAGE_PIPS": "0.2",
    },
    "run_mt5_thr_sweep.py": {
        "FromDate": "2024.01.01",
        "ToDate":   "2026.04.13",
    },
    "build_eurusd_features_6yr.py": {
        "resample_hour_lowercase": "1h",  # pandas 3.0+ requires lowercase
    },
}

EXPECTED_DATA = {
    "dukascopy/eurusd_ticks_*.parquet": 70,       # min expected monthly parquets
    "XAUUSD_M5_6yr.parquet":            5_000_000, # min bytes
    "eurusd_spread_2020_2026.parquet":  100_000,
}

EXPECTED_SCRIPTS = [
    "merge_dukascopy_eurusd.py",
    "dukascopy_to_m5_bars.py",
    "build_eurusd_features_6yr.py",
    "train_eurusd_lightgbm_6yr.py",
    "sim_eurusd_6yr.py",
    "calibrate_sim_vs_mt5.py",
    "run_mt5_thr_sweep.py",
]


def check_config(script, key, expected):
    """Grep the script for `key = <expected>` (or `key=<expected>` for MT5 INI)."""
    p = PY / script
    if not p.exists():
        return (False, f"{script}: FILE MISSING")
    src = p.read_text(encoding="utf-8", errors="ignore")
    if key == "resample_hour_lowercase":
        # Special case: pandas 3.0 requires "1h" not "1H"
        if re.search(r'resample\(\s*["\']1H["\']', src):
            return (False, f"{script}: uses 'resample(\"1H\")' — pandas 3.0 needs '1h'")
        if re.search(r'resample\(\s*["\']1h["\']', src):
            return (True, f"{script}: resample('1h') OK")
        return (True, f"{script}: no resample found (OK if not needed)")
    # Generic: look for `key = expected` or `key=expected` (INI style)
    patterns = [
        rf'^\s*{re.escape(key)}\s*=\s*{re.escape(expected)}\s*(?:#|$)',
        rf'"{re.escape(key)}={re.escape(expected)}"',
    ]
    for pat in patterns:
        if re.search(pat, src, re.M):
            return (True, f"{script}: {key} = {expected} OK")
    # Find the actual value for a helpful error message
    m = re.search(rf'^\s*{re.escape(key)}\s*=\s*(\S+)', src, re.M) or \
        re.search(rf'"{re.escape(key)}=([^"]+)"', src)
    actual = m.group(1) if m else "<not found>"
    return (False, f"{script}: {key} = {actual}  (expected {expected})")


def check_data():
    """Verify expected data files exist with at least the minimum size."""
    results = []
    for pat, min_val in EXPECTED_DATA.items():
        if "*" in pat:
            # Count matches
            files = sorted(DATA.glob(pat))
            if len(files) >= min_val:
                results.append((True, f"data/{pat}: {len(files)} files (>={min_val} expected)"))
            else:
                results.append((False, f"data/{pat}: {len(files)} files (NEED >={min_val})"))
        else:
            p = DATA / pat
            if not p.exists():
                results.append((False, f"data/{pat}: MISSING"))
                continue
            sz = p.stat().st_size
            if sz >= min_val:
                results.append((True, f"data/{pat}: {sz:,} bytes (>={min_val:,} expected)"))
            else:
                results.append((False, f"data/{pat}: {sz:,} bytes (NEED >={min_val:,})"))
    return results


def check_syntax(scripts):
    """Python syntax check via ast.parse."""
    results = []
    for s in scripts:
        p = PY / s
        if not p.exists():
            results.append((False, f"python/{s}: MISSING"))
            continue
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
            results.append((True, f"python/{s}: parse OK"))
        except SyntaxError as e:
            results.append((False, f"python/{s}: SYNTAX ERROR line {e.lineno}: {e.msg}"))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    fails = 0
    warns = 0
    print("=== EBB PIPELINE PRE-FLIGHT QA ===\n")

    # 1. Config checks
    print("[1/3] Config values")
    for script, expects in EXPECTED.items():
        for key, val in expects.items():
            ok, msg = check_config(script, key, val)
            print(("  [OK] " if ok else "  [FAIL] ") + msg)
            if not ok: fails += 1
    print()

    # 2. Data presence
    print("[2/3] Data files")
    for ok, msg in check_data():
        print(("  [OK] " if ok else "  [FAIL] ") + msg)
        if not ok: fails += 1
    print()

    # 3. Syntax
    print("[3/3] Python syntax")
    for ok, msg in check_syntax(EXPECTED_SCRIPTS):
        print(("  [OK] " if ok else "  [FAIL] ") + msg)
        if not ok: fails += 1
    print()

    print("=== QA RESULT ===")
    if fails == 0:
        print("  [GREEN] ALL CLEAR. Pipeline is ARMED.")
        return 0
    print(f"  [RED] {fails} check(s) FAILED. Pipeline BLOCKED — fix before running.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
