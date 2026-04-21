#!/usr/bin/env python3
"""EBB Wave-1 Mac launcher — EUR fade_long VT-sweep, every-tick MT5.

4 prefixes in parallel: VT=[0.30, 0.50, 0.70, 1.0]
Goal: discover whether gold-trained ONNX vol model gates anything sensibly on EUR
features. VT=1.0 is control (gate effectively disabled — accept all signals).

Period: 2020.01.03 -> 2026.04.10
Symbol: EURUSD
Strategy: fade_long (EntryMode=4) — matches WG2 #1 "NY Reversal Fade"
Session: 13-20 (NY) — most active for EUR
SL_ATR_Mult: 0.5 (looser than gold's 0.20 — EUR has ~1-pip spread vs 18-pt gold)

NOTE: this is a DISCOVERY scan, not a validation. Survivors must pass
null-test + multi-instrument + walk-forward before any candidate status.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HOME = Path.home()
WINE = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")
EXPERT = "GoldBigBrain\\GBB_Core"
OUT_ROOT = HOME / "eurobigbrain"

VT_BY_PREFIX = {1: 0.30, 2: 0.50, 3: 0.70, 4: 1.00}
MAGIC_BASE = 20260422000

BASE_PARAMS = {
    "EntryMode": 4,
    "RiskPercent": 0.6,
    "SL_ATR_Mult": 0.50,
    "TP_ATR_Mult": 2.0,
    "BracketOffset": 0.3,
    "BracketBars": 3,
    "MaxTradesPerDay": 20,
    "DailyLossCapPct": 5.0,
    "SessionStart": 13,
    "SessionEnd": 20,
    "MaxLotSize": 1.0,
    "EnableBreakEven": True,
    "BE_ATR_Mult": 0.5,
    "EnableTrailing": True,
    "Trail_ATR_Mult": 0.3,
    "EnableTimeStop": True,
    "MaxHoldBars": 12,
    "AsianStart": 0,
    "AsianEnd": 7,
    "BreakoutBars": 20,
    "VolSpikeThresh": 2.0,
    "FadeLongRSI": 40.0,
    "FadeShortRSI": 65.0,
    "EmaFastPeriod": 8,
    "EmaSlowPeriod": 21,
    "ML_LongThreshold": 0.65,
    "ML_ShortThreshold": 0.35,
    "ML_EntryOffsetATR": 0.10,
    "OR_OpenHour": 8,
    "EnableTrendFilter": False,
}

FROM_DATE = "2020.01.03"
TO_DATE = "2026.04.10"

def fmt(v):
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0"); return s + "0" if s.endswith(".") else s
    return str(v)

def build_ini(params, ini_path, report_name):
    lines = ["[Tester]", "Expert=" + EXPERT, "Symbol=EURUSD", "Period=M5",
             "Model=8", "Optimization=0", "FromDate=" + FROM_DATE, "ToDate=" + TO_DATE,
             "ForwardMode=0", "Deposit=1000", "Currency=USD", "Leverage=500",
             "ExecutionMode=0", "Visual=0", "ShutdownTerminal=1", "ReplaceReport=1",
             "Report=" + report_name, "", "[TesterInputs]"]
    for k, v in sorted(params.items()):
        fv = fmt(v); lines.append(k + "=" + fv + "||" + fv + "||0||" + fv + "||N")
    ini_path.write_text("\n".join(lines) + "\n")

def fire_one(prefix_id, vt):
    label = f"EBB_W1_VT{int(vt*100):03d}"
    prefix = HOME / f"mt5_prefix_{prefix_id}"
    mt5_dir = prefix / "drive_c/Program Files/MetaTrader 5"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = dict(BASE_PARAMS)
    params["VolThreshold"] = vt
    params["MagicNumber"] = MAGIC_BASE + int(vt * 100)
    ini = mt5_dir / f"ebb_w1_{label}.ini"
    report = f"ebb_w1_{label}_{stamp}.htm"
    build_ini(params, ini, report)
    ini_wine = f"C:\\Program Files\\MetaTrader 5\\ebb_w1_{label}.ini"
    cmd_path = Path(f"/tmp/ebb_w1_{label}_{stamp}.command")
    log_stdout = OUT_ROOT / f"{label}_{stamp}.out"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cmd_path.write_text(
        '#!/bin/bash\n'
        'export WINEPREFIX="' + str(prefix) + '"\n'
        'export WINEDLLOVERRIDES="mmdevapi=d;winepulse.drv=d;winecoreaudio.drv=d"\n'
        'export WINE_NO_AUDIO=1\n'
        'caffeinate -i -s "' + str(WINE) + '" "C:\\Program Files\\MetaTrader 5\\terminal64.exe" "/config:' + ini_wine + '" > "' + str(log_stdout) + '" 2>&1\n'
        'exit 0\n'
    )
    cmd_path.chmod(0o755)
    subprocess.run(["open", str(cmd_path)], capture_output=True)
    return {"label": label, "prefix": prefix_id, "vt": vt,
            "report_abs_path": str(mt5_dir / report),
            "magic": params["MagicNumber"], "ini": str(ini),
            "launched_at": datetime.now().isoformat()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true")
    args = ap.parse_args()
    if not args.fire:
        print("Pass --fire to launch all 4 Mac prefixes")
        sys.exit(2)
    out = []
    for pid, vt in VT_BY_PREFIX.items():
        info = fire_one(pid, vt)
        out.append(info)
        print(f"[FIRED] prefix{pid} VT={vt} magic={info['magic']}")
        time.sleep(2.0)
    manifest = OUT_ROOT / f"wave1_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[MANIFEST] {manifest}")

if __name__ == "__main__":
    main()
