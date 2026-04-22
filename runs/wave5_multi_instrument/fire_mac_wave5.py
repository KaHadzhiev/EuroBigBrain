#!/usr/bin/env python3
"""EBB Wave-5 — multi-instrument decisive test (M113-killer rule).

Wave-4 verdict: NO atr_brk EUR config hits deploy gate (PF<=1.16 ceiling).
Wave-5 tests if the atr_brk archetype generalizes to OTHER instruments at all.
If it fails on GBP/XAG/JPY = third confirmation EBB direction needs pivot.

Configs:
  1. W4_atr_S1317_VT050_SL050 on GBPUSD  (top EUR winner)
  2. W4_atr_S1317_VT050_SL050 on XAGUSD
  3. W4_atr_S1317_VT050_SL050 on USDJPY
  4. W4_atr_S1317_VT050_SL030 on GBPUSD  (#2 EUR winner cross-check)

4 cfgs = 1 batch on 4 prefixes ~4 min.
Pass criteria: PF>=1.0 on ANY non-EUR instrument with trades>=50.
If all fail = pivot recommendation in morning report.
"""
from __future__ import annotations
import json, subprocess, time
from datetime import datetime
from pathlib import Path

HOME = Path.home()
WINE = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")
EXPERT = "GoldBigBrain\\GBB_Core"
OUT_ROOT = HOME / "eurobigbrain" / "wave5"

JOBS = [
    {"label": "W5_atr_GBPUSD_S1317_VT050_SL050", "symbol": "GBPUSD", "ss": 13, "se": 17, "vt": 0.50, "sl": 0.50},
    {"label": "W5_atr_USDCHF_S1317_VT050_SL050", "symbol": "USDCHF", "ss": 13, "se": 17, "vt": 0.50, "sl": 0.50},
    {"label": "W5_atr_USDJPY_S1317_VT050_SL050", "symbol": "USDJPY", "ss": 13, "se": 17, "vt": 0.50, "sl": 0.50},
    {"label": "W5_atr_GBPUSD_S1317_VT050_SL030", "symbol": "GBPUSD", "ss": 13, "se": 17, "vt": 0.50, "sl": 0.30},
]

MAGIC_BASE = 20260422500

BASE_PARAMS = {
    "EntryMode": 5,
    "RiskPercent": 0.6, "TP_ATR_Mult": 2.0,
    "BracketOffset": 0.3, "BracketBars": 3, "MaxTradesPerDay": 20, "DailyLossCapPct": 5.0,
    "MaxLotSize": 1.0, "EnableBreakEven": True, "BE_ATR_Mult": 0.5, "EnableTrailing": True,
    "Trail_ATR_Mult": 0.3, "EnableTimeStop": True, "MaxHoldBars": 12,
    "AsianStart": 0, "AsianEnd": 7, "BreakoutBars": 20, "VolSpikeThresh": 2.0,
    "FadeLongRSI": 40.0, "FadeShortRSI": 65.0, "EmaFastPeriod": 8, "EmaSlowPeriod": 21,
    "ML_LongThreshold": 0.65, "ML_ShortThreshold": 0.35, "ML_EntryOffsetATR": 0.10,
    "OR_OpenHour": 8, "EnableTrendFilter": False,
}
FROM_DATE = "2020.01.03"
TO_DATE = "2026.04.10"

def fmt(v):
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0"); return s + "0" if s.endswith(".") else s
    return str(v)

def build_ini(params, ini_path, report_name, symbol):
    lines = ["[Tester]", "Expert=" + EXPERT, "Symbol=" + symbol, "Period=M5",
             "Model=8", "Optimization=0", "FromDate=" + FROM_DATE, "ToDate=" + TO_DATE,
             "ForwardMode=0", "Deposit=1000", "Currency=USD", "Leverage=500",
             "ExecutionMode=0", "Visual=0", "ShutdownTerminal=1", "ReplaceReport=1",
             "Report=" + report_name, "", "[TesterInputs]"]
    for k, v in sorted(params.items()):
        fv = fmt(v); lines.append(k + "=" + fv + "||" + fv + "||0||" + fv + "||N")
    ini_path.write_text("\n".join(lines) + "\n")

def fire_one(prefix_id, job):
    label = job["label"]
    symbol = job["symbol"]
    prefix = HOME / f"mt5_prefix_{prefix_id}"
    mt5_dir = prefix / "drive_c/Program Files/MetaTrader 5"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = dict(BASE_PARAMS)
    params["SessionStart"] = job["ss"]; params["SessionEnd"] = job["se"]
    params["VolThreshold"] = job["vt"]
    params["SL_ATR_Mult"] = job["sl"]
    params["MagicNumber"] = MAGIC_BASE + prefix_id*1000 + int(job["vt"]*100)*10 + int(job["sl"]*100)
    ini = mt5_dir / f"ebb_w5_{label}.ini"
    report_name = f"ebb_w5_{label}_{stamp}.htm"
    build_ini(params, ini, report_name, symbol)
    ini_wine = f"C:\\Program Files\\MetaTrader 5\\ebb_w5_{label}.ini"
    cmd_path = Path(f"/tmp/ebb_w5_{label}_{stamp}.command")
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
    return {"label": label, "prefix": prefix_id, "symbol": symbol,
            "session": [job["ss"], job["se"]], "vt": job["vt"], "sl": job["sl"],
            "report_abs_path": str(mt5_dir / report_name), "fired_at": datetime.now().isoformat()}

def main():
    print(f"[wave5] {len(JOBS)} multi-instrument jobs")
    fired = []
    for pfx_idx, job in enumerate(JOBS):
        prefix_id = pfx_idx + 1
        info = fire_one(prefix_id, job)
        fired.append(info)
        print(f"[fire] prefix {prefix_id} <- {job['label']}")
        time.sleep(2.0)
    print(f"[fire] all {len(fired)} fired, waiting up to 600s for completion")
    deadline = time.time() + 600
    while time.time() < deadline:
        done = sum(1 for f in fired if Path(f["report_abs_path"]).exists())
        print(f"[wait] {done}/{len(fired)} HTMs present")
        if done == len(fired): break
        time.sleep(15)
    manifest = OUT_ROOT / f"wave5_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps(fired, indent=2, default=str))
    print(f"\n[MANIFEST] {manifest}")

if __name__ == "__main__":
    main()
