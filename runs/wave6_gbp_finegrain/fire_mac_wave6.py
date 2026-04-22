#!/usr/bin/env python3
"""EBB Wave-6 — GBP fine-grain after Wave 5 cross-instrument WIN.

Wave-5 verdict: atr_breakout has CLUSTER edge (EUR + GBP, NOT CHF/JPY).
GBPUSD @ VT=0.50/SL=0.30 hit PF=1.28 on FULL 6yr backtest but only 80 trades.
Wave-6 sweeps VT[0.20-0.50] × SL[0.20-0.40] on GBPUSD to find sweet spot
that boosts trade frequency while preserving PF.

9 configs × 4 prefixes = 3 batches ~5-6 min on Mac.

Pass criteria: ANY GBPUSD config with PF>=1.2 AND trades>=200 = real candidate.
"""
from __future__ import annotations
import json, subprocess, time
from datetime import datetime
from pathlib import Path
from itertools import product

HOME = Path.home()
WINE = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")
EXPERT = "GoldBigBrain\\GBB_Core"
OUT_ROOT = HOME / "eurobigbrain" / "wave6"

VTS = [0.20, 0.35, 0.50]
SLS = [0.20, 0.30, 0.40]
SYMBOL = "GBPUSD"
SS, SE = 13, 17

MAGIC_BASE = 20260422600

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
    "SessionStart": SS, "SessionEnd": SE,
}
FROM_DATE = "2020.01.03"
TO_DATE = "2026.04.10"

def fmt(v):
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0"); return s + "0" if s.endswith(".") else s
    return str(v)

def build_ini(params, ini_path, report_name):
    lines = ["[Tester]", "Expert=" + EXPERT, "Symbol=" + SYMBOL, "Period=M5",
             "Model=8", "Optimization=0", "FromDate=" + FROM_DATE, "ToDate=" + TO_DATE,
             "ForwardMode=0", "Deposit=1000", "Currency=USD", "Leverage=500",
             "ExecutionMode=0", "Visual=0", "ShutdownTerminal=1", "ReplaceReport=1",
             "Report=" + report_name, "", "[TesterInputs]"]
    for k, v in sorted(params.items()):
        fv = fmt(v); lines.append(k + "=" + fv + "||" + fv + "||0||" + fv + "||N")
    ini_path.write_text("\n".join(lines) + "\n")

def fire_one(prefix_id, vt, sl):
    label = f"W6_atr_GBP_S{SS:02d}{SE:02d}_VT{int(vt*100):03d}_SL{int(sl*100):03d}"
    prefix = HOME / f"mt5_prefix_{prefix_id}"
    mt5_dir = prefix / "drive_c/Program Files/MetaTrader 5"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = dict(BASE_PARAMS)
    params["VolThreshold"] = vt
    params["SL_ATR_Mult"] = sl
    params["MagicNumber"] = MAGIC_BASE + int(vt*100)*100 + int(sl*100)
    ini = mt5_dir / f"ebb_w6_{label}.ini"
    report_name = f"ebb_w6_{label}_{stamp}.htm"
    build_ini(params, ini, report_name)
    ini_wine = f"C:\\Program Files\\MetaTrader 5\\ebb_w6_{label}.ini"
    cmd_path = Path(f"/tmp/ebb_w6_{label}_{stamp}.command")
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
    return {"label": label, "prefix": prefix_id, "symbol": SYMBOL,
            "session": [SS, SE], "vt": vt, "sl": sl,
            "report_abs_path": str(mt5_dir / report_name), "fired_at": datetime.now().isoformat()}

def main():
    jobs = list(product(VTS, SLS))
    print(f"[wave6] {len(jobs)} GBPUSD jobs")
    all_results = []
    for batch_idx, i in enumerate(range(0, len(jobs), 4)):
        batch = jobs[i:i+4]
        print(f"[batch {batch_idx+1}/{(len(jobs)+3)//4}] firing {len(batch)} configs")
        fired = []
        for pfx_idx, (vt, sl) in enumerate(batch):
            prefix_id = pfx_idx + 1
            info = fire_one(prefix_id, vt, sl)
            fired.append(info)
            time.sleep(2.0)
        deadline = time.time() + 360
        while time.time() < deadline:
            done = sum(1 for f in fired if Path(f["report_abs_path"]).exists())
            if done == len(fired): break
            time.sleep(10)
        all_results.extend(fired)
        print(f"[batch {batch_idx+1}] done={done}/{len(fired)}")
    manifest = OUT_ROOT / f"wave6_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[MANIFEST] {manifest}")

if __name__ == "__main__":
    main()
