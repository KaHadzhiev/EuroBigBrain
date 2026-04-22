#!/usr/bin/env python3
"""EBB Wave-7 — GBP null test (HARD RULE: real ≥ 5× null required).

Wave-6 verdict: GBPUSD @ VT=0.50/SL=0.30/S1317 = PF=1.28 / 80 trades / DD=6.96%.
Lowering VT catastrophically wipes (1200-1600 trades = 99% DD).
The high-VT filter IS the edge.

Wave-7 tests if the (S1317) session is special by shuffling to other 4-hour windows:
  S0913 (London-Asia overlap)
  S1014 (London-AM)
  S1115 (London-mid)
  S1620 (NY-PM)
Same VT=0.50, SL=0.30, all other params identical to W6 winner.

Pass criteria: real PF (1.28) ≥ 5× null median PF.
If 1+ null beats real → reject as session-rotational noise.

4 cfgs = 1 batch on 4 prefixes ~3-4 min.
"""
from __future__ import annotations
import json, subprocess, time
from datetime import datetime
from pathlib import Path

HOME = Path.home()
WINE = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")
EXPERT = "GoldBigBrain\\GBB_Core"
OUT_ROOT = HOME / "eurobigbrain" / "wave7"
SYMBOL = "GBPUSD"

# Real session = (13, 17). Nulls = same 4-hour width, different position.
NULL_SESSIONS = [(9, 13), (10, 14), (11, 15), (16, 20)]

MAGIC_BASE = 20260422700

BASE_PARAMS = {
    "EntryMode": 5,
    "RiskPercent": 0.6, "TP_ATR_Mult": 2.0, "SL_ATR_Mult": 0.30, "VolThreshold": 0.50,
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

def build_ini(params, ini_path, report_name):
    lines = ["[Tester]", "Expert=" + EXPERT, "Symbol=" + SYMBOL, "Period=M5",
             "Model=8", "Optimization=0", "FromDate=" + FROM_DATE, "ToDate=" + TO_DATE,
             "ForwardMode=0", "Deposit=1000", "Currency=USD", "Leverage=500",
             "ExecutionMode=0", "Visual=0", "ShutdownTerminal=1", "ReplaceReport=1",
             "Report=" + report_name, "", "[TesterInputs]"]
    for k, v in sorted(params.items()):
        fv = fmt(v); lines.append(k + "=" + fv + "||" + fv + "||0||" + fv + "||N")
    ini_path.write_text("\n".join(lines) + "\n")

def fire_one(prefix_id, ss, se):
    label = f"W7_null_GBP_S{ss:02d}{se:02d}_VT050_SL030"
    prefix = HOME / f"mt5_prefix_{prefix_id}"
    mt5_dir = prefix / "drive_c/Program Files/MetaTrader 5"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = dict(BASE_PARAMS)
    params["SessionStart"] = ss; params["SessionEnd"] = se
    params["MagicNumber"] = MAGIC_BASE + ss*100 + se
    ini = mt5_dir / f"ebb_w7_{label}.ini"
    report_name = f"ebb_w7_{label}_{stamp}.htm"
    build_ini(params, ini, report_name)
    ini_wine = f"C:\\Program Files\\MetaTrader 5\\ebb_w7_{label}.ini"
    cmd_path = Path(f"/tmp/ebb_w7_{label}_{stamp}.command")
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
            "session": [ss, se],
            "report_abs_path": str(mt5_dir / report_name), "fired_at": datetime.now().isoformat()}

def main():
    print(f"[wave7] {len(NULL_SESSIONS)} GBP null sessions")
    fired = []
    for pfx_idx, (ss, se) in enumerate(NULL_SESSIONS):
        prefix_id = pfx_idx + 1
        info = fire_one(prefix_id, ss, se)
        fired.append(info)
        print(f"[fire] prefix {prefix_id} <- W7_null_GBP_S{ss:02d}{se:02d}")
        time.sleep(2.0)
    deadline = time.time() + 600
    while time.time() < deadline:
        done = sum(1 for f in fired if Path(f["report_abs_path"]).exists())
        print(f"[wait] {done}/{len(fired)} HTMs present")
        if done == len(fired): break
        time.sleep(15)
    manifest = OUT_ROOT / f"wave7_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps(fired, indent=2, default=str))
    print(f"\n[MANIFEST] {manifest}")

if __name__ == "__main__":
    main()
