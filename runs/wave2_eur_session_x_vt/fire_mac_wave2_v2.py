#!/usr/bin/env python3
"""EBB Wave-2 v2 — batched firing via `open` (same pattern as Wave 1).

Fires 4 parallel .command files via `open`, polls for HTM reports, fires next batch.
5 batches x 4 parallel = 20 configs x ~2min = ~10 min total.
"""
from __future__ import annotations
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from itertools import product

HOME = Path.home()
WINE = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")
EXPERT = "GoldBigBrain\\GBB_Core"
OUT_ROOT = HOME / "eurobigbrain" / "wave2"

SESSIONS = [(7, 12), (13, 17), (14, 21), (8, 20)]
VTS = [0.20, 0.35, 0.50, 0.70, 1.00]
MAGIC_BASE = 20260422200

BASE_PARAMS = {
    "EntryMode": 4, "RiskPercent": 0.6, "SL_ATR_Mult": 0.50, "TP_ATR_Mult": 2.0,
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
    lines = ["[Tester]", "Expert=" + EXPERT, "Symbol=EURUSD", "Period=M5",
             "Model=8", "Optimization=0", "FromDate=" + FROM_DATE, "ToDate=" + TO_DATE,
             "ForwardMode=0", "Deposit=1000", "Currency=USD", "Leverage=500",
             "ExecutionMode=0", "Visual=0", "ShutdownTerminal=1", "ReplaceReport=1",
             "Report=" + report_name, "", "[TesterInputs]"]
    for k, v in sorted(params.items()):
        fv = fmt(v); lines.append(k + "=" + fv + "||" + fv + "||0||" + fv + "||N")
    ini_path.write_text("\n".join(lines) + "\n")

def fire_one(prefix_id, ss, se, vt):
    label = f"W2_S{ss:02d}{se:02d}_VT{int(vt*100):03d}"
    prefix = HOME / f"mt5_prefix_{prefix_id}"
    mt5_dir = prefix / "drive_c/Program Files/MetaTrader 5"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = dict(BASE_PARAMS)
    params["SessionStart"] = ss; params["SessionEnd"] = se; params["VolThreshold"] = vt
    params["MagicNumber"] = MAGIC_BASE + ss*100 + int(vt*100)
    ini = mt5_dir / f"ebb_w2_{label}.ini"
    report_name = f"ebb_w2_{label}_{stamp}.htm"
    build_ini(params, ini, report_name)
    ini_wine = f"C:\\Program Files\\MetaTrader 5\\ebb_w2_{label}.ini"
    cmd_path = Path(f"/tmp/ebb_w2_{label}_{stamp}.command")
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
    return {"label": label, "prefix": prefix_id, "session": [ss, se], "vt": vt,
            "report_abs_path": str(mt5_dir / report_name), "fired_at": datetime.now().isoformat()}

def main():
    jobs = []
    for (ss, se), vt in product(SESSIONS, VTS):
        jobs.append((ss, se, vt))
    # 4 prefixes, 5 batches
    all_results = []
    for batch_idx, i in enumerate(range(0, len(jobs), 4)):
        batch = jobs[i:i+4]
        print(f"[batch {batch_idx+1}/{(len(jobs)+3)//4}] firing {len(batch)} configs")
        fired = []
        for pfx_idx, (ss, se, vt) in enumerate(batch):
            prefix_id = pfx_idx + 1
            info = fire_one(prefix_id, ss, se, vt)
            fired.append(info)
            time.sleep(2.0)
        # Poll for HTM files to appear; max wait 6 min
        deadline = time.time() + 360
        while time.time() < deadline:
            done = 0
            for f in fired:
                if Path(f["report_abs_path"]).exists():
                    done += 1
            if done == len(fired):
                break
            time.sleep(10)
        all_results.extend(fired)
        print(f"[batch {batch_idx+1}] done={done}/{len(fired)}")
    manifest = OUT_ROOT / f"wave2_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[MANIFEST] {manifest}")

if __name__ == "__main__":
    main()
