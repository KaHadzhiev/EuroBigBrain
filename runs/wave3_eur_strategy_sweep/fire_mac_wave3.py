#!/usr/bin/env python3
"""EBB Wave-3 — pivoted from fade_long fine-grain to STRATEGY variant sweep.

Verdict from Wave 1+2: fade_long doesn't transfer to EUR (every cfg PF<1 or wipe).
This wave tests other entry modes to see if ANY archetype has EUR edge.

EntryMode mapping (from GBB_Core.mq5 / GBB_Generic.mq5):
  1 = momentum_long   (EMA cross + RSI buy)
  2 = breakout_range  (N-bar high break)
  3 = asian_range     (Asian-session range break)
  4 = fade_long       (RSI<40 buy) — already tested, FAILS
  5 = atr_breakout    (ATR-multiple break)
  6 = ema_cross_long  (clean EMA cross)
  7 = fade_short      (RSI>65 sell) — opposite of #4

Sessions tested:
  (8, 12)   — London open momentum window
  (13, 17)  — NY morning
  (14, 16)  — Krohn NY-overlap fade window (per EUR sessions doc)

Total: 6 entry modes × 3 sessions × 1 VT (0.50 mid) = 18 configs
Plus baseline cfg74-style with each entry mode = ~3 more
= ~21 configs. 4 prefixes parallel = 6 batches × 2-3 min = ~15-18 min.
"""
from __future__ import annotations
import json, subprocess, time
from datetime import datetime
from pathlib import Path
from itertools import product

HOME = Path.home()
WINE = Path("/Applications/MetaTrader 5.app/Contents/SharedSupport/wine/bin/wine64")
EXPERT = "GoldBigBrain\\GBB_Core"
OUT_ROOT = HOME / "eurobigbrain" / "wave3"

ENTRY_MODES = [1, 2, 3, 5, 6, 7]  # skip 4 (fade_long, already failed)
SESSIONS = [(8, 12), (13, 17), (14, 16)]
VT = 0.50
MAGIC_BASE = 20260422300

BASE_PARAMS = {
    "RiskPercent": 0.6, "SL_ATR_Mult": 0.50, "TP_ATR_Mult": 2.0,
    "BracketOffset": 0.3, "BracketBars": 3, "MaxTradesPerDay": 20, "DailyLossCapPct": 5.0,
    "MaxLotSize": 1.0, "EnableBreakEven": True, "BE_ATR_Mult": 0.5, "EnableTrailing": True,
    "Trail_ATR_Mult": 0.3, "EnableTimeStop": True, "MaxHoldBars": 12,
    "AsianStart": 0, "AsianEnd": 7, "BreakoutBars": 20, "VolSpikeThresh": 2.0,
    "FadeLongRSI": 40.0, "FadeShortRSI": 65.0, "EmaFastPeriod": 8, "EmaSlowPeriod": 21,
    "ML_LongThreshold": 0.65, "ML_ShortThreshold": 0.35, "ML_EntryOffsetATR": 0.10,
    "OR_OpenHour": 8, "EnableTrendFilter": False, "VolThreshold": VT,
}
FROM_DATE = "2020.01.03"
TO_DATE = "2026.04.10"
ENTRY_NAMES = {1:"mom_long", 2:"brk_range", 3:"asian", 5:"atr_brk", 6:"ema_x", 7:"fade_short"}

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

def fire_one(prefix_id, entry, ss, se):
    label = f"W3_{ENTRY_NAMES[entry]}_S{ss:02d}{se:02d}"
    prefix = HOME / f"mt5_prefix_{prefix_id}"
    mt5_dir = prefix / "drive_c/Program Files/MetaTrader 5"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = dict(BASE_PARAMS)
    params["EntryMode"] = entry; params["SessionStart"] = ss; params["SessionEnd"] = se
    params["MagicNumber"] = MAGIC_BASE + entry*10000 + ss*100 + se
    ini = mt5_dir / f"ebb_w3_{label}.ini"
    report_name = f"ebb_w3_{label}_{stamp}.htm"
    build_ini(params, ini, report_name)
    ini_wine = f"C:\\Program Files\\MetaTrader 5\\ebb_w3_{label}.ini"
    cmd_path = Path(f"/tmp/ebb_w3_{label}_{stamp}.command")
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
    return {"label": label, "prefix": prefix_id, "entry": entry, "session": [ss, se],
            "report_abs_path": str(mt5_dir / report_name), "fired_at": datetime.now().isoformat()}

def main():
    jobs = list(product(ENTRY_MODES, SESSIONS))  # 6*3 = 18
    print(f"[wave3] {len(jobs)} jobs queued: {len(ENTRY_MODES)} entries × {len(SESSIONS)} sessions @ VT={VT}")
    all_results = []
    for batch_idx, i in enumerate(range(0, len(jobs), 4)):
        batch = jobs[i:i+4]
        print(f"[batch {batch_idx+1}/{(len(jobs)+3)//4}] firing {len(batch)} configs")
        fired = []
        for pfx_idx, (entry, (ss, se)) in enumerate(batch):
            prefix_id = pfx_idx + 1
            info = fire_one(prefix_id, entry, ss, se)
            fired.append(info)
            time.sleep(2.0)
        deadline = time.time() + 360
        while time.time() < deadline:
            done = sum(1 for f in fired if Path(f["report_abs_path"]).exists())
            if done == len(fired): break
            time.sleep(10)
        all_results.extend(fired)
        print(f"[batch {batch_idx+1}] done={done}/{len(fired)}")
    manifest = OUT_ROOT / f"wave3_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[MANIFEST] {manifest}")

if __name__ == "__main__":
    main()
