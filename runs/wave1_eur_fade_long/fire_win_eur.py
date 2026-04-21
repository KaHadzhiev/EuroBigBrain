#!/usr/bin/env python3
"""EBB Wave-1 Win launcher — EUR fade_long session-sweep, every-tick MT5.

3 instances in parallel:
- Inst1: session 8-12 (London open)
- Inst2: session 13-17 (NY-only)
- Inst3: session 7-21 (full active hours)

VT fixed at 0.50 (mid-range — see Mac VT-sweep for that axis).

Period: 2020.01.03 -> 2026.04.10
Symbol: EURUSD
Strategy: fade_long (EntryMode=4)
SL_ATR_Mult: 0.5 (EUR-scaled)

NOTE: discovery scan, not validation. Survivors need null + multi-instrument
+ walk-forward before candidate status.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time, os
from datetime import datetime
from pathlib import Path

INSTANCES = {
    1: {"path": Path("C:/MT5-Instances/Instance1"), "session": (8, 12),  "label": "EBB_W1_SESS0812"},
    2: {"path": Path("C:/MT5-Instances/Instance2"), "session": (13, 17), "label": "EBB_W1_SESS1317"},
    3: {"path": Path("C:/MT5-Instances/Instance3"), "session": (7, 21),  "label": "EBB_W1_SESS0721"},
}
EXPERT = "GoldBigBrain\\GBB_Core"
MAGIC_BASE = 20260422100
OUT_ROOT = Path("C:/Users/kahad/IdeaProjects/EuroBigBrain/runs/wave1_eur_fade_long/results")

BASE_PARAMS = {
    "EntryMode": 4,
    "RiskPercent": 0.6,
    "SL_ATR_Mult": 0.50,
    "TP_ATR_Mult": 2.0,
    "BracketOffset": 0.3,
    "BracketBars": 3,
    "MaxTradesPerDay": 20,
    "DailyLossCapPct": 5.0,
    "MaxLotSize": 1.0,
    "EnableBreakEven": True,
    "BE_ATR_Mult": 0.5,
    "EnableTrailing": True,
    "Trail_ATR_Mult": 0.3,
    "EnableTimeStop": True,
    "MaxHoldBars": 12,
    "VolThreshold": 0.50,
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
    # MT5 needs UTF-16 LE BOM for /config ini
    ini_path.write_bytes(b'\xff\xfe' + ("\n".join(lines) + "\n").encode("utf-16-le"))

def fire_one(inst_id, info):
    inst_path = info["path"]
    sess_start, sess_end = info["session"]
    label = info["label"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = dict(BASE_PARAMS)
    params["SessionStart"] = sess_start
    params["SessionEnd"] = sess_end
    params["MagicNumber"] = MAGIC_BASE + inst_id
    ini_local = inst_path / f"ebb_w1_{label}.ini"
    report = f"ebb_w1_{label}_{stamp}.htm"
    build_ini(params, ini_local, report)
    terminal = inst_path / "terminal64.exe"
    log_stdout = OUT_ROOT / f"{label}_{stamp}.out"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Launch hidden via PowerShell
    ps_cmd = f'Start-Process -FilePath "{terminal}" -ArgumentList "/config:{ini_local}" -WindowStyle Hidden -RedirectStandardOutput "{log_stdout}" -RedirectStandardError "{log_stdout}.err"'
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True)
    return {"label": label, "instance": inst_id, "session": [sess_start, sess_end],
            "report_abs_path": str(inst_path / report),
            "magic": params["MagicNumber"], "ini": str(ini_local),
            "launched_at": datetime.now().isoformat()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true")
    args = ap.parse_args()
    if not args.fire:
        print("Pass --fire to launch all 3 Win instances")
        sys.exit(2)
    out = []
    for inst_id, info in INSTANCES.items():
        result = fire_one(inst_id, info)
        out.append(result)
        print(f"[FIRED] Inst{inst_id} sess={info['session']} magic={result['magic']}")
        time.sleep(3.0)
    manifest = OUT_ROOT / f"wave1_win_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[MANIFEST] {manifest}")

if __name__ == "__main__":
    main()
