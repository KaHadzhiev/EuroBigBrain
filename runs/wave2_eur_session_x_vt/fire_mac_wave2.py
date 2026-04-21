#!/usr/bin/env python3
"""EBB Wave-2 Mac overnight — EUR fade_long SESSION x VT matrix, claim-queue style.

4-prefix parallel on Mac; 20 configs (4 sessions x 5 VTs).
Each prefix pulls the next unclaimed job from a shared queue dir.

Sessions: (7,12)=London AM, (13,17)=NY morning, (14,21)=NY afternoon-fade (Krohn), (8,20)=full London+NY
VTs: 0.20, 0.35, 0.50, 0.70, 1.00 (control)

Goal: find the session x VT combo with best PF and >= 300 trades across 6 years.
Total expected run: ~20 configs / 4 parallel = 5 waves x ~2 min = ~10 min.

NOTE: discovery scan. Survivors feed into Wave-3 null-test + SL/TP sensitivity.
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
QUEUE_DIR = OUT_ROOT / "queue"

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

def make_jobs():
    jobs = []
    for (ss, se), vt in product(SESSIONS, VTS):
        label = f"W2_S{ss:02d}{se:02d}_VT{int(vt*100):03d}"
        jobs.append({"label": label, "session": (ss, se), "vt": vt,
                     "magic": MAGIC_BASE + ss*100 + int(vt*100)})
    return jobs

def claim_job(pid):
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(QUEUE_DIR.glob("*.job")):
        claim = f.with_suffix(".claim")
        try:
            f.rename(claim)
            claim.write_text(str(pid))
            return json.loads(claim.read_text().split("\n", 1)[0] if False else
                              Path(str(claim).replace(".claim", ".jobdata")).read_text())
        except Exception:
            continue
    return None

def run_one(prefix_id, job):
    label = job["label"]
    ss, se = job["session"]
    vt = job["vt"]
    prefix = HOME / f"mt5_prefix_{prefix_id}"
    mt5_dir = prefix / "drive_c/Program Files/MetaTrader 5"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = dict(BASE_PARAMS)
    params["SessionStart"] = ss
    params["SessionEnd"] = se
    params["VolThreshold"] = vt
    params["MagicNumber"] = job["magic"]
    ini = mt5_dir / f"ebb_w2_{label}.ini"
    report = f"ebb_w2_{label}_{stamp}.htm"
    build_ini(params, ini, report)
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
    )
    cmd_path.chmod(0o755)
    subprocess.run([str(cmd_path)], capture_output=True, timeout=600)
    return {"label": label, "prefix": prefix_id, "session": [ss, se], "vt": vt,
            "report_abs_path": str(mt5_dir / report), "magic": job["magic"],
            "completed_at": datetime.now().isoformat()}

def worker_loop(prefix_id):
    """Simple poll-and-run loop — one prefix grabs jobs from QUEUE_DIR until empty."""
    import os
    pid = os.getpid()
    results = []
    while True:
        claimed = None
        for f in sorted(QUEUE_DIR.glob("*.job")):
            new_name = f.with_suffix(f".claim_{pid}")
            try:
                f.rename(new_name)
                claimed = json.loads(new_name.read_text())
                break
            except Exception:
                continue
        if claimed is None:
            break
        print(f"[prefix{prefix_id}] running {claimed['label']}")
        res = run_one(prefix_id, claimed)
        results.append(res)
        (OUT_ROOT / f"{claimed['label']}_result.json").write_text(json.dumps(res, indent=2, default=str))
    return results

def main():
    if len(sys.argv) >= 2 and sys.argv[1].startswith("--worker="):
        pid = int(sys.argv[1].split("=")[1])
        res = worker_loop(pid)
        print(f"[prefix{pid}] done {len(res)} jobs")
        return
    # Populate queue
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs()
    for j in jobs:
        (QUEUE_DIR / f"{j['label']}.job").write_text(json.dumps(j))
    print(f"[queue] seeded {len(jobs)} jobs at {QUEUE_DIR}")
    # Fan out 4 workers in background
    script = str(Path(__file__).resolve())
    procs = []
    for pid in [1, 2, 3, 4]:
        log = OUT_ROOT / f"worker_{pid}.log"
        p = subprocess.Popen(["python3", "-u", script, f"--worker={pid}"],
                             stdout=open(log, "w"), stderr=subprocess.STDOUT)
        procs.append(p)
    print(f"[spawn] 4 workers started — poll {OUT_ROOT} for *_result.json")

if __name__ == "__main__":
    main()
