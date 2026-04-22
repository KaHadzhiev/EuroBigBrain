#!/usr/bin/env python3
"""Sweep ProbThreshold via MT5 every-tick. Fire 10 thresholds sequentially.
Each test ~15 sec. Total ~2-3 min. Parse all reports for PF + trades + DD.
"""
import os, subprocess, time, re, argparse
from pathlib import Path
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--report-name-prefix", default="EBB_TripleBarrier_h10",
                help="Prefix for MT5 report filenames so multiple runs don't collide")
args = ap.parse_args()

INSTANCE = Path(r"C:\MT5-Instances\Instance2")
TERMINAL = INSTANCE / "terminal64.exe"
THRESHOLDS = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.50]
results = []

for thr in THRESHOLDS:
    TS = int(time.time() * 1000)
    REPORT = f"{args.report_name_prefix}_thr{int(thr*100):02d}_{TS}"
    INI = INSTANCE / f"{REPORT}.ini"
    lines = [
        "[Tester]",
        "Expert=EuroBigBrain\\EBB_TripleBarrier.ex5",
        "Symbol=EURUSD","Period=M5","Optimization=0","Model=1",
        "FromDate=2024.01.01","ToDate=2026.04.13",
        "ForwardMode=0","Deposit=10000","Currency=USD","Leverage=500",
        "ExecutionMode=0","Visual=0","ShutdownTerminal=1",
        f"Report={REPORT}.htm","ReplaceReport=1","",
        "[TesterInputs]",
        f"ProbThreshold={thr}||{thr}||0||{thr}||N",
        "SL_ATR_Mult=0.7||0.7||0||0.7||N",
        "TP_ATR_Mult=2.0||2.0||0||2.0||N",
        "MaxHoldBars=10||10||0||10||N",
        "RiskPercent=0.6||0.6||0||0.6||N",
        "MaxLotSize=0.10||0.10||0||0.10||N",
        "MaxTradesPerDay=20||20||0||20||N",
        "DailyLossCapPct=5.0||5.0||0||5.0||N",
        "MagicNumber=26042201||26042201||0||26042201||N",
        "DebugEveryNTicks=100||100||0||100||N",
        "RequireCrossSymbols=false||false||0||false||N",
    ]
    content = "\r\n".join(lines) + "\r\n"
    with open(INI, "wb") as f:
        f.write(b"\xff\xfe")
        f.write(content.encode("utf-16-le"))
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW | 0x00000004
    si.wShowWindow = 7; si.dwX = -32000; si.dwY = -32000
    print(f"[thr={thr}] launching...", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [str(TERMINAL), f"/config:{INI}", "/portable"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        startupinfo=si, creationflags=0x08000000,
    )
    proc.wait(timeout=300)
    elapsed = time.time() - t0
    # Find report
    report_file = INSTANCE / f"{REPORT}.htm"
    if not report_file.exists():
        print(f"  [thr={thr}] NO REPORT after {elapsed:.1f}s")
        results.append({"thr": thr, "error": "no_report"})
        continue
    # Parse report
    raw = report_file.read_bytes()
    if raw[:2] == b"\xff\xfe":
        html = raw[2:].decode("utf-16-le")
    else:
        html = raw.decode("utf-8", errors="ignore")
    out = {"thr": thr, "elapsed_s": round(elapsed, 1)}
    for label, key in [
        ("Total Net Profit", "net_profit"),
        ("Profit Factor", "pf"),
        ("Total Trades", "trades"),
        ("Balance Drawdown Maximal", "dd_balance"),
        ("Equity Drawdown Maximal", "dd_equity"),
        ("Sharpe Ratio", "sharpe"),
        ("Profit Trades \\(% of total\\)", "wins_pct"),
        ("Loss Trades \\(% of total\\)", "losses_pct"),
    ]:
        pat = re.compile(label + r":?\s*</td>\s*<td[^>]*><b>([^<]+)</b>", re.I)
        m = pat.search(html)
        if m: out[key] = m.group(1).strip()
    results.append(out)
    print(f"  [thr={thr}] PF={out.get('pf','?')} trades={out.get('trades','?')} DD={out.get('dd_equity','?')}")

print("\n=== SUMMARY ===")
print(f"{'thr':>5} {'PF':>6} {'trades':>7} {'DD_eq':>15} {'NetProfit':>10}")
for r in results:
    print(f"{r['thr']:>5} {r.get('pf','?'):>6} {r.get('trades','?'):>7} {r.get('dd_equity','?'):>15} {r.get('net_profit','?'):>10}")

import json
with open(r"C:\Users\kahad\IdeaProjects\EuroBigBrain\runs\mt5_thr_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved C:/Users/kahad/IdeaProjects/EuroBigBrain/runs/mt5_thr_sweep.json")
