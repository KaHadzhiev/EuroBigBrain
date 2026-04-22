#!/usr/bin/env python3
"""Run EBB MT5 every-tick on FULL 2020.01.01 → 2026.04.13.
First run will trigger Vantage history download for EUR (~5-10 min).
Subsequent runs will hit .hcs cache and be fast.
"""
import os, subprocess, time
from pathlib import Path

INSTANCE = Path(r"C:\MT5-Instances\Instance2")
TERMINAL = INSTANCE / "terminal64.exe"
TS = int(time.time() * 1000)
REPORT = f"EBB_TripleBarrier_h10_6yr_{TS}"
INI = INSTANCE / f"{REPORT}.ini"

ini_lines = [
    "[Tester]",
    "Expert=EuroBigBrain\\EBB_TripleBarrier.ex5",
    "Symbol=EURUSD","Period=M5","Optimization=0","Model=1",
    "FromDate=2020.01.01","ToDate=2026.04.13",
    "ForwardMode=0","Deposit=10000","Currency=USD","Leverage=500",
    "ExecutionMode=0","Visual=0","ShutdownTerminal=1",
    f"Report={REPORT}.htm","ReplaceReport=1","",
    "[TesterInputs]",
    "ProbThreshold=0.35||0.35||0||0.35||N",  # MT5-best threshold
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
content = "\r\n".join(ini_lines) + "\r\n"
with open(INI, "wb") as f: f.write(b"\xff\xfe"); f.write(content.encode("utf-16-le"))
print(f"INI: {INI}")
print(f"Period: 2020.01.01 to 2026.04.13 (6.3 years)")
print(f"Threshold: 0.35 (MT5-best from sweep)")

si = subprocess.STARTUPINFO()
si.dwFlags |= subprocess.STARTF_USESHOWWINDOW | 0x00000004
si.wShowWindow = 7; si.dwX = -32000; si.dwY = -32000

t0 = time.time()
proc = subprocess.Popen(
    [str(TERMINAL), f"/config:{INI}", "/portable"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    startupinfo=si, creationflags=0x08000000,
)
print(f"Launched PID {proc.pid}, expecting ~5-15 min for first 6yr run...")
proc.wait(timeout=1800)  # 30 min cap
elapsed = time.time() - t0
print(f"MT5 exited after {elapsed:.1f}s ({elapsed/60:.1f} min)")

import re
report_file = INSTANCE / f"{REPORT}.htm"
if not report_file.exists():
    print("NO REPORT — check tester logs"); exit(1)
sz = report_file.stat().st_size
raw = report_file.read_bytes()
html = raw[2:].decode("utf-16-le") if raw[:2] == b"\xff\xfe" else raw.decode("utf-8", errors="ignore")
print(f"\nReport: {sz} bytes")
for label, key in [("Total Net Profit","NetProfit"),("Profit Factor","PF"),("Total Trades","Trades"),("Balance Drawdown Maximal","DDbal"),("Equity Drawdown Maximal","DDeq"),("Sharpe Ratio","Sharpe"),("Profit Trades \\(% of total\\)","Wins")]:
    pat = re.compile(label + r":?\s*</td>\s*<td[^>]*><b>([^<]+)</b>", re.I)
    m = pat.search(html)
    if m: print(f"  {key:10s}: {m.group(1).strip()}")
