#!/usr/bin/env python3
"""Run EBB_ExportEUR_M5_6yr.mq5 via Strategy Tester to dump 6yr EURUSD CSV.
The EA does its work in OnInit then calls TerminalClose, so tester wraps and exits.
"""
import os, subprocess, time
from pathlib import Path

INSTANCE = Path(r"C:\MT5-Instances\Instance2")
TERMINAL = INSTANCE / "terminal64.exe"
TS = int(time.time() * 1000)
REPORT = f"EBB_export_{TS}"
INI = INSTANCE / f"{REPORT}.ini"

ini_lines = [
    "[Tester]",
    "Expert=EuroBigBrain\\EBB_ExportEUR_M5_6yr.ex5",
    "Symbol=EURUSD","Period=M5","Optimization=0","Model=4",
    "FromDate=2020.01.01","ToDate=2026.04.13",
    "ForwardMode=0","Deposit=10000","Currency=USD","Leverage=500",
    "ExecutionMode=0","Visual=0","ShutdownTerminal=1",
    f"Report={REPORT}.htm","ReplaceReport=1",
    "","[TesterInputs]",
    "InpSymbol=EURUSD",
    "InpFromDate=D'2020.01.01'",
    "InpToDate=D'2026.04.13'",
    "InpBarsFile=EURUSD_M5_6yr.csv",
    "InpShutdownTerminal=true",
]
content = "\r\n".join(ini_lines) + "\r\n"
with open(INI, "wb") as f:
    f.write(b"\xff\xfe")
    f.write(content.encode("utf-16-le"))
print(f"INI written: {INI}")

si = subprocess.STARTUPINFO()
si.dwFlags |= subprocess.STARTF_USESHOWWINDOW | 0x00000004
si.wShowWindow = 7; si.dwX = -32000; si.dwY = -32000

t0 = time.time()
proc = subprocess.Popen(
    [str(TERMINAL), f"/config:{INI}", "/portable"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    startupinfo=si, creationflags=0x08000000,
)
print(f"Launched PID {proc.pid}, waiting up to 5 min...")
proc.wait(timeout=300)
elapsed = time.time() - t0
print(f"Exited after {elapsed:.1f}s")

# CSV in MQL5/Files of the tester agent
candidates = list(INSTANCE.rglob("EURUSD_M5_6yr.csv"))
print(f"CSV candidates: {candidates}")
if candidates:
    p = candidates[0]
    sz = p.stat().st_size
    nlines = sum(1 for _ in open(p, errors="ignore"))
    print(f"FOUND: {p} ({sz} bytes, {nlines} lines)")
else:
    print("CSV NOT WRITTEN — check tester log")
