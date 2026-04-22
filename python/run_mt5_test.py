#!/usr/bin/env python3
"""Run MT5 Strategy Tester for EBB_TripleBarrier — GBB-pattern (proven working).
INI at instance root, CRLF line endings, subprocess.Popen with Windows flags.
"""
import os, sys, subprocess, time
from pathlib import Path
from datetime import datetime

INSTANCE = Path(r"C:\MT5-Instances\Instance2")
TERMINAL = INSTANCE / "terminal64.exe"
TS = int(time.time() * 1000)
REPORT_NAME = f"EBB_TripleBarrier_h10_{TS}"
INI = INSTANCE / f"{REPORT_NAME}.ini"

ini_lines = [
    "[Tester]",
    "Expert=EuroBigBrain\\EBB_TripleBarrier.ex5",
    "Symbol=EURUSD",
    "Period=M5",
    "Optimization=0",
    "Model=1",  # Every tick based on real ticks
    "FromDate=2025.01.01",
    "ToDate=2026.04.13",
    "ForwardMode=0",
    "Deposit=10000",
    "Currency=USD",
    "ProfitInPips=0",
    "Leverage=500",
    "ExecutionMode=0",
    "Visual=0",
    "ShutdownTerminal=1",
    f"Report={REPORT_NAME}.htm",
    "ReplaceReport=1",
    "",
    "[TesterInputs]",
    "ProbThreshold=0.44||0.44||0||0.44||N",
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
with open(INI, "wb") as f:
    f.write(b"\xff\xfe")
    f.write(content.encode("utf-16-le"))
print(f"INI written: {INI} ({len(content)} chars)")

log_dir = INSTANCE / "Tester" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
datestamp = datetime.now().strftime("%Y%m%d")

# GBB pattern: STARTUPINFO with hidden window + CREATE_NO_WINDOW
si = subprocess.STARTUPINFO()
si.dwFlags |= subprocess.STARTF_USESHOWWINDOW | 0x00000004
si.wShowWindow = 7  # SW_SHOWMINNOACTIVE
si.dwX = -32000
si.dwY = -32000

print(f"Launching: {TERMINAL} /config:{INI} /portable")
t0 = time.time()
proc = subprocess.Popen(
    [str(TERMINAL), f"/config:{INI}", "/portable"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    startupinfo=si,
    creationflags=0x08000000,  # CREATE_NO_WINDOW
)
print(f"Launched PID {proc.pid}. Waiting up to 30 min for ShutdownTerminal=1...")
try:
    proc.wait(timeout=1800)
    elapsed = time.time() - t0
    print(f"MT5 exited after {elapsed:.1f}s")
except subprocess.TimeoutExpired:
    print(f"TIMEOUT after 30 min — killing")
    proc.kill()
    sys.exit(1)

# Look for the HTML report
report_paths = list(INSTANCE.rglob(f"{REPORT_NAME}.htm*")) + list(INSTANCE.rglob("*EBB*report*.htm*"))
print(f"\nReport candidates: {report_paths}")
if report_paths:
    p = report_paths[0]
    sz = p.stat().st_size
    print(f"FOUND: {p} ({sz} bytes)")
    print(f"Open it in a browser or run: cat '{p}' | head -50")
else:
    print("NO REPORT — tester likely didn't run. Check Tester/logs/")
    for log in (log_dir).glob("*.log"):
        print(f"  log: {log} ({log.stat().st_size} bytes)")
