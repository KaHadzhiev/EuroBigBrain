#!/usr/bin/env python3
"""Pull EURUSD M5 6yr via Python MT5 lib — uses broker's history not tester slice.
Connects to existing running terminal64 OR launches one if needed.
"""
import os, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd

INSTANCE_PATH = r"C:\MT5-Instances\Instance2\terminal64.exe"
OUT = Path(r"C:\Users\kahad\IdeaProjects\EuroBigBrain\data\EURUSD_M5_6yr.csv")

# Init — pass path to specific terminal so it knows which to use
print("Initializing MT5...")
if not mt5.initialize(path=INSTANCE_PATH, timeout=30000):
    print("MT5 init failed:", mt5.last_error())
    sys.exit(1)
ti = mt5.terminal_info()
ai = mt5.account_info()
print(f"MT5: {ti.name if ti else '?'} build {mt5.version()[0]}")
print(f"Account: {ai.login if ai else 'NOT LOGGED IN'} broker={ai.company if ai else '?'}")

# Force EURUSD into MarketWatch
sym = "EURUSD"
if not mt5.symbol_select(sym, True):
    print(f"FAIL symbol_select({sym}): {mt5.last_error()}")
    mt5.shutdown(); sys.exit(1)
si = mt5.symbol_info(sym)
print(f"Symbol: {sym} visible={si.visible} digits={si.digits}")

# Pull range
end = datetime.now()
start = datetime(2020, 1, 1)
print(f"Pulling EURUSD M5: {start} -> {end}")
t0 = time.time()
rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, start, end)
elapsed = time.time() - t0
mt5.shutdown()

if rates is None or len(rates) == 0:
    print(f"FAIL: 0 bars returned in {elapsed:.1f}s. last_error={mt5.last_error() if hasattr(mt5,'last_error') else '?'}")
    print("Possible reason: broker doesn't expose 6yr to demo account, OR MT5 needs to download history first.")
    sys.exit(1)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.rename(columns={'tick_volume': 'volume'})[['time','open','high','low','close','volume']]
print(f"Got {len(df):,} bars in {elapsed:.1f}s. Range {df['time'].min()} -> {df['time'].max()}")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Saved {OUT} ({OUT.stat().st_size:,} bytes)")
print(f"Years covered: {(df['time'].max() - df['time'].min()).days / 365.25:.1f}")
