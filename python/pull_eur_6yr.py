#!/usr/bin/env python3
"""Pull EURUSD M5 6 years from MT5 history on Win.
Saves to ~/IdeaProjects/EuroBigBrain/data/EURUSD_M5_6yr.csv
"""
import os, sys, time
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd

OUT = os.path.expanduser("~/IdeaProjects/EuroBigBrain/data/EURUSD_M5_6yr.csv")

if not mt5.initialize():
    print("MT5 init failed:", mt5.last_error()); sys.exit(1)
print(f"MT5: {mt5.terminal_info().name} build {mt5.version()[0]}")

end = datetime.now()
start = end - timedelta(days=365*6 + 30)
print(f"Pulling EURUSD M5: {start} -> {end}")

t0 = time.time()
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M5, start, end)
mt5.shutdown()
if rates is None or len(rates) == 0:
    print("FAIL: no data returned"); sys.exit(1)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.rename(columns={'tick_volume': 'volume'})[['time','open','high','low','close','volume']]
print(f"Got {len(df):,} bars in {time.time()-t0:.1f}s. Range {df['time'].min()} -> {df['time'].max()}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Saved {OUT}")
