#!/usr/bin/env python3
"""Pull EURUSD M5 in YEARLY chunks (forces broker download per year).
"""
import os, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd

INSTANCE_PATH = r"C:\MT5-Instances\Instance2\terminal64.exe"
OUT = Path(r"C:\Users\kahad\IdeaProjects\EuroBigBrain\data\EURUSD_M5_6yr.csv")

if not mt5.initialize(path=INSTANCE_PATH, timeout=30000):
    print("MT5 init failed:", mt5.last_error()); sys.exit(1)

sym = "EURUSD"
mt5.symbol_select(sym, True)

# Pull year by year backwards from now
all_chunks = []
end = datetime(2026, 4, 13)
for year_start in [2026, 2025, 2024, 2023, 2022, 2021, 2020]:
    s = datetime(year_start, 1, 1)
    e = datetime(year_start + 1, 1, 1) if year_start < 2026 else end
    t0 = time.time()
    rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, s, e)
    n = 0 if rates is None else len(rates)
    print(f"  {year_start}: {n:,} bars in {time.time()-t0:.1f}s")
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        all_chunks.append(df)

mt5.shutdown()

if not all_chunks:
    print("ALL chunks empty — broker has no EURUSD M5 history exposed")
    sys.exit(1)

df = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
df = df.rename(columns={'tick_volume': 'volume'})[['time','open','high','low','close','volume']]
print(f"\nTOTAL: {len(df):,} bars")
print(f"Range: {df['time'].min()} -> {df['time'].max()}")
print(f"Years: {(df['time'].max() - df['time'].min()).days / 365.25:.2f}")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Saved {OUT} ({OUT.stat().st_size:,} bytes)")
