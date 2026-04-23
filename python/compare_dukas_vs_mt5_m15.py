#!/usr/bin/env python3
"""Compare Dukascopy-resampled M15 vs Vantage MT5 native M15 (last 3mo).

If aligned, our 6yr Dukascopy backtest IS broker-accurate.
If diverged, we need to recalibrate.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

if not mt5.initialize():
    sys.exit(1)

end = datetime.now(tz=timezone.utc)
start = end - pd.Timedelta(days=90)
print(f"window: {start} -> {end}")

print("\n--- Vantage MT5 M15 ---")
mt5_rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M15, start, end)
mt5.shutdown()
mt5_df = pd.DataFrame(mt5_rates)
mt5_df['time'] = pd.to_datetime(mt5_df['time'], unit='s')
mt5_df = mt5_df.sort_values('time').reset_index(drop=True)
print(f"  rows={len(mt5_df):,}  span {mt5_df['time'].iloc[0]} -> {mt5_df['time'].iloc[-1]}")

print("\n--- Dukascopy M15 (resampled) ---")
duk = pd.read_parquet(DATA / "eurusd_m15_2020_2026.parquet")
duk['time'] = pd.to_datetime(duk['time'])
duk = duk[(duk['time'] >= start.replace(tzinfo=None)) & (duk['time'] < end.replace(tzinfo=None))].reset_index(drop=True)
print(f"  rows={len(duk):,}  span {duk['time'].iloc[0]} -> {duk['time'].iloc[-1]}")

# Inner-join on time
print("\n--- Inner-join + compare ---")
duk['time_naive'] = duk['time'].dt.tz_localize(None) if duk['time'].dt.tz is not None else duk['time']
mt5_df['time_naive'] = mt5_df['time'].dt.tz_localize(None) if mt5_df['time'].dt.tz is not None else mt5_df['time']
merged = pd.merge(duk, mt5_df, on='time_naive', suffixes=('_d', '_m'), how='inner')
print(f"  matched {len(merged):,} bars on common timestamps")
if len(merged) == 0:
    print("  -- NO COMMON TIMESTAMPS — check timezone alignment --")
    print(f"  Dukas first 3 times: {duk['time_naive'].iloc[:3].tolist()}")
    print(f"  MT5   first 3 times: {mt5_df['time_naive'].iloc[:3].tolist()}")
    sys.exit(0)

# Stats
for col in ['open', 'high', 'low', 'close']:
    diff = (merged[f'{col}_m'] - merged[f'{col}_d']).abs()
    diff_pips = diff * 10000
    print(f"  {col:>6}: |diff| mean={diff_pips.mean():.3f}p  median={diff_pips.median():.3f}p  p95={diff_pips.quantile(0.95):.3f}p  max={diff_pips.max():.3f}p")

# Bar-count alignment
print(f"\n  bar-count: Dukas={len(duk)}  MT5={len(mt5_df)}  ratio={len(duk)/len(mt5_df):.3f}")

# Volume? Vantage has tick count, Dukas has tick count too (from our resample sum)
if 'tick_volume' in merged.columns:
    duk_vol_mean = merged['volume'].mean()
    mt5_vol_mean = merged['tick_volume'].mean()
    print(f"  volume: Dukas mean={duk_vol_mean:.1f}  MT5 mean={mt5_vol_mean:.1f}  ratio={duk_vol_mean/mt5_vol_mean:.2f}")

# Save merged for inspection
merged.to_parquet(DATA / "eurusd_m15_dukas_vs_mt5.parquet", compression="snappy", index=False)
print(f"\n  saved overlap parquet")
