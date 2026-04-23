#!/usr/bin/env python3
"""Resample EURUSD M5 6yr -> M15 OHLCV.

Output: data/eurusd_m15_2020_2026.parquet
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

src = DATA / "eurusd_m5_2020_2026.parquet"
dst = DATA / "eurusd_m15_2020_2026.parquet"

print(f"loading {src}")
df = pd.read_parquet(src)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time').sort_index()

print(f"  M5 rows: {len(df):,}  span={df.index[0]} -> {df.index[-1]}")

m15 = df.resample('15min', label='left', closed='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum'
}).dropna()

m15 = m15.reset_index()
print(f"  M15 rows: {len(m15):,}")

m15.to_parquet(dst, compression='snappy', index=False)
print(f"saved {dst} ({dst.stat().st_size/1024:.1f} KB)")
