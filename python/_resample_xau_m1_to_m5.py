#!/usr/bin/env python3
"""One-shot: resample XAU M1 (GBB CSV) to M5 parquet for EBB cross-asset feature."""
import os, time
import pandas as pd

INP = r"C:\Users\kahad\IdeaProjects\GoldBigBrain\data\XAUUSD_M1_2020_2026.csv"
OUT = r"C:\Users\kahad\IdeaProjects\EuroBigBrain\data\XAUUSD_M5_6yr.parquet"

t0 = time.time()
print(f"[load] {INP}")
df = pd.read_csv(INP)
print(f"  raw rows: {len(df):,}, cols: {list(df.columns)}")

# Parse time
df["time"] = pd.to_datetime(df["time_iso"], format="%Y.%m.%d %H:%M:%S")
df = df.set_index("time").sort_index()

agg = {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
m5 = df.resample("5min").agg(agg).dropna(subset=["open", "high", "low", "close"])
m5 = m5.rename(columns={"tick_volume": "volume"}).reset_index()

print(f"[resample] M5 rows: {len(m5):,} ({len(df)/max(len(m5),1):.2f}x reduction)")
print(f"[span] {m5['time'].min()} -> {m5['time'].max()}")
print(f"[OHLC] open=[{m5['open'].min():.2f},{m5['open'].max():.2f}] "
      f"close=[{m5['close'].min():.2f},{m5['close'].max():.2f}]")
nan_count = m5[["open", "high", "low", "close"]].isna().sum().sum()
print(f"[NaN check] OHLC NaNs: {nan_count}")

assert nan_count == 0, "NaN in OHLC after resample"
assert m5["close"].between(1400, 6000).all(), "OHLC out of XAU sane range"  # 2026 highs ~$5500
assert len(m5) > 100_000, "M5 rows suspiciously low"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
m5.to_parquet(OUT, compression="snappy")
sz = os.path.getsize(OUT)
print(f"[save] {OUT} ({sz/1024/1024:.2f} MB)")
print(f"[done] {time.time()-t0:.1f}s | input={os.path.getsize(INP)/1024/1024:.1f}MB output={sz/1024/1024:.2f}MB")
