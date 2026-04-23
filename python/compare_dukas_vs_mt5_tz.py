#!/usr/bin/env python3
"""Test if Dukas-vs-MT5 mismatch is just a timezone offset."""
import sys
from pathlib import Path
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

if not mt5.initialize(): sys.exit(1)
end = datetime.now(tz=timezone.utc)
start = end - pd.Timedelta(days=90)
mt5_rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M15, start, end)

# Get current bid/ask vs current Dukas close
tick = mt5.symbol_info_tick("EURUSD")
print(f"MT5 current: bid={tick.bid}  ask={tick.ask}  spread_pips={(tick.ask-tick.bid)*10000:.1f}")
mt5.shutdown()

mt5_df = pd.DataFrame(mt5_rates)
mt5_df['time'] = pd.to_datetime(mt5_df['time'], unit='s')

# Print first 3 bars of MT5 vs first 3 of Dukas
print(f"\nMT5 first 3 bars:")
print(mt5_df[['time', 'open', 'high', 'low', 'close']].head(3).to_string(index=False))

duk = pd.read_parquet(DATA / "eurusd_m15_2020_2026.parquet")
duk['time'] = pd.to_datetime(duk['time'])
duk_recent = duk[duk['time'] >= start.replace(tzinfo=None)].reset_index(drop=True)
print(f"\nDukas first 3 bars (>= {start.replace(tzinfo=None)}):")
print(duk_recent[['time', 'open', 'high', 'low', 'close']].head(3).to_string(index=False))

# Try shifts: 0, +1h, +2h, +3h, -1h, -2h, -3h
print(f"\n=== Try timezone shifts (MT5 timestamps shifted by N hours) ===")
mt5_df['time_naive'] = mt5_df['time']

best_shift = None
best_diff = float('inf')
for shift_h in [0, +1, +2, +3, -1, -2, -3]:
    mt5_shifted = mt5_df.copy()
    mt5_shifted['time_naive'] = mt5_shifted['time'] + pd.Timedelta(hours=shift_h)
    duk['time_naive'] = duk['time']
    m = pd.merge(duk, mt5_shifted, on='time_naive', suffixes=('_d', '_m'), how='inner')
    if len(m) == 0:
        print(f"  shift {shift_h:+d}h: no overlap")
        continue
    diff_pips = ((m['close_m'] - m['close_d']).abs() * 10000).median()
    print(f"  shift {shift_h:+d}h: matched={len(m):,}  median |close diff|={diff_pips:.3f} pips")
    if diff_pips < best_diff:
        best_diff = diff_pips
        best_shift = shift_h

print(f"\nBEST: shift {best_shift:+d}h gives {best_diff:.3f} pip median close-diff")
