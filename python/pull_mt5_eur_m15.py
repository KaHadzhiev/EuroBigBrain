#!/usr/bin/env python3
"""Try to pull EURUSD M15 from MT5 directly (Vantage Standard via Python lib).

Per memory: Python MT5 lib usually returns ~3mo only. But worth checking M15
because cache behavior may differ from M5 (we tested M5 last time, got 0 bars).

Usage:
  python pull_mt5_eur_m15.py
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
DATA.mkdir(exist_ok=True)

if not mt5.initialize():
    print(f"FAIL: mt5.initialize() failed: {mt5.last_error()}")
    sys.exit(1)

ti = mt5.terminal_info()
ai = mt5.account_info()
print(f"terminal: {ti.name}  build={ti.build}  path={ti.data_path}")
print(f"account:  {ai.login}@{ai.server}  ({ai.company})  balance={ai.balance}")
print(f"connected={ti.connected}")

# Symbol info
sym = mt5.symbol_info("EURUSD")
if sym is None:
    print("FAIL: EURUSD symbol not found")
    mt5.shutdown(); sys.exit(1)
print(f"EURUSD spread={sym.spread} points  digits={sym.digits}  trade_mode={sym.trade_mode}")
mt5.symbol_select("EURUSD", True)

# Try a wide range first
print(f"\n--- pull EURUSD M15 from 2020-01-01 to 2026-04-22 ---")
start = datetime(2020, 1, 1, tzinfo=timezone.utc)
end = datetime(2026, 4, 22, tzinfo=timezone.utc)
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M15, start, end)
if rates is None or len(rates) == 0:
    print(f"  GOT 0 BARS  err={mt5.last_error()}")
else:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    print(f"  rows={len(df):,}  span {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    yrs = df['time'].dt.year.value_counts().sort_index()
    print(f"  per year: {dict(yrs)}")
    out = DATA / "eurusd_m15_mt5.parquet"
    df.to_parquet(out, compression="snappy", index=False)
    print(f"  saved {out} ({out.stat().st_size//1024} KB)")

# Also try recent 3mo as control
print(f"\n--- pull EURUSD M15 last 3 months (control) ---")
end2 = datetime.now(tz=timezone.utc)
start2 = end2 - pd.Timedelta(days=90)
rates2 = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M15, start2, end2)
if rates2 is None or len(rates2) == 0:
    print(f"  GOT 0 BARS  err={mt5.last_error()}")
else:
    print(f"  rows={len(rates2):,}  (recent 3mo control)")

mt5.shutdown()
print("done")
