"""
EuroBigBrain — MT5 M5 history fetcher.

Pulls M5 OHLCV for EURUSD + multi-instrument null-test basket
(GBPUSD, USDJPY, XAGUSD) from a local MT5 terminal and writes
parquet files into ../data/{symbol}_M5_2020_2026.parquet.

Default date range: 2020-01-03 -> 2026-04-10 (matches the gold
6-year corpus so validation horizons line up).

Usage:
    python fetch_history.py                 # fetch all four symbols
    python fetch_history.py --symbols EURUSD
    python fetch_history.py --start 2020-01-03 --end 2026-04-10

Notes:
- MT5 `copy_rates_range` returns broker-server timestamps (UTC-ish
  with broker DST). We normalise to naive UTC-equivalent for parquet.
- `copy_rates_range` is chunked in 50k-bar windows to dodge the
  documented upper bound of ~100k rows per call on some builds.
- Expected EURUSD M5 bar count for 6 years is ~450k (forex runs
  ~23h5/d, 5d/week -> ~277 bars/day * ~1620 trading days).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Iterable

import MetaTrader5 as mt5
import pandas as pd


DEFAULT_SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "XAGUSD")
DEFAULT_START = datetime(2020, 1, 3)
DEFAULT_END = datetime(2026, 4, 10)
CHUNK_DAYS = 180  # MT5 copy_rates_range safe chunk size
MT5_PATH_CANDIDATES = (
    r"C:\MT5-Instances\Instance1\terminal64.exe",
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def _init_mt5() -> bool:
    for path in MT5_PATH_CANDIDATES:
        if os.path.exists(path) and mt5.initialize(path=path):
            return True
    # Fallback: use whichever terminal Windows' registry points at.
    return mt5.initialize()


def _fetch_chunked(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """copy_rates_range in CHUNK_DAYS slices; concat; drop dupes."""
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, cursor, chunk_end)
        if rates is None:
            err = mt5.last_error()
            print(f"  WARN {symbol} {cursor.date()}->{chunk_end.date()}: {err}")
        elif len(rates) > 0:
            frames.append(pd.DataFrame(rates))
        cursor = chunk_end
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    # MT5 returns: time, open, high, low, close, tick_volume, spread, real_volume
    cols = ["time", "open", "high", "low", "close",
            "tick_volume", "spread", "real_volume"]
    df = df[cols]
    return df


def fetch_symbol(symbol: str, start: datetime, end: datetime,
                 out_dir: str = OUT_DIR) -> str | None:
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"[{symbol}] unknown symbol; enable in Market Watch first")
        return None
    if not info.visible:
        mt5.symbol_select(symbol, True)

    print(f"[{symbol}] fetching {start.date()} -> {end.date()} (M5)")
    df = _fetch_chunked(symbol, start, end)
    if df.empty:
        print(f"[{symbol}] NO DATA")
        return None

    os.makedirs(out_dir, exist_ok=True)
    y0, y1 = start.year, end.year
    out = os.path.join(out_dir, f"{symbol}_M5_{y0}_{y1}.parquet")
    df.to_parquet(out, index=False, compression="snappy")
    print(f"[{symbol}] wrote {len(df):,} bars -> {out}")
    print(f"[{symbol}] span {df['time'].min()} .. {df['time'].max()}")
    return out


def main(symbols: Iterable[str], start: datetime, end: datetime) -> int:
    if not _init_mt5():
        print(f"MT5 init failed: {mt5.last_error()}", file=sys.stderr)
        return 2
    try:
        term = mt5.terminal_info()
        print(f"MT5 terminal: {term.name} build={term.build}")
        rc = 0
        for sym in symbols:
            path = fetch_symbol(sym, start, end)
            if path is None:
                rc = 1
        return rc
    finally:
        mt5.shutdown()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch MT5 M5 history -> parquet")
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
                   default=DEFAULT_START)
    p.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
                   default=DEFAULT_END)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args.symbols, args.start, args.end))
