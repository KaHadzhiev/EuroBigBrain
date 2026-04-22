#!/usr/bin/env python3
"""Aggregate Dukascopy EURUSD ticks (bid/ask) into M5 OHLCV bars.

Input:  data/eurusd_ticks_2020_2026.parquet  (from merge_dukascopy_eurusd.py)
Output: data/eurusd_m5_2020_2026.parquet     (parquet, schema: time/open/high/low/close/volume)
        data/EURUSD_M5_6yr.csv               (CSV mirror — same schema as GBB EURUSD_M5_full.csv)

OHLC source: bid mid (use bid only — matches GBB convention; spread modelling lives in sim).
Volume: tick count per 5-min bucket (Dukascopy exposes lot volumes but they're noisy;
        tick count is the standard MT5-aligned proxy).

Drops bars with zero ticks (weekends, holidays).
"""
import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
TICKS_PATH = REPO / "data" / "eurusd_ticks_2020_2026.parquet"
OUT_PARQUET = REPO / "data" / "eurusd_m5_2020_2026.parquet"
OUT_CSV = REPO / "data" / "EURUSD_M5_6yr.csv"

# EUR sanity bounds (NOT gold's 200-20000)
EUR_PRICE_MIN = 0.8
EUR_PRICE_MAX = 1.7


def smoke_test() -> int:
    """Synthesize 1 day of EUR ticks, aggregate to M5, verify OHLCV math."""
    print("[smoke] dukascopy_to_m5_bars")
    rng = np.random.default_rng(0)
    n = 5000
    t = pd.date_range("2024-06-03 00:00", periods=n, freq="15s", tz="UTC")
    bid = 1.085 + rng.normal(0, 0.0001, n).cumsum()
    ticks = pd.DataFrame({"bid": bid}, index=t)
    bars = ticks["bid"].resample("5min").ohlc()
    bars["volume"] = ticks["bid"].resample("5min").count().astype("int64")
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars = bars[bars["volume"] > 0]
    # Verify: M5-aligned, OHLC math sane, EUR range, volume > 0
    aligned = (bars.index.minute % 5 == 0).all()
    ohlc_ok = ((bars["high"] >= bars["open"]) & (bars["high"] >= bars["close"])
               & (bars["low"] <= bars["open"]) & (bars["low"] <= bars["close"])).all()
    range_ok = EUR_PRICE_MIN <= bars["close"].min() and bars["close"].max() <= EUR_PRICE_MAX
    vol_ok = (bars["volume"] > 0).all()
    n_expected = n * 15 / 300  # ticks/sec * 15s / 300s_per_bar
    nbars_ok = abs(len(bars) - n_expected) <= 2
    if aligned and ohlc_ok and range_ok and vol_ok and nbars_ok:
        print(f"[smoke] PASS: {len(bars)} M5 bars, OHLC math + EUR range + alignment OK")
        return 0
    print(f"[smoke] FAIL: aligned={aligned} ohlc={ohlc_ok} range={range_ok} vol={vol_ok} nbars={nbars_ok}")
    return 1


def main() -> int:
    t0 = _time.time()
    if not TICKS_PATH.exists():
        print(f"FATAL: {TICKS_PATH} not found. Run merge_dukascopy_eurusd.py first.", file=sys.stderr)
        return 1

    print(f"Reading {TICKS_PATH} ...")
    ticks = pq.read_table(TICKS_PATH, columns=["time", "bid"]).to_pandas()
    print(f"  ticks: {len(ticks):,}")

    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    ticks = ticks.set_index("time")

    print("Aggregating to M5 OHLCV (bid mid)...")
    bars = ticks["bid"].resample("5min").ohlc()
    bars["volume"] = ticks["bid"].resample("5min").count().astype("int64")
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars = bars[bars["volume"] > 0]
    bars = bars.reset_index().rename(columns={"time": "time"})
    # Strip TZ to match GBB CSV convention (naive UTC)
    bars["time"] = pd.to_datetime(bars["time"]).dt.tz_convert(None).dt.tz_localize(None) \
                   if hasattr(bars["time"].dt, "tz_convert") and bars["time"].dt.tz is not None \
                   else pd.to_datetime(bars["time"])

    print(f"  M5 bars: {len(bars):,}  span={bars['time'].min()} -> {bars['time'].max()}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(OUT_PARQUET, compression="snappy", index=False)
    sz_p = OUT_PARQUET.stat().st_size / 1024 / 1024
    print(f"Saved {OUT_PARQUET} ({sz_p:.1f} MB)")

    # CSV mirror (matches GBB ~/GoldBigBrain/data/EURUSD_M5_full.csv schema)
    bars.to_csv(OUT_CSV, index=False, float_format="%.5f")
    sz_c = OUT_CSV.stat().st_size / 1024 / 1024
    print(f"Saved {OUT_CSV} ({sz_c:.1f} MB)")

    print(f"Elapsed: {_time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", help="run smoke test and exit")
    args = ap.parse_args()
    if args.smoke_only:
        sys.exit(smoke_test())
    sys.exit(main())
