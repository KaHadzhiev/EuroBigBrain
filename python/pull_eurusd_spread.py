#!/usr/bin/env python3
"""Pull EURUSD per-bar spread series from MT5 history.

Output: data/eurusd_spread_2020_2026.parquet (or .csv fallback).
Schema: time (int64 unix-seconds), spread_points (int32), spread_price (float32).

Used by mt5_sim.py to replace fixed-spread assumption with bar-by-bar
historical spread replay.
"""
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed", file=sys.stderr)
    sys.exit(2)

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "data"
OUT_DIR.mkdir(exist_ok=True)
OUT_PARQUET = OUT_DIR / "eurusd_spread_2020_2026.parquet"
OUT_CSV = OUT_DIR / "eurusd_spread_2020_2026.csv"

MT5_PATH = r"C:/MT5-Instances/Instance1/terminal64.exe"
PT_PRICE = 0.0001  # 1 pt for EURUSD = 0.01 USD


def main() -> int:
    if not mt5.initialize(MT5_PATH):
        print(f"MT5 init failed: {mt5.last_error()}", file=sys.stderr)
        return 1
    if not mt5.symbol_select("EURUSD", True):
        print("EURUSD not available", file=sys.stderr)
        mt5.shutdown()
        return 1

    from datetime import timedelta
    from_date = datetime(2020, 1, 1)
    to_date = datetime(2026, 5, 1)
    print(f"Pulling EURUSD M1 {from_date:%Y-%m-%d} -> {to_date:%Y-%m-%d} "
          f"in 30-day chunks...")
    chunks = []
    cur = from_date
    while cur < to_date:
        nxt = min(cur + timedelta(days=30), to_date)
        r = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M1, cur, nxt)
        if r is not None and len(r) > 0:
            chunks.append(r)
            print(f"  {cur:%Y-%m-%d}->{nxt:%Y-%m-%d}: {len(r):,}")
        cur = nxt
    mt5.shutdown()
    if not chunks:
        print("ERROR: no rates returned in any chunk", file=sys.stderr)
        return 1
    rates = np.concatenate(chunks)
    # dedupe (chunk overlap shouldn't occur but safe)
    _, uniq = np.unique(rates["time"], return_index=True)
    rates = rates[np.sort(uniq)]

    times = rates["time"].astype(np.int64)
    spreads = rates["spread"].astype(np.int32)
    print(f"Rows: {len(rates):,}")
    print(f"Range: {datetime.fromtimestamp(int(times[0]))} -> "
          f"{datetime.fromtimestamp(int(times[-1]))}")
    print(f"Spread stats (points): min={spreads.min()} "
          f"med={int(np.median(spreads))} "
          f"p95={int(np.percentile(spreads, 95))} max={spreads.max()}")
    print(f"Distinct spread values: {sorted(np.unique(spreads).tolist())[:20]}")
    print(f"Variability (frac != median): "
          f"{((spreads != int(np.median(spreads))).sum() / len(spreads)):.4f}")

    spread_price = (spreads.astype(np.float32) * PT_PRICE).astype(np.float32)

    # Try parquet first, fall back to csv
    saved = None
    try:
        import pandas as pd
        df = pd.DataFrame({
            "time": times,
            "spread_points": spreads,
            "spread_price": spread_price,
        })
        try:
            df.to_parquet(OUT_PARQUET, index=False)
            saved = OUT_PARQUET
        except Exception as e:
            print(f"parquet failed ({e!r}); writing CSV", file=sys.stderr)
            df.to_csv(OUT_CSV, index=False)
            saved = OUT_CSV
    except ImportError:
        # Pure numpy CSV
        with open(OUT_CSV, "w", encoding="utf-8") as f:
            f.write("time,spread_points,spread_price\n")
            for t, sp, spp in zip(times, spreads, spread_price):
                f.write(f"{t},{sp},{spp:.4f}\n")
        saved = OUT_CSV

    print(f"Saved -> {saved}  ({saved.stat().st_size/1024/1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
