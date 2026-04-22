#!/usr/bin/env python3
"""Concatenate 76 monthly Dukascopy EURUSD parquet shards into a single file.

Input:  data/dukascopy/eurusd_ticks_YYYY-MM.parquet  (one per month, 2020-01..2026-04)
Output: data/eurusd_ticks_2020_2026.parquet         (single file, sorted, deduped)

Schema (matches GBB merged file):
  time    timestamp[ms, UTC]
  bid     float64
  ask     float64
  last    float64
  volume  int64
  flags   int32

Cloned 1:1 from GBB pipeline pattern. Dedup by `time`, keep first.
"""
import argparse
import sys
import tempfile
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
SHARD_DIR = REPO / "data" / "dukascopy"
OUT_PATH = REPO / "data" / "eurusd_ticks_2020_2026.parquet"

# EUR sanity bounds (NOT gold's 200-20000)
EUR_PRICE_MIN = 0.8
EUR_PRICE_MAX = 1.7

SCHEMA = pa.schema([
    ("time",   pa.timestamp("ms", tz="UTC")),
    ("bid",    pa.float64()),
    ("ask",    pa.float64()),
    ("last",   pa.float64()),
    ("volume", pa.int64()),
    ("flags",  pa.int32()),
])


def smoke_test() -> int:
    """Synthesize 2 fake monthly EUR shards, merge them, verify schema + EUR price range."""
    print("[smoke] merge_dukascopy_eurusd")
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        rng = np.random.default_rng(42)
        for ym in ("2020-01", "2020-02"):
            n = 1000
            t = pd.date_range(f"{ym}-01", periods=n, freq="1min", tz="UTC")
            bid = 1.10 + rng.normal(0, 0.001, n).cumsum() * 0.01
            df = pd.DataFrame({"time": t, "bid": bid, "ask": bid + 0.0001,
                               "last": 0.0, "volume": rng.integers(1, 10, n).astype("int64"),
                               "flags": np.zeros(n, dtype="int32")})
            tbl = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
            pq.write_table(tbl, tdp / f"eurusd_ticks_{ym}.parquet")
        # Load+merge inline (mirror main()'s logic)
        shards = sorted(tdp.glob("eurusd_ticks_*.parquet"))
        big = pd.concat([pq.read_table(p).to_pandas() for p in shards], ignore_index=True)
        big = big.sort_values("time").drop_duplicates(subset="time", keep="first")
        ok_cols = list(big.columns) == ["time", "bid", "ask", "last", "volume", "flags"]
        ok_range = EUR_PRICE_MIN <= big["bid"].min() and big["bid"].max() <= EUR_PRICE_MAX
        ok_rows = len(big) == 2000
        if ok_cols and ok_range and ok_rows:
            print(f"[smoke] PASS: rows={len(big)} bid_range=[{big['bid'].min():.4f},{big['bid'].max():.4f}]")
            return 0
        print(f"[smoke] FAIL: cols={ok_cols} range={ok_range} rows={ok_rows}")
        return 1


def main() -> int:
    t0 = _time.time()
    if not SHARD_DIR.exists():
        print(f"FATAL: {SHARD_DIR} not found", file=sys.stderr)
        return 1
    shards = sorted(SHARD_DIR.glob("eurusd_ticks_*.parquet"))
    if not shards:
        print(f"FATAL: no shards in {SHARD_DIR}", file=sys.stderr)
        return 1

    print(f"Found {len(shards)} monthly shards in {SHARD_DIR}")
    frames = []
    total_rows = 0
    for i, p in enumerate(shards, 1):
        df = pq.read_table(p).to_pandas()
        total_rows += len(df)
        frames.append(df)
        if i % 12 == 0 or i == len(shards):
            print(f"  [{i}/{len(shards)}] read {p.name}: {len(df):,} rows")

    print(f"\nConcatenating {len(frames)} frames ({total_rows:,} total rows)...")
    big = pd.concat(frames, ignore_index=True)
    print(f"Sorting by time + dedup...")
    big = big.sort_values("time").drop_duplicates(subset="time", keep="first").reset_index(drop=True)
    print(f"  rows after dedup: {len(big):,} (dropped {total_rows - len(big):,})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(big, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, OUT_PATH, compression="snappy", row_group_size=1_000_000)
    sz_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nSaved {OUT_PATH}")
    print(f"  rows={len(big):,}  size={sz_mb:.1f} MB  range={big['time'].min()} -> {big['time'].max()}")
    print(f"Elapsed: {_time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", help="run smoke test and exit")
    args = ap.parse_args()
    if args.smoke_only:
        sys.exit(smoke_test())
    sys.exit(main())
