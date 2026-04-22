"""
Generic Dukascopy tick puller — works for any symbol with proper divisor + sanity.

Usage examples:
  python pull_dukascopy_generic.py --symbol GBPUSD --divisor 100000 --pmin 0.9  --pmax 1.7   --from 2020-01-01 --to 2026-04-13
  python pull_dukascopy_generic.py --symbol USDCHF --divisor 100000 --pmin 0.7  --pmax 1.3   --from 2020-01-01 --to 2026-04-13
  python pull_dukascopy_generic.py --symbol USDJPY --divisor 1000   --pmin 70   --pmax 180   --from 2020-01-01 --to 2026-04-13
  python pull_dukascopy_generic.py --symbol XAUUSD --divisor 1000   --pmin 1500 --pmax 4000  --from 2020-01-01 --to 2026-04-13

Cloned from pull_dukascopy_eurusd.py with symbol/divisor/sanity made parametric.
"""
import sys, struct, lzma, logging, argparse, calendar
import time as time_mod
import urllib.request, urllib.error
from datetime import datetime, timezone, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "dukascopy"
LOG_DIR = REPO / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA = pa.schema([
    ("time",   pa.timestamp("ms", tz="UTC")),
    ("bid",    pa.float64()),
    ("ask",    pa.float64()),
    ("last",   pa.float64()),
    ("volume", pa.int64()),
    ("flags",  pa.int32()),
])

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}


def _is_market_closed(dt_utc):
    wd = dt_utc.weekday()
    if wd == 5: return True
    if wd == 4 and dt_utc.hour >= 22: return True
    if wd == 6 and dt_utc.hour < 22: return True
    return False


def download_hour_ticks(symbol, year, month, day, hour, divisor, pmin, pmax, retries=3, backoff=1.0):
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{year:04d}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    base_ts_ms = int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp() * 1000)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp: raw = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404: return []
            if attempt < retries - 1: time_mod.sleep(backoff * (attempt + 1)); continue
            return []
        except Exception:
            if attempt < retries - 1: time_mod.sleep(backoff * (attempt + 1)); continue
            return []
    if len(raw) == 0: return []
    try: data = lzma.decompress(raw)
    except lzma.LZMAError: return []
    n = len(data) // 20
    if n == 0: return []
    ticks = []
    for i in range(n):
        off = i * 20
        ms_offset, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">IIIff", data[off:off+20])
        ts_ms = base_ts_ms + ms_offset
        bid = bid_raw / divisor
        ask = ask_raw / divisor
        if not (pmin < bid < pmax): continue
        ticks.append((ts_ms, bid, ask))
    return ticks


def download_day(symbol, year, month, day, divisor, pmin, pmax, workers=4):
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    if _is_market_closed(dt): return []
    hours = [h for h in range(24) if not _is_market_closed(datetime(year, month, day, h, tzinfo=timezone.utc))]
    all_ticks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_hour_ticks, symbol, year, month, day, h, divisor, pmin, pmax): h for h in hours}
        for fut in as_completed(futures): all_ticks.extend(fut.result())
    all_ticks.sort(key=lambda x: x[0])
    return all_ticks


def ticks_to_df(ticks):
    if not ticks: return pd.DataFrame(columns=["time","bid","ask","last","volume","flags"])
    ts_ms = [t[0] for t in ticks]; bids = [t[1] for t in ticks]; asks = [t[2] for t in ticks]; n = len(ticks)
    return pd.DataFrame({
        "time": pd.to_datetime(ts_ms, unit="ms", utc=True).astype("datetime64[ms, UTC]"),
        "bid": pd.array(bids, dtype="float64"),
        "ask": pd.array(asks, dtype="float64"),
        "last": pd.array([0.0]*n, dtype="float64"),
        "volume": pd.array([0]*n, dtype="int64"),
        "flags": pd.array([134]*n, dtype="int32"),
    })


def download_month(symbol, year, month, divisor, pmin, pmax, day_workers=4, log=None):
    out_path = DATA_DIR / f"{symbol.lower()}_ticks_{year:04d}-{month:02d}.parquet"
    if out_path.exists():
        existing = pq.read_metadata(out_path).num_rows
        log.info(f"  SKIP {symbol} {year:04d}-{month:02d}: already exists ({existing:,})")
        return existing
    days_in = calendar.monthrange(year, month)[1]
    log.info(f"  Downloading {symbol} {year:04d}-{month:02d} ({days_in} days)")
    all_frames = []; total_ticks = 0
    for day in range(1, days_in + 1):
        ticks = download_day(symbol, year, month, day, divisor, pmin, pmax, workers=day_workers)
        if ticks: all_frames.append(ticks_to_df(ticks)); total_ticks += len(ticks)
    if not all_frames:
        log.warning(f"  {symbol} {year:04d}-{month:02d}: 0 ticks"); return 0
    df_month = pd.concat(all_frames, ignore_index=True).sort_values("time").drop_duplicates(subset="time")
    table = pa.Table.from_pandas(df_month, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy", row_group_size=500_000)
    sz_mb = out_path.stat().st_size / 1024 / 1024
    log.info(f"  Saved {out_path.name}: {len(df_month):,} rows, {sz_mb:.1f} MB")
    return total_ticks


def iter_months(start_date, end_date):
    y, m = start_date.year, start_date.month
    ey, em = end_date.year, end_date.month
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12: m = 1; y += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, help="e.g. GBPUSD, USDCHF, USDJPY, XAUUSD")
    p.add_argument("--divisor", type=float, required=True, help="Dukascopy raw → price divisor (100000 for 5dec, 1000 for 3dec)")
    p.add_argument("--pmin", type=float, required=True)
    p.add_argument("--pmax", type=float, required=True)
    p.add_argument("--from", dest="start", default="2020-01-01")
    p.add_argument("--to", dest="end", default="2026-04-13")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    LOG_FILE = LOG_DIR / f"dukascopy_pull_{args.symbol.lower()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    log = logging.getLogger(__name__)

    start_date = date.fromisoformat(args.start); end_date = date.fromisoformat(args.end)
    log.info("="*60)
    log.info(f"Dukascopy {args.symbol} pull: {start_date} to {end_date}")
    log.info(f"Divisor={args.divisor} Sanity=[{args.pmin}, {args.pmax}]")
    log.info("="*60)

    months = list(iter_months(start_date, end_date))
    log.info(f"Months: {len(months)}")
    t_start = time_mod.time(); total_rows = 0
    for i, (y, m) in enumerate(months, 1):
        rows = download_month(args.symbol, y, m, args.divisor, args.pmin, args.pmax, args.workers, log=log)
        total_rows += rows
        log.info(f"[{i}/{len(months)}] {args.symbol} {y:04d}-{m:02d}: {rows:,} rows")
    log.info(f"DONE {args.symbol}. Total: {total_rows:,} rows in {(time_mod.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
