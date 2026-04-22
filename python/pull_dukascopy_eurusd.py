"""
Download EURUSD tick data from Dukascopy datafeed for 2020-01-01 to 2026-04-13 (6yr).
Cloned 1:1 from GBB pull_dukascopy_xauusd.py. Only changes: symbol, price divisor, sanity, filenames.

Schema (matches MT5 tick parquet):
  time    timestamp[ms, UTC]
  bid     float64
  ask     float64
  last    float64   (always 0.0)
  volume  int64     (always 0)
  flags   int32     (always 134)

Dukascopy URL pattern (same as gold):
  https://datafeed.dukascopy.com/datafeed/EURUSD/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5
  Month is 0-indexed (January = 00).

Binary format per tick (big-endian, 20 bytes):
  uint32  ms_offset   milliseconds since start of hour
  uint32  ask_raw     ask * 100000 (divide by 100000 for USD price for EURUSD)
  uint32  bid_raw     bid * 100000
  float32 ask_vol     ask volume in lots
  float32 bid_vol     bid volume in lots

Usage:
  python pull_dukascopy_eurusd.py [--from 2020-01-01] [--to 2026-04-13] [--workers 6]
"""

import sys, os, struct, lzma, logging, argparse, calendar
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

LOG_FILE = LOG_DIR / "dukascopy_pull_eurusd.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── EURUSD-specific constants ─────────────────────────────────────────────────
SYMBOL = "EURUSD"
PRICE_DIVISOR = 100000.0  # raw / 100000 = USD/EUR price (5 decimals)
PRICE_MIN, PRICE_MAX = 0.5, 2.5  # EURUSD sanity range

SCHEMA = pa.schema([
    ("time",   pa.timestamp("ms", tz="UTC")),
    ("bid",    pa.float64()),
    ("ask",    pa.float64()),
    ("last",   pa.float64()),
    ("volume", pa.int64()),
    ("flags",  pa.int32()),
])

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def _is_market_closed(dt_utc: datetime) -> bool:
    wd = dt_utc.weekday()
    if wd == 5: return True
    if wd == 4 and dt_utc.hour >= 22: return True
    if wd == 6 and dt_utc.hour < 22: return True
    return False


def _build_url(year, month, day, hour):
    return f"https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{year:04d}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"


def download_hour_ticks(year, month, day, hour, retries=3, backoff=1.0):
    url = _build_url(year, month, day, hour)
    base_ts_ms = int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp() * 1000)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp: raw = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404: return []
            if attempt < retries - 1: time_mod.sleep(backoff * (attempt + 1)); continue
            log.warning(f"HTTP {e.code} after {retries} retries: {url}"); return []
        except Exception as e:
            if attempt < retries - 1: time_mod.sleep(backoff * (attempt + 1)); continue
            log.warning(f"Network error for {url}: {e}"); return []
    if len(raw) == 0: return []
    try: data = lzma.decompress(raw)
    except lzma.LZMAError as e: log.warning(f"LZMA decode error: {url} {e}"); return []
    n = len(data) // 20
    if n == 0: return []
    ticks = []
    for i in range(n):
        off = i * 20
        ms_offset, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">IIIff", data[off:off+20])
        ts_ms = base_ts_ms + ms_offset
        bid = bid_raw / PRICE_DIVISOR
        ask = ask_raw / PRICE_DIVISOR
        if not (PRICE_MIN < bid < PRICE_MAX): continue
        ticks.append((ts_ms, bid, ask))
    return ticks


def download_day(year, month, day, workers=4):
    dt = datetime(year, month, day, tzinfo=timezone.utc)
    if _is_market_closed(dt): return []
    hours = [h for h in range(24) if not _is_market_closed(datetime(year, month, day, h, tzinfo=timezone.utc))]
    all_ticks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_hour_ticks, year, month, day, h): h for h in hours}
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


def month_output_path(year, month):
    return DATA_DIR / f"eurusd_ticks_{year:04d}-{month:02d}.parquet"


def save_month(df, year, month):
    df = df.sort_values("time").drop_duplicates(subset="time", keep="first")
    out = month_output_path(year, month)
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, out, compression="snappy", row_group_size=500_000)
    sz_mb = out.stat().st_size / 1024 / 1024
    log.info(f"  Saved {out.name}: {len(df):,} rows, {sz_mb:.1f} MB")
    return out


def download_month(year, month, day_workers=4):
    out_path = month_output_path(year, month)
    if out_path.exists():
        existing = pq.read_metadata(out_path).num_rows
        log.info(f"  SKIP {year:04d}-{month:02d}: already exists ({existing:,} rows)"); return existing
    days_in = calendar.monthrange(year, month)[1]
    log.info(f"  Downloading {year:04d}-{month:02d} ({days_in} days) ...")
    all_frames = []; total_ticks = 0
    for day in range(1, days_in + 1):
        ticks = download_day(year, month, day, workers=day_workers)
        if ticks: all_frames.append(ticks_to_df(ticks)); total_ticks += len(ticks)
    if not all_frames: log.warning(f"  {year:04d}-{month:02d}: 0 ticks"); return 0
    df_month = pd.concat(all_frames, ignore_index=True)
    save_month(df_month, year, month)
    if total_ticks < 500_000:
        log.warning(f"  FLAG: {year:04d}-{month:02d} has only {total_ticks:,} ticks")
    return total_ticks


def iter_months(start_date, end_date):
    y, m = start_date.year, start_date.month
    ey, em = end_date.year, end_date.month
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12: m = 1; y += 1


def smoke_test():
    log.info("=== SMOKE TEST: EURUSD 2025-11-03 (Monday) ===")
    ticks = download_day(2025, 11, 3, workers=4)
    if not ticks: log.error("SMOKE FAIL: 0 ticks"); return False
    df = ticks_to_df(ticks)
    log.info(f"  Smoke rows: {len(df):,}")
    mid_bid = df["bid"].median()
    if not (0.9 < mid_bid < 1.7):
        log.error(f"SMOKE FAIL: median bid={mid_bid:.5f} out of EURUSD range")
        return False
    log.info(f"  Bid range: {df['bid'].min():.5f} – {df['bid'].max():.5f}")
    log.info(f"  Ask range: {df['ask'].min():.5f} – {df['ask'].max():.5f}")
    log.info(f"  Sample:\n{df.head(3).to_string(index=False)}")
    log.info("=== SMOKE TEST PASSED ===")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", default="2020-01-01")
    parser.add_argument("--to", dest="end", default="2026-04-13")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start); end_date = date.fromisoformat(args.end)
    log.info("="*60)
    log.info(f"Dukascopy EURUSD tick pull: {start_date} to {end_date}")
    log.info(f"Output: {DATA_DIR}, Workers: {args.workers}")
    log.info("="*60)

    if not args.skip_smoke:
        if not smoke_test(): sys.exit(1)
        if args.smoke_only: sys.exit(0)

    months = list(iter_months(start_date, end_date))
    log.info(f"Total months: {len(months)}")
    t_start = time_mod.time(); total_rows = 0; flagged = []
    for i, (y, m) in enumerate(months, 1):
        t0 = time_mod.time()
        rows = download_month(y, m, day_workers=args.workers)
        elapsed = time_mod.time() - t0
        total_rows += rows
        eta_h = ((time_mod.time() - t_start) / (i / len(months))) * (1 - i / len(months)) / 3600 if i > 0 else 0
        log.info(f"[{i}/{len(months)}] {y:04d}-{m:02d}: {rows:,} rows ({elapsed:.0f}s) ETA: {eta_h:.1f}h")
        if 0 < rows < 500_000: flagged.append((y, m, rows))

    log.info("="*60)
    log.info(f"DONE. Total: {total_rows:,} rows across {len(months)} months in {(time_mod.time()-t_start)/3600:.2f}h")
    if flagged:
        log.warning(f"FLAGGED ({len(flagged)} months with <500k rows):")
        for y, m, r in flagged: log.warning(f"  {y:04d}-{m:02d}: {r:,}")


if __name__ == "__main__":
    main()
