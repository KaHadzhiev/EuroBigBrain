#!/usr/bin/env python3
"""6yr extension of build_eurusd_features.py — same 23-feature schema.

Difference vs build_eurusd_features.py: reads the 6yr Dukascopy-derived M5 parquet
(eurusd_m5_2020_2026.parquet) instead of GBB's 16-mo CSV. Cross-asset features
(eurgbp/xau/dxy) fall back to GBB's CSVs if available, else neutral 0.0 (so the
EBB EA contract is preserved — 23 features in the EXACT same order).

Output: data/eurusd_features_6yr.parquet
"""
import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
M5_PARQUET = REPO / "data" / "eurusd_m5_2020_2026.parquet"
GBB_DATA_DIR = Path.home() / "GoldBigBrain" / "data"
XAU_M5_PARQUET = REPO / "data" / "XAUUSD_M5_6yr.parquet"
OUT_PATH = REPO / "data" / "eurusd_features_6yr.parquet"

# EUR sanity bounds (NOT gold's 200-20000)
EUR_PRICE_MIN = 0.8
EUR_PRICE_MAX = 1.7
EXPECTED_FEATURE_COLS = 23  # 21 features + close + 2 targets - 1 (close kept)


def smoke_test() -> int:
    """Build features on synthetic 1-month EUR M5 series; verify column count + value ranges."""
    print("[smoke] build_eurusd_features_6yr")
    rng = np.random.default_rng(7)
    n = 30 * 24 * 12  # 30 days of M5 bars
    t = pd.date_range("2024-06-01", periods=n, freq="5min")
    close = 1.085 + rng.normal(0, 0.0002, n).cumsum()
    high = close + np.abs(rng.normal(0, 0.0003, n))
    low = close - np.abs(rng.normal(0, 0.0003, n))
    opn = close + rng.normal(0, 0.0001, n)
    vol = rng.integers(50, 500, n).astype(float)
    eur = pd.DataFrame({"open": opn, "high": high, "low": low, "close": close,
                        "volume": vol}, index=t)
    f = pd.DataFrame(index=eur.index)
    f["close"] = eur["close"]
    h = eur.index.hour
    f["session_asia"] = ((h >= 22) | (h < 7)).astype(int)
    f["session_london"] = ((h >= 7) & (h < 12)).astype(int)
    f["session_overlap"] = ((h >= 12) & (h < 16)).astype(int)
    f["session_ny"] = ((h >= 16) & (h < 22)).astype(int)
    a14 = atr(eur, 14)
    f["atr14_norm"] = a14 / eur["close"]
    f["rsi14"] = rsi(eur["close"], 14)
    f = f.dropna()
    # Verify EUR-typical ranges
    close_ok = EUR_PRICE_MIN <= f["close"].min() and f["close"].max() <= EUR_PRICE_MAX
    atr_norm_ok = (f["atr14_norm"].between(0, 0.01)).all()  # EUR ATR/price < 1%
    rsi_ok = f["rsi14"].between(0, 100).all()
    sess_ok = ((f["session_asia"] + f["session_london"]
                + f["session_overlap"] + f["session_ny"]) == 1).all()
    if close_ok and atr_norm_ok and rsi_ok and sess_ok:
        print(f"[smoke] PASS: {len(f)} rows, EUR price range + atr/rsi/session sanity OK")
        return 0
    print(f"[smoke] FAIL: close={close_ok} atr={atr_norm_ok} rsi={rsi_ok} sess={sess_ok}")
    return 1


def load_eur_m5() -> pd.DataFrame:
    """Load 6yr EURUSD M5 from Dukascopy parquet."""
    if not M5_PARQUET.exists():
        raise FileNotFoundError(f"{M5_PARQUET} not found - run dukascopy_to_m5_bars.py first")
    df = pd.read_parquet(M5_PARQUET)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    cols = ["open", "high", "low", "close"]
    if "volume" not in df.columns:
        df["volume"] = 1
    return df[cols + ["volume"]].astype({"open": float, "high": float, "low": float,
                                          "close": float, "volume": float})


def load_gbb_m5(name: str) -> pd.DataFrame | None:
    """Try to load a cross-asset M5 CSV from GBB data dir. Returns None if missing."""
    p = GBB_DATA_DIR / f"{name}_M5_full.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df.columns = [c.lower().strip() for c in df.columns]
    if "time" in df.columns:
        df["ts"] = pd.to_datetime(df["time"])
    elif "datetime" in df.columns:
        df["ts"] = pd.to_datetime(df["datetime"])
    else:
        for c in df.columns:
            if "time" in c or "date" in c:
                df["ts"] = pd.to_datetime(df[c])
                break
    df = df.set_index("ts").sort_index()
    return df[["close"]].astype(float)


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def main():
    t0 = time.time()
    print(f"Loading EURUSD M5 (6yr Dukascopy) ...")
    eur = load_eur_m5()
    print(f"  rows={len(eur):,}, span={eur.index.min()} -> {eur.index.max()}")

    f = pd.DataFrame(index=eur.index)
    f["close"] = eur["close"]

    # === Tier 1: OHLCV-only (matches build_eurusd_features.py exactly) ===
    h = eur.index.hour
    f["session_asia"]    = ((h >= 22) | (h < 7)).astype(int)
    f["session_london"]  = ((h >= 7) & (h < 12)).astype(int)
    f["session_overlap"] = ((h >= 12) & (h < 16)).astype(int)
    f["session_ny"]      = ((h >= 16) & (h < 22)).astype(int)
    f["min_since_london_open"] = ((h - 7) % 24) * 60 + eur.index.minute
    f["min_to_ny_fix"] = ((16 - h) % 24) * 60 - eur.index.minute
    f["near_ny_fix"] = ((h == 15) & (eur.index.minute >= 55) | (h == 16) & (eur.index.minute <= 5)).astype(int)
    f["dow"] = eur.index.dayofweek

    a14 = atr(eur, 14)
    f["atr14_norm"] = a14 / eur["close"]
    f["vol_of_vol"] = a14.rolling(20).std() / a14.rolling(20).mean()
    f["rsi14"] = rsi(eur["close"], 14)
    f["rsi7"]  = rsi(eur["close"], 7)

    e20 = eur["close"].ewm(span=20).mean()
    e50 = eur["close"].ewm(span=50).mean()
    f["ema20_zdist"] = (eur["close"] - e20) / a14
    f["ema50_zdist"] = (eur["close"] - e50) / a14

    session_id = (h.astype(str) + "_" + eur.index.date.astype(str))
    sess_open = eur.groupby(session_id)["open"].transform("first")
    f["bar_of_sess_mom"] = (eur["close"] - sess_open) / a14

    bar_range = eur["high"] - eur["low"]
    f["range_exp"] = bar_range / bar_range.rolling(20).median()

    h1 = eur["close"].resample("1h").last()
    h1_ema = h1.ewm(span=50).mean()
    h1_slope = (h1_ema - h1_ema.shift(3)).apply(np.sign)
    f["h1_ema50_slope"] = h1_slope.reindex(eur.index, method="ffill")

    if "volume" in eur.columns:
        v = eur["volume"]
        f["tickvol_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    else:
        f["tickvol_z"] = 0.0

    # === Tier 2: cross-asset (load from GBB CSVs if present, else neutral 0) ===
    print("Loading cross-asset (from GBB data dir if available)...")
    gbp = load_gbb_m5("GBPUSD")
    if gbp is not None:
        gbp_c = gbp["close"].reindex(eur.index, method="ffill")
        eurgbp = eur["close"] / gbp_c
        f["eurgbp_ret_5"] = np.log(eurgbp / eurgbp.shift(5))
    else:
        print("  GBPUSD missing -> eurgbp_ret_5 = 0")
        f["eurgbp_ret_5"] = 0.0

    # XAU: prefer pre-resampled 6yr parquet, fall back to GBB CSV, else neutral 0
    xau = None
    if XAU_M5_PARQUET.exists():
        try:
            xdf = pd.read_parquet(XAU_M5_PARQUET)
            xdf["time"] = pd.to_datetime(xdf["time"])
            xdf = xdf.set_index("time").sort_index()
            xau = xdf[["close"]].astype(float)
            print(f"  XAUUSD loaded from parquet ({len(xau):,} rows)")
        except Exception as e:
            print(f"  XAUUSD parquet load failed ({e}); trying GBB CSV")
            xau = None
    if xau is None:
        xau = load_gbb_m5("XAUUSD")
    if xau is not None:
        xau_c = xau["close"].reindex(eur.index, method="ffill")
        f["xau_ret_5"] = np.log(xau_c / xau_c.shift(5))
    else:
        print("  XAUUSD missing -> xau_ret_5 = 0")
        f["xau_ret_5"] = 0.0

    jpy = load_gbb_m5("USDJPY")
    chf = load_gbb_m5("USDCHF")
    if jpy is not None and chf is not None and gbp is not None:
        jpy_c = jpy["close"].reindex(eur.index, method="ffill")
        chf_c = chf["close"].reindex(eur.index, method="ffill")
        gbp_c = gbp["close"].reindex(eur.index, method="ffill")
        dxy_proxy = np.log(jpy_c) + np.log(chf_c) - np.log(gbp_c)
        f["dxy_proxy"] = dxy_proxy
        f["dxy_ret_5"] = (dxy_proxy - dxy_proxy.shift(5))
        f["dxy_z50"]   = (dxy_proxy - dxy_proxy.rolling(50).mean()) / dxy_proxy.rolling(50).std()
    else:
        print("  USDJPY/USDCHF/GBPUSD missing -> dxy_* = 0")
        f["dxy_proxy"] = 0.0
        f["dxy_ret_5"] = 0.0
        f["dxy_z50"]   = 0.0

    # === Targets (same two as 16mo script — TB label is recomputed in trainer) ===
    f["target_next_dir"] = (eur["close"].shift(-1) > eur["close"]).astype(int)
    fwd_max = eur["high"].rolling(5).max().shift(-5)
    f["target_5bar_up_1atr"] = ((fwd_max - eur["close"]) > a14).astype(int)

    f = f.dropna()
    print(f"Final feature set: {f.shape[0]:,} rows x {f.shape[1]} cols")
    print(f"Date range: {f.index.min()} -> {f.index.max()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(OUT_PATH, compression="snappy")
    print(f"Saved {OUT_PATH} ({os.path.getsize(OUT_PATH)//1024} KB)")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", help="run smoke test and exit")
    args = ap.parse_args()
    if args.smoke_only:
        sys.exit(smoke_test())
    main()
