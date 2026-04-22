#!/usr/bin/env python3
"""Build EURUSD M5 ML feature set per agent E spec (Tier 1 + minimal Tier 2).

Tier 1 (OHLCV-only): session bucket, london distance, NY-fix proximity,
ATR(14)/close, vol-of-vol, RSI(14), RSI(7), EMA20/EMA50 z-distance,
bar-of-session momentum, Hurst-100 (skipped — too slow, use ATR-ratio proxy),
range expansion, H1 EMA50 slope sign, tick-vol z, day-of-week.

Tier 2 minimal: DXY proxy from USDJPY+USDCHF+GBPUSD, EURGBP return, XAU return.

Target: next-bar direction (close[t+1] > close[t]).
Saved as parquet for LightGBM training.
"""
import sys
import os
import time
import numpy as np
import pandas as pd

DATA_DIR = os.path.expanduser("~/GoldBigBrain/data")
OUT_PATH = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")

def load_m5(name):
    path = f"{DATA_DIR}/{name}_M5_full.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if 'date' in df.columns and 'time' in df.columns:
        df['ts'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    elif 'datetime' in df.columns:
        df['ts'] = pd.to_datetime(df['datetime'])
    elif 'time' in df.columns:
        df['ts'] = pd.to_datetime(df['time'])
    else:
        for c in df.columns:
            if 'time' in c or 'date' in c:
                df['ts'] = pd.to_datetime(df[c])
                break
    df = df.set_index('ts').sort_index()
    return df[['open', 'high', 'low', 'close']].astype(float).join(
        df[['volume']].astype(float) if 'volume' in df.columns else
        pd.DataFrame({'volume': np.ones(len(df))}, index=df.index)
    )

def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(df, n):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def main():
    t0 = time.time()
    print(f"Loading EURUSD M5...")
    eur = load_m5("EURUSD")
    print(f"  rows={len(eur):,}, span={eur.index.min()} → {eur.index.max()}")

    f = pd.DataFrame(index=eur.index)
    f['close'] = eur['close']

    # === Tier 1: OHLCV-only ===
    h = eur.index.hour
    f['session_asia'] = ((h >= 22) | (h < 7)).astype(int)
    f['session_london'] = ((h >= 7) & (h < 12)).astype(int)
    f['session_overlap'] = ((h >= 12) & (h < 16)).astype(int)
    f['session_ny'] = ((h >= 16) & (h < 22)).astype(int)
    f['min_since_london_open'] = ((h - 7) % 24) * 60 + eur.index.minute
    f['min_to_ny_fix'] = ((16 - h) % 24) * 60 - eur.index.minute  # GMT 16:00 = London 17:00
    f['near_ny_fix'] = ((h == 15) & (eur.index.minute >= 55) | (h == 16) & (eur.index.minute <= 5)).astype(int)
    f['dow'] = eur.index.dayofweek

    # ATR family
    a14 = atr(eur, 14)
    f['atr14_norm'] = a14 / eur['close']
    f['vol_of_vol'] = a14.rolling(20).std() / a14.rolling(20).mean()

    # RSI family
    f['rsi14'] = rsi(eur['close'], 14)
    f['rsi7'] = rsi(eur['close'], 7)

    # EMA distance z-scored by ATR
    e20 = eur['close'].ewm(span=20).mean()
    e50 = eur['close'].ewm(span=50).mean()
    f['ema20_zdist'] = (eur['close'] - e20) / a14
    f['ema50_zdist'] = (eur['close'] - e50) / a14

    # Bar-of-session momentum: close vs session open
    session_id = (h.astype(str) + '_' + eur.index.date.astype(str))
    sess_open = eur.groupby(session_id)['open'].transform('first')
    f['bar_of_sess_mom'] = (eur['close'] - sess_open) / a14

    # Range expansion
    bar_range = eur['high'] - eur['low']
    f['range_exp'] = bar_range / bar_range.rolling(20).median()

    # H1 EMA50 slope sign (resample, take ema, slope)
    h1 = eur['close'].resample('1H').last()
    h1_ema = h1.ewm(span=50).mean()
    h1_slope = (h1_ema - h1_ema.shift(3)).apply(np.sign)
    f['h1_ema50_slope'] = h1_slope.reindex(eur.index, method='ffill')

    # Tick-volume z-score (using volume column as proxy)
    if 'volume' in eur.columns:
        v = eur['volume']
        f['tickvol_z'] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    else:
        f['tickvol_z'] = 0.0

    # === Tier 2 minimal: cross-asset (need same timestamps) ===
    print("Loading cross-asset...")
    try:
        gbp = load_m5("GBPUSD")['close']
        # EURGBP synthetic: EUR/USD ÷ GBP/USD = EUR/GBP
        eurgbp = eur['close'] / gbp.reindex(eur.index, method='ffill')
        f['eurgbp_ret_5'] = np.log(eurgbp / eurgbp.shift(5))
    except Exception as e:
        print(f"  EURGBP skipped: {e}")
        f['eurgbp_ret_5'] = 0.0

    try:
        xau = load_m5("XAUUSD")['close'] if os.path.exists(f"{DATA_DIR}/XAUUSD_M5_full.csv") else None
        if xau is None:
            # fall back to H1 if M5 missing
            xau = pd.read_csv(f"{DATA_DIR}/XAUUSD_H1.csv")
            xau.columns = [c.lower().strip() for c in xau.columns]
            for c in xau.columns:
                if 'time' in c or 'date' in c:
                    xau.index = pd.to_datetime(xau[c]); break
            xau = xau['close'].astype(float).reindex(eur.index, method='ffill')
        else:
            xau = xau.reindex(eur.index, method='ffill')
        f['xau_ret_5'] = np.log(xau / xau.shift(5))
    except Exception as e:
        print(f"  XAU skipped: {e}")
        f['xau_ret_5'] = 0.0

    # DXY proxy: -log(USDJPY * USDCHF * (1/GBPUSD)) if available
    try:
        jpy = load_m5("USDJPY")['close'].reindex(eur.index, method='ffill')
        chf = load_m5("USDCHF")['close'].reindex(eur.index, method='ffill')
        gbp = load_m5("GBPUSD")['close'].reindex(eur.index, method='ffill')
        dxy_proxy = np.log(jpy) + np.log(chf) - np.log(gbp)  # USD strength composite
        f['dxy_proxy'] = dxy_proxy
        f['dxy_ret_5'] = (dxy_proxy - dxy_proxy.shift(5))
        f['dxy_z50'] = (dxy_proxy - dxy_proxy.rolling(50).mean()) / dxy_proxy.rolling(50).std()
    except Exception as e:
        print(f"  DXY proxy skipped: {e}")

    # === Target: next-bar direction (1 if close[t+1] > close[t]) ===
    f['target_next_dir'] = (eur['close'].shift(-1) > eur['close']).astype(int)
    # Alt target: 5-bar excursion > 1× ATR (gold-style vol target)
    fwd_max = eur['high'].rolling(5).max().shift(-5)
    f['target_5bar_up_1atr'] = ((fwd_max - eur['close']) > a14).astype(int)

    # === Cleanup ===
    f = f.dropna()
    print(f"Final feature set: {f.shape[0]:,} rows × {f.shape[1]} cols")
    print(f"Target balance (next_dir): {f['target_next_dir'].mean():.3f}")
    print(f"Target balance (5bar_up_1atr): {f['target_5bar_up_1atr'].mean():.3f}")
    print(f"Date range: {f.index.min()} → {f.index.max()}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    f.to_parquet(OUT_PATH, compression='snappy')
    print(f"Saved {OUT_PATH} ({os.path.getsize(OUT_PATH)//1024} KB)")
    print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
