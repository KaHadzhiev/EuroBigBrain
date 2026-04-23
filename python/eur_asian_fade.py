#!/usr/bin/env python3
"""EUR Asian-session fade — FXStabilizer / STARLIGHT clone test.

Hypothesis (per WG-2 agents #1, #4): EUR mean-reverts during Asian session
(low vol, range-bound). Trade with ASYMMETRIC R:R: wide SL, tight TP, high
win-rate. Opposite of what our 5,400-config sweep tried (which was tight SL,
wide TP, low win-rate).

Pattern:
  - Window: 21:00 GMT (Asian open) -> 01:00 GMT next day
  - Entry: when price extends >= N * ATR(M15) from EMA(M15, 50), fade it
  - SL:    1.5 * ATR(M15) wide (accept large losses)
  - TP:    0.7 * ATR(M15) tight (small win, snap-back)
  - Hold:  4 hr max
  - Cost:  Vantage 1.3 pip + 0.2 slip = 1.5 pip round-trip

Run a small grid:
  - extension_atr: [1.0, 1.5, 2.0]
  - sl_atr:        [1.0, 1.5, 2.0]
  - tp_atr:        [0.4, 0.6, 0.8, 1.0]
  - session:       [(21, 1), (22, 2), (20, 0)]
  - hold_bars:     [24, 48]   # M5 = 2hr / 4hr

Output: per-config metrics + log of best.
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"
RESULTS.mkdir(parents=True, exist_ok=True)

PIP = 0.0001
COST_PIPS = 1.5
USD_PER_PIP = 1.0


def _atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=True, min_periods=span).mean().to_numpy()


def session_bars(hours, s, e):
    """Return mask of bars where hour in [s, e). Wraps midnight if e<s."""
    if e <= s:
        return (hours >= s) | (hours < e)
    return (hours >= s) & (hours < e)


def run_one(ctx, ext_atr, sl_atr, tp_atr, sess, hold_bars):
    s, e = sess
    h = ctx['h']; l = ctx['l']; c = ctx['c']
    atr15 = ctx['atr15']
    ema15 = ctx['ema15_50']
    hours = ctx['hours']
    times = ctx['times']
    n = len(c)

    in_sess = session_bars(hours, s, e)
    rows = []
    cooldown_until = -1

    for i in range(50, n - hold_bars):
        if not in_sess[i] or i <= cooldown_until:
            continue
        a = atr15[i]
        if not np.isfinite(a) or a <= 0:
            continue
        em = ema15[i]
        if not np.isfinite(em):
            continue
        dist = c[i] - em

        if dist >= ext_atr * a:
            # too high -> fade SHORT
            direction = 'short'
            entry = c[i]
            sl_px = entry + sl_atr * a
            tp_px = entry - tp_atr * a
        elif -dist >= ext_atr * a:
            # too low -> fade LONG
            direction = 'long'
            entry = c[i]
            sl_px = entry - sl_atr * a
            tp_px = entry + tp_atr * a
        else:
            continue

        # walk forward
        exit_px = np.nan
        exit_bar = i + hold_bars
        exit_reason = 'timeout'
        for k in range(1, hold_bars + 1):
            j = i + k
            if direction == 'long':
                if l[j] <= sl_px:
                    exit_px, exit_bar, exit_reason = sl_px, j, 'sl'
                    break
                if h[j] >= tp_px:
                    exit_px, exit_bar, exit_reason = tp_px, j, 'tp'
                    break
            else:
                if h[j] >= sl_px:
                    exit_px, exit_bar, exit_reason = sl_px, j, 'sl'
                    break
                if l[j] <= tp_px:
                    exit_px, exit_bar, exit_reason = tp_px, j, 'tp'
                    break
        if not np.isfinite(exit_px):
            j = min(i + hold_bars, n - 1)
            exit_px = c[j]
            exit_bar = j
            exit_reason = 'timeout'
        if direction == 'long':
            pnl_price = exit_px - entry
        else:
            pnl_price = entry - exit_px
        pnl_pips = pnl_price / PIP - COST_PIPS
        pnl_usd = pnl_pips * USD_PER_PIP

        rows.append({
            'open_time': times[i], 'close_time': times[exit_bar],
            'direction': direction, 'entry_px': float(entry), 'exit_px': float(exit_px),
            'pnl_pips': float(pnl_pips), 'pnl': float(pnl_usd),
            'exit_reason': exit_reason,
        })
        cooldown_until = exit_bar  # avoid pyramiding

    log = pd.DataFrame(rows)
    if not len(log):
        return None, log
    pnls = log['pnl'].to_numpy()
    eq = np.cumsum(pnls)
    peaks = np.maximum.accumulate(eq)
    dd = (peaks - eq).max()
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum())
    return {
        'trades': int(len(log)),
        'pf': float(gp / gl) if gl > 0 else 0.0,
        'wr': float((pnls > 0).mean()),
        'net': float(pnls.sum()),
        'dd': float(dd),
    }, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="2020-01-01")
    ap.add_argument("--to", dest="d_to", default="2026-04-13")
    args = ap.parse_args()

    df = pd.read_parquet(DATA / "eurusd_m5_2020_2026.parquet")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df[(df['time'] >= args.d_from) & (df['time'] <= args.d_to)].reset_index(drop=True)

    # Resample M5 to M15 for ATR/EMA basis (then map back to M5 index for scanning)
    df15 = df.set_index('time').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    atr15 = _atr(df15['high'].to_numpy(), df15['low'].to_numpy(), df15['close'].to_numpy(), 14)
    ema15 = _ema(df15['close'].to_numpy(), 50)
    df15['atr15'] = atr15
    df15['ema15_50'] = ema15

    # Forward-fill the M15 indicators onto each M5 bar
    df = df.set_index('time')
    df['atr15'] = df15['atr15'].reindex(df.index, method='ffill')
    df['ema15_50'] = df15['ema15_50'].reindex(df.index, method='ffill')
    df = df.reset_index()
    print(f"loaded {len(df):,} M5 bars; M15 indicators ready")

    ctx = {
        'h': df['high'].to_numpy(), 'l': df['low'].to_numpy(),
        'c': df['close'].to_numpy(),
        'atr15': df['atr15'].to_numpy(),
        'ema15_50': df['ema15_50'].to_numpy(),
        'hours': df['time'].dt.hour.to_numpy(),
        'times': df['time'].to_numpy(),
    }

    rows = []
    grid = list(product(
        [1.0, 1.5, 2.0],         # ext_atr
        [1.0, 1.5, 2.0],         # sl_atr
        [0.4, 0.6, 0.8, 1.0],    # tp_atr
        [(21, 1), (22, 2), (20, 0)],  # session
        [24, 48],                # hold_bars (2hr or 4hr)
    ))
    print(f"grid: {len(grid)} cfgs\n")

    best = None
    best_log = None
    for ext_atr, sl_atr, tp_atr, sess, hb in grid:
        m, log = run_one(ctx, ext_atr, sl_atr, tp_atr, sess, hb)
        if m is None:
            continue
        row = {'ext_atr': ext_atr, 'sl_atr': sl_atr, 'tp_atr': tp_atr,
               'sess': f"{sess[0]}-{sess[1]}", 'hold': hb, **m}
        rows.append(row)
        if best is None or m['pf'] > best['pf']:
            if m['trades'] >= 100:
                best = row
                best_log = log

    df_results = pd.DataFrame(rows).sort_values('pf', ascending=False)
    out_csv = RESULTS / 'eur_asian_fade__grid.csv'
    df_results.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    print("\nTop 10:")
    print(df_results.head(10).to_string(index=False))
    if best is not None:
        print(f"\nBest cfg with trades>=100: {best}")
        best_log.to_csv(RESULTS / 'eur_asian_fade__best_trades.csv', index=False)


if __name__ == '__main__':
    main()
