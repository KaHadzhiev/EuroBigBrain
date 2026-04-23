#!/usr/bin/env python3
"""Null methodology diagnostic — why are EUR strategies failing the 5x null gate?

Runs 4 null types on momentum_short_NY (highest-conviction strategy):

  A. bar_shift:       circular-shift OHLCV by N bars (current method)
  B. hour_shuffle:    shuffle bars to random hours-of-day (kills session-time pattern)
  C. dir_flip:        invert long/short of every signal (tests direction edge)
  D. signal_shuffle:  keep prices, but pick random NEW signal times within same session
                      (tests if specific bar selection has edge vs any session bar)
  E. day_shuffle:     keep hour-of-day, shuffle which DATE each bar belongs to
                      (tests if specific date alignment has edge)

For each, run 30 trials, report PF distribution.

Goal: find which type of null makes the strategy DISTINGUISHABLE.
That tells us where the real edge lives (if any).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio" / "null_tests"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from eur_portfolio_run import build_ctx, simulate_signals, PIP, USD_PER_PIP, SPREAD_PIPS, SLIPPAGE_PIPS  # noqa: E402

CFG = {"entry_type": "momentum_short", "sess": "13-20",
       "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6,
       "bracket_offset": 0.3}
N_TRIALS = 30


def _pf(pnls):
    pnls = np.asarray(pnls)
    if not len(pnls): return 0.0
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum())
    return float(gp / gl) if gl > 0 else 0.0


def _real_run(df, cfg):
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    s, e = [int(x) for x in cfg["sess"].split("-")]
    cfg2 = {"entry_type": cfg["entry_type"], "vt": cfg["vt_vol"],
            "sess_start": s, "sess_end": e,
            "bracket_offset": cfg.get("bracket_offset", 0.3)}
    signals = generate_signals(ctx, cfg2, test_indices, vol_ratio)
    log = simulate_signals(ctx, signals, sl_atr=cfg["sl_atr"], tp_atr=cfg["tp_atr"],
                           hold_bars=cfg["hold"])
    return _pf(log["pnl"].to_numpy()), len(log), log


# ----- 5 null types -----

def null_A_bar_shift(df, cfg, seed):
    rng = np.random.default_rng(seed)
    n = len(df)
    shift = int(rng.integers(int(0.1 * n), int(0.9 * n)))
    df2 = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df2[col] = np.roll(df2[col].to_numpy(), shift)
    return _real_run(df2, cfg)


def null_B_hour_shuffle(df, cfg, seed):
    """Re-stamp every bar with a random hour-of-day. Kills session-time pattern."""
    rng = np.random.default_rng(seed)
    df2 = df.copy()
    times = df2['time'].copy()
    # shuffle just the hour part by adding a random offset (multiple of 15min) per bar
    n = len(df2)
    offsets = rng.integers(0, 96, size=n) * 15  # minutes
    df2['time'] = pd.to_datetime(df2['time']) + pd.to_timedelta(offsets, unit='m')
    df2 = df2.sort_values('time').reset_index(drop=True)
    return _real_run(df2, cfg)


def null_C_dir_flip(df, cfg, seed):
    """Run real strategy then INVERT direction of every trade."""
    real_pf, real_n, log = _real_run(df, cfg)
    if not len(log):
        return 0.0, 0, log
    log2 = log.copy()
    # Flip pnl sign while keeping COSTS subtracted (so flip-cost is realistic)
    raw_pips_per_trade = log2['pnl_pips'] + (SPREAD_PIPS + SLIPPAGE_PIPS)
    flipped_pips = -raw_pips_per_trade - (SPREAD_PIPS + SLIPPAGE_PIPS)
    log2['pnl_pips'] = flipped_pips
    log2['pnl'] = flipped_pips * USD_PER_PIP
    return _pf(log2['pnl'].to_numpy()), len(log2), log2


def null_D_signal_shuffle(df, cfg, seed):
    """Generate the strategy signals normally, then for each signal, REPLACE
    its bar index with a random other bar in the SAME session window.
    Tests: does the strategy pick BETTER bars than random within its session?"""
    rng = np.random.default_rng(seed)
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    s, e = [int(x) for x in cfg["sess"].split("-")]
    cfg2 = {"entry_type": cfg["entry_type"], "vt": cfg["vt_vol"],
            "sess_start": s, "sess_end": e,
            "bracket_offset": cfg.get("bracket_offset", 0.3)}
    signals = generate_signals(ctx, cfg2, test_indices, vol_ratio)
    if not signals:
        return 0.0, 0, pd.DataFrame()
    # Build pool of all bar indices in session window
    hours = ctx['hours']
    if e <= s:
        in_sess = (hours >= s) | (hours < e)
    else:
        in_sess = (hours >= s) & (hours < e)
    sess_idx = np.where(in_sess)[0]
    n_sigs = len(signals)
    # randomize signal bar to a random session bar (preserving direction)
    new_signals = []
    for sig in signals:
        old_i, buy_lvl, sell_lvl = sig
        new_i = int(rng.choice(sess_idx))
        # Recompute the level relative to new bar's price (otherwise the level
        # from old bar's price is meaningless on new bar)
        old_close = ctx['c_v'][old_i]
        new_close = ctx['c_v'][new_i]
        delta = new_close - old_close
        new_buy = buy_lvl + delta if buy_lvl is not None else None
        new_sell = sell_lvl + delta if sell_lvl is not None else None
        new_signals.append((new_i, new_buy, new_sell))
    log = simulate_signals(ctx, new_signals, sl_atr=cfg["sl_atr"], tp_atr=cfg["tp_atr"],
                           hold_bars=cfg["hold"])
    return _pf(log["pnl"].to_numpy()), len(log), log


def null_E_random_signals(df, cfg, seed):
    """Generate the SAME number of signals at random session bars with random
    directions. Tests: is the COUNT alone enough? Or is bar selection real?"""
    rng = np.random.default_rng(seed)
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    s, e = [int(x) for x in cfg["sess"].split("-")]
    cfg2 = {"entry_type": cfg["entry_type"], "vt": cfg["vt_vol"],
            "sess_start": s, "sess_end": e,
            "bracket_offset": cfg.get("bracket_offset", 0.3)}
    real_signals = generate_signals(ctx, cfg2, test_indices, vol_ratio)
    n_sigs = len(real_signals)
    if n_sigs == 0:
        return 0.0, 0, pd.DataFrame()
    hours = ctx['hours']
    if e <= s:
        in_sess = (hours >= s) | (hours < e)
    else:
        in_sess = (hours >= s) & (hours < e)
    sess_idx = np.where(in_sess)[0]
    # entry_type "momentum_short" means short-only. Replicate the direction pattern.
    n_long = sum(1 for s in real_signals if s[1] is not None)
    n_short = sum(1 for s in real_signals if s[2] is not None)
    new_signals = []
    bo = cfg.get("bracket_offset", 0.3)
    for i in range(n_sigs):
        bar_i = int(rng.choice(sess_idx))
        a = ctx['atr14'][bar_i]
        c = ctx['c_v'][bar_i]
        if not np.isfinite(a) or a <= 0:
            continue
        # match the direction proportion
        if i < n_long:
            new_signals.append((bar_i, c + bo * a, None))
        else:
            new_signals.append((bar_i, None, c - bo * a))
    log = simulate_signals(ctx, new_signals, sl_atr=cfg["sl_atr"], tp_atr=cfg["tp_atr"],
                           hold_bars=cfg["hold"])
    return _pf(log["pnl"].to_numpy()), len(log), log


def main():
    df = pd.read_parquet(DATA / "eurusd_m15_2020_2026.parquet")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"loaded {len(df):,} M15 bars\n")

    real_pf, real_n, _ = _real_run(df, CFG)
    print(f"REAL: PF={real_pf:.3f} trades={real_n}\n")

    null_types = [
        ("A: bar_shift (current)", null_A_bar_shift),
        ("B: hour_shuffle", null_B_hour_shuffle),
        ("C: dir_flip", null_C_dir_flip),
        ("D: signal_shuffle in session", null_D_signal_shuffle),
        ("E: random_signals in session", null_E_random_signals),
    ]
    results = {}
    for name, fn in null_types:
        t0 = time.time()
        pfs = []
        ns = []
        for i in range(N_TRIALS):
            pf, n, _ = fn(df, CFG, seed=42 + i)
            pfs.append(pf)
            ns.append(n)
        pfs = np.array(pfs); ns = np.array(ns)
        med = float(np.median(pfs))
        p5 = float(np.percentile(pfs, 5))
        p95 = float(np.percentile(pfs, 95))
        p_above = float((pfs >= real_pf).mean())
        edge = real_pf / med if med > 0 else float('inf')
        results[name] = {"pfs": pfs.tolist(), "ns": ns.tolist(),
                          "median": med, "p5": p5, "p95": p95,
                          "p_above": p_above, "edge_ratio": edge,
                          "n_mean": float(ns.mean()),
                          "elapsed": time.time() - t0}
        print(f"  {name:38s}  med={med:.3f}  p5={p5:.3f}  p95={p95:.3f}  "
              f"edge={edge:.2f}x  p(null>=real)={p_above:.3f}  n_avg={ns.mean():.0f}  "
              f"({time.time()-t0:.1f}s)")

    out = {"cfg": CFG, "real_pf": real_pf, "real_n": real_n,
           "n_trials": N_TRIALS, "results": results}
    op = RESULTS / "diagnostic_momentum_short_NY.json"
    op.write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
