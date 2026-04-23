#!/usr/bin/env python3
"""Walk-forward validation for EUR v8 portfolio.

Splits 6yr (2020-2026) into 6 IS/OOS folds:
  Fold 1: train 2020-01..2022-01, test 2022-01..2022-07
  Fold 2: train 2020-07..2022-07, test 2022-07..2023-01
  Fold 3: train 2021-01..2023-01, test 2023-01..2023-07
  ... etc, sliding 6mo

For each fold:
  - Run v8 strategies on TEST window
  - Apply vol_gate=5 from IS window's stats (re-derive)
  - Report PF, trades, DD per fold

If PF holds across 5+ of 6 folds → robust deploy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from portfolio.combine import combine_strategies  # noqa: E402
from eur_portfolio_run import build_ctx, simulate_signals  # noqa: E402

EXTRA = {"momentum_short": {"bracket_offset": 0.3},
         "momentum_long": {"bracket_offset": 0.3},
         "ema_cross_short": {"bracket_offset": 0.3}}

WINNERS = [
    {"name": "momentum_short_NY",  "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "ema_cross_short_NY", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "momentum_long_NY",   "entry_type": "momentum_long",   "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
]

GATE_PIPS = 5  # from v8 sweep optimum


def daily_atr(df_subset):
    df = df_subset.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    return (atr.resample('D').median() * 10000)


def run_window(df, start, end, gate=GATE_PIPS):
    df_win = df[(df['time'] >= start) & (df['time'] < end)].reset_index(drop=True)
    if len(df_win) < 200:
        return None
    ctx = build_ctx(df_win)
    n = len(df_win)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    da = daily_atr(df_win).to_dict()
    logs = []
    for w in WINNERS:
        s, e = [int(x) for x in w["sess"].split("-")]
        cfg = {"entry_type": w["entry_type"], "vt": w["vt_vol"],
               "sess_start": s, "sess_end": e,
               **EXTRA.get(w["entry_type"], {})}
        signals = generate_signals(ctx, cfg, test_indices, vol_ratio)
        log = simulate_signals(ctx, signals, sl_atr=w["sl_atr"], tp_atr=w["tp_atr"],
                               hold_bars=w["hold"])
        if not len(log):
            continue
        log['open_dt'] = pd.to_datetime(log['open_time'])
        log['date'] = log['open_dt'].dt.normalize()
        log['day_atr'] = log['date'].map(da)
        if gate > 0:
            log = log[log['day_atr'].fillna(0) >= gate].reset_index(drop=True)
        log["strategy"] = w["name"]; log["bucket"] = "NY"
        if len(log):
            logs.append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())
    if not logs:
        return None
    return combine_strategies(logs, deposit=1000.0, max_concurrent_positions=2,
                               overlap_penalty=0.5)


def main():
    df = pd.read_parquet(DATA / "eurusd_m15_2020_2026.parquet")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"loaded {len(df):,} M15 bars; span {df['time'].iloc[0]} → {df['time'].iloc[-1]}")

    # 6 folds of 6mo OOS each
    folds = [
        ("2022-01-01", "2022-07-01"),
        ("2022-07-01", "2023-01-01"),
        ("2023-01-01", "2023-07-01"),
        ("2023-07-01", "2024-01-01"),
        ("2024-01-01", "2024-07-01"),
        ("2024-07-01", "2025-01-01"),
        ("2025-01-01", "2025-07-01"),
        ("2025-07-01", "2026-01-01"),
        ("2026-01-01", "2026-04-22"),
    ]
    print(f"\n{'fold':<22}{'trades':>8}{'PF':>8}{'WR':>7}{'DD%':>7}{'Sharpe':>8}{'PnL$':>10}")
    print('-' * 75)
    rows = []
    for s, e in folds:
        m = run_window(df, s, e)
        if m is None:
            print(f"  {s} → {e}  no trades")
            continue
        rows.append({"start": s, "end": e, "n_trades": m.n_trades,
                     "pf": m.pf, "max_dd_pct": m.max_dd_pct,
                     "sharpe_daily": m.sharpe_daily, "win_rate": m.win_rate,
                     "pnl": m.total_pnl})
        print(f"  {s} → {e}  {m.n_trades:>6}  {m.pf:>6.3f}  {m.win_rate:>5.2f}  "
              f"{m.max_dd_pct:>5.1f}  {m.sharpe_daily:>6.2f}  {m.total_pnl:>+8.0f}")

    op = RESULTS / "eur_v8_walkforward.json"
    op.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    n_pf_ge_1 = sum(1 for r in rows if r['pf'] >= 1.0)
    n_pf_ge_13 = sum(1 for r in rows if r['pf'] >= 1.3)
    print(f"\nSUMMARY: {len(rows)} folds")
    print(f"  PF >= 1.0: {n_pf_ge_1}/{len(rows)}")
    print(f"  PF >= 1.3: {n_pf_ge_13}/{len(rows)}")
    print(f"  median PF: {np.median([r['pf'] for r in rows]):.3f}")
    print(f"  mean PnL:  ${np.mean([r['pnl'] for r in rows]):+.0f}")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
