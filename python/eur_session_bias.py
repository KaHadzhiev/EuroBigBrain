#!/usr/bin/env python3
"""Pure session-bias test — Breedon-Ranaldo prediction.

No SL, no TP, no indicators. Just buy at session open, sell at session close.
Two legs:
  - Long EUR  at 16:00 UTC (NY open),  exit 20:00 UTC (NY close-ish)
  - Short EUR at 08:00 UTC (London open), exit 14:00 UTC (NY pre-open)

This is what Breedon-Ranaldo's intraday FX flow paper predicts: corporates
buy USD in EU hours (EUR weakens) and reverse in NY hours (EUR strengthens).

Cost model: 1.3 pip Vantage spread + 0.2 pip slip = 1.5 pip per round-trip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"
RESULTS.mkdir(parents=True, exist_ok=True)

PIP = 0.0001
COST_PIPS = 1.5  # Vantage round-trip


def session_trades(df, side, entry_hour, exit_hour, entry_minute=0, exit_minute=0):
    """Generate one trade per day at the session window. Side = 'long' or 'short'."""
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute

    entry_mask = (df['hour'] == entry_hour) & (df['minute'] == entry_minute)
    exit_mask = (df['hour'] == exit_hour) & (df['minute'] == exit_minute)

    entries = df[entry_mask].set_index('date')
    exits = df[exit_mask].set_index('date')
    common = entries.index.intersection(exits.index)
    entries = entries.loc[common]
    exits = exits.loc[common]

    if side == 'long':
        pnl_price = exits['open'].to_numpy() - entries['open'].to_numpy()
    else:
        pnl_price = entries['open'].to_numpy() - exits['open'].to_numpy()

    pnl_pips = pnl_price / PIP - COST_PIPS
    pnl_usd = pnl_pips * 1.0  # $1 per pip on 0.10 lot equivalent

    out = pd.DataFrame({
        'open_time': entries['time'].to_numpy(),
        'close_time': exits['time'].to_numpy(),
        'entry_px': entries['open'].to_numpy(),
        'exit_px': exits['open'].to_numpy(),
        'pnl_pips': pnl_pips,
        'pnl': pnl_usd,
        'side': side,
    })
    return out


def metrics(log: pd.DataFrame) -> dict:
    if not len(log):
        return {'trades': 0, 'pf': 0.0, 'wr': 0.0, 'net': 0.0, 'dd': 0.0}
    pnls = log['pnl'].to_numpy()
    eq = np.cumsum(pnls)
    peaks = np.maximum.accumulate(eq)
    dd = (peaks - eq).max()
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum())
    return {
        'trades': int(len(pnls)),
        'pf': float(gp / gl) if gl > 0 else 0.0,
        'wr': float((pnls > 0).mean()),
        'net': float(pnls.sum()),
        'dd': float(dd),
        'avg_win': float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0.0,
        'avg_loss': float(pnls[pnls <= 0].mean()) if (pnls <= 0).any() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='d_from', default='2020-01-01')
    ap.add_argument('--to', dest='d_to', default='2026-04-13')
    args = ap.parse_args()

    df = pd.read_parquet(DATA / 'eurusd_m5_2020_2026.parquet')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df[(df['time'] >= args.d_from) & (df['time'] <= args.d_to)].reset_index(drop=True)
    print(f"loaded {len(df):,} M5 bars  span={df['time'].iloc[0]} -> {df['time'].iloc[-1]}\n")

    # Test variants — wider window, narrower window, with/without inversion
    tests = [
        ('NY_long_16-20',         'long',  16, 20),
        ('NY_long_15-20',         'long',  15, 20),
        ('NY_long_13-19',         'long',  13, 19),
        ('NY_long_14-21',         'long',  14, 21),
        ('EU_short_08-14',        'short',  8, 14),
        ('EU_short_07-13',        'short',  7, 13),
        ('EU_short_08-12',        'short',  8, 12),
        ('EU_short_07-11',        'short',  7, 11),
        ('Asian_long_22-07',      'long',  22, 7),  # Asian-Pac demand
        ('Late_NY_short_20-23',   'short', 20, 23),
    ]
    rows = []
    per_strategy_logs = {}
    for name, side, eh, xh in tests:
        log = session_trades(df, side, eh, xh)
        m = metrics(log)
        rows.append({'name': name, 'side': side, 'entry_h': eh, 'exit_h': xh, **m})
        per_strategy_logs[name] = log
        print(f"  {name:25s}  trades={m['trades']:4d}  PF={m['pf']:.3f}  WR={m['wr']:.3f}  "
              f"net=${m['net']:+7.0f}  DD=${m['dd']:6.0f}  avg_win=${m['avg_win']:+5.2f}  avg_loss=${m['avg_loss']:+5.2f}")

    df_results = pd.DataFrame(rows).sort_values('pf', ascending=False)
    out_csv = RESULTS / 'eur_session_bias.csv'
    df_results.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}\n")
    print("Top 3 by PF:")
    print(df_results.head(3).to_string(index=False))

    # Save per-strategy logs for the top-3 (for portfolio combination later)
    for name, log in per_strategy_logs.items():
        log.to_csv(RESULTS / f'eur_session_bias__{name}__trades.csv', index=False)

    # Also test the COMBINED long+short pair (offsetting sessions, no concurrent positions)
    print("\n=== Combined NY-long + EU-short pairs ===")
    for ny_name in ['NY_long_16-20', 'NY_long_15-20', 'NY_long_13-19']:
        for eu_name in ['EU_short_08-14', 'EU_short_07-13']:
            comb = pd.concat([per_strategy_logs[ny_name], per_strategy_logs[eu_name]], ignore_index=True)
            comb = comb.sort_values('open_time').reset_index(drop=True)
            m = metrics(comb)
            print(f"  {ny_name} + {eu_name}: trades={m['trades']} PF={m['pf']:.3f} WR={m['wr']:.3f} net=${m['net']:+.0f} DD=${m['dd']:.0f}")
    return 0


if __name__ == '__main__':
    main()
