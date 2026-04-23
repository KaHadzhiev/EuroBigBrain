#!/usr/bin/env python3
"""v7 portfolio + VOL-REGIME GATE study.

Hypothesis: 2020/2021 (and 2026 partial) bled because median daily ATR was low
(5.5-7.2 pip vs 9.1 in 2022). Cost is fixed, so low vol = high cost-drag.

Test: skip trades on days where daily ATR < N pips. Sweep N in [4, 5, 6, 7, 8].

If a vol gate boosts PF and trims the bad years without nuking the good years,
add it to v7.
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


def daily_atr_pips(df):
    """Compute daily median ATR(14) of M15 bars in pips."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    daily_atr = atr.resample('D').median() * 10000
    return daily_atr


def simulate_with_gate(df, ctx, vol_ratio, n, test_indices, daily_atr_pips_series, gate_pips):
    """Run all 3 v7 strategies, apply daily-ATR gate, combine."""
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
        # Apply gate: drop trades where the OPEN-TIME's daily ATR < gate_pips
        log['open_dt'] = pd.to_datetime(log['open_time'])
        log['date'] = log['open_dt'].dt.normalize()
        log['day_atr'] = log['date'].map(daily_atr_pips_series.to_dict())
        if gate_pips > 0:
            log = log[log['day_atr'].fillna(0) >= gate_pips].reset_index(drop=True)
        log["strategy"] = w["name"]; log["bucket"] = "NY"
        logs.append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())
    if not logs:
        return None
    return combine_strategies(logs, deposit=1000.0, max_concurrent_positions=2,
                               overlap_penalty=0.5)


def main():
    df = pd.read_parquet(DATA / "eurusd_m15_2020_2026.parquet")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"loaded {len(df):,} M15 bars")

    daily_atr = daily_atr_pips(df)
    print(f"daily ATR distribution (pips):  med={daily_atr.median():.1f}  p10={daily_atr.quantile(0.1):.1f}  p25={daily_atr.quantile(0.25):.1f}  p75={daily_atr.quantile(0.75):.1f}  p90={daily_atr.quantile(0.9):.1f}")

    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)

    print(f"\n{'gate_pips':>10}{'trades':>8}{'PF':>8}{'DD%':>8}{'Sharpe':>8}{'return%':>10}{'years_+':>10}")
    print("-" * 65)

    rows = []
    for gate in [0, 4, 5, 6, 7, 8]:
        m = simulate_with_gate(df, ctx, vol_ratio, n, test_indices, daily_atr, gate)
        if m is None:
            continue
        years_pos = sum(1 for y, n_t, pf, pn in m.per_year if pn > 0)
        rows.append({
            "gate_pips": gate, "n_trades": m.n_trades, "pf": m.pf,
            "max_dd_pct": m.max_dd_pct, "sharpe_daily": m.sharpe_daily,
            "return_pct": m.return_pct, "win_rate": m.win_rate,
            "years_pos": years_pos,
            "per_year": m.per_year,
        })
        print(f"{gate:>10}{m.n_trades:>8}{m.pf:>8.3f}{m.max_dd_pct:>8.1f}{m.sharpe_daily:>8.2f}{m.return_pct:>10.1f}{years_pos:>10}/7")

    print(f"\n=== Per-year by gate ===")
    print(f"{'gate':>5}", '  '.join(f'{y:>7}' for y in range(2020, 2027)))
    for r in rows:
        print(f"{r['gate_pips']:>5}", '  '.join(
            f"{(pn if pn else 0):>+7.0f}" for y, n_t, pf, pn in r['per_year'][:7]
        ))

    op = RESULTS / "eur_v7_vol_gate.json"
    op.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
