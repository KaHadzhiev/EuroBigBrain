#!/usr/bin/env python3
"""Walk-forward on M30 v2 portfolio (the production candidate)."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals
from portfolio.combine import combine_strategies
from eur_portfolio_run import build_ctx, simulate_signals

EXTRA = {"momentum_short": {"bracket_offset": 0.3},
         "breakout_range": {"lookback": 12}}
WINNERS = [
    {"name": "breakout_range_NY_M30",     "entry_type": "breakout_range",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "momentum_short_London_M30", "entry_type": "momentum_short",  "sess": "7-13",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
]
GATE_PIPS = 5
FOLDS = [
    ("2022-01-01", "2022-07-01"), ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"), ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01"),
    ("2026-01-01", "2026-04-22"),
]


def daily_atr(df_subset):
    df = df_subset.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return (tr.rolling(14, min_periods=14).mean().resample('D').median() * 10000)


def run_window(df, start, end):
    df_win = df[(df['time'] >= start) & (df['time'] < end)].reset_index(drop=True)
    if len(df_win) < 100:
        return None
    ctx = build_ctx(df_win)
    n = len(df_win); test_indices = np.arange(n)
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
        if not len(log): continue
        log['open_dt'] = pd.to_datetime(log['open_time'])
        log['date'] = log['open_dt'].dt.normalize()
        log['day_atr'] = log['date'].map(da)
        log = log[log['day_atr'].fillna(0) >= GATE_PIPS].reset_index(drop=True)
        if not len(log): continue
        log["strategy"] = w["name"]; log["bucket"] = "NY"
        logs.append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())
    if not logs: return None
    return combine_strategies(logs, deposit=1000.0, max_concurrent_positions=2,
                               overlap_penalty=0.5)


def main():
    df = pd.read_parquet(DATA / "eurusd_m30_2020_2026.parquet")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"loaded {len(df):,} M30 bars")
    print(f"\nfold                        trades   PF      WR    DD%  Sharpe   PnL$")
    print('-' * 75)
    rows = []
    for s, e in FOLDS:
        m = run_window(df, s, e)
        if m is None:
            print(f"  {s} -> {e}  no trades")
            continue
        rows.append({"start": s, "end": e, "n_trades": m.n_trades,
                     "pf": m.pf, "max_dd_pct": m.max_dd_pct,
                     "sharpe": m.sharpe_daily, "pnl": m.total_pnl})
        print(f"  {s} -> {e}  {m.n_trades:>6}  {m.pf:>6.3f}  {m.win_rate:>5.2f}  "
              f"{m.max_dd_pct:>5.1f}  {m.sharpe_daily:>6.2f}  {m.total_pnl:>+8.0f}")
    op = RESULTS / "eur_m30_v2_walkforward.json"
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
