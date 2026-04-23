#!/usr/bin/env python3
"""Walk-forward on the 10-strategy production champion."""
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
         "momentum_long": {"bracket_offset": 0.3},
         "ema_cross_short": {"bracket_offset": 0.3},
         "ema_cross_long": {"bracket_offset": 0.3},
         "breakout_range": {"lookback": 12},
         "asian_range": {"max_asian_atr": 6.0}}

STRATEGIES = [
    {"name": "EUR_M15_momentum_short_NY",  "data": "eurusd_m15_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "EUR_M15_ema_cross_short_NY", "data": "eurusd_m15_2020_2026.parquet", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "EUR_M30_breakout_NY",       "data": "eurusd_m30_2020_2026.parquet", "entry_type": "breakout_range",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "EUR_M30_momentum_short_Lon","data": "eurusd_m30_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "7-13",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "GBP_M15_momentum_long_NY",  "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "momentum_long",   "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12},
    {"name": "GBP_M15_momentum_short_NY", "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12},
    {"name": "GBP_M15_ema_cross_short_NY","data": "gbpusd_m15_2020_2026.parquet", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12},
    {"name": "GBP_M15_ema_cross_long_NY", "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "ema_cross_long",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12},
    {"name": "GBP_M30_breakout_FullDay", "data": "gbpusd_m30_2020_2026.parquet", "entry_type": "breakout_range",  "sess": "7-20",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 1.5, "hold": 24},
    {"name": "GBP_M30_asian_range_London","data": "gbpusd_m30_2020_2026.parquet", "entry_type": "asian_range",    "sess": "7-13",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 24},
]
GATE_PIPS = 5
FOLDS = [
    ("2022-01-01","2022-07-01"),("2022-07-01","2023-01-01"),
    ("2023-01-01","2023-07-01"),("2023-07-01","2024-01-01"),
    ("2024-01-01","2024-07-01"),("2024-07-01","2025-01-01"),
    ("2025-01-01","2025-07-01"),("2025-07-01","2026-01-01"),
    ("2026-01-01","2026-04-22"),
]
DATA_CACHE = {}


def load_df(data_file):
    if data_file in DATA_CACHE:
        return DATA_CACHE[data_file]
    df = pd.read_parquet(DATA / data_file)
    df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    DATA_CACHE[data_file] = df
    return df


def daily_atr(df_subset):
    df = df_subset.copy()
    df = df.set_index('time').sort_index()
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    s = (tr.rolling(14, min_periods=14).mean().resample('D').median() * 10000)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s


def run_strategy_in_window(w, start, end):
    df = load_df(w["data"])
    df_win = df[(df['time'] >= start) & (df['time'] < end)].reset_index(drop=True)
    if len(df_win) < 100: return None
    ctx = build_ctx(df_win)
    n = len(df_win); test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"], out=np.zeros(n), where=ctx["vol_ma20"]>0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    da = daily_atr(df_win).to_dict()
    s, e = [int(x) for x in w["sess"].split("-")]
    cfg = {"entry_type": w["entry_type"], "vt": w["vt_vol"],
           "sess_start": s, "sess_end": e,
           **EXTRA.get(w["entry_type"], {})}
    signals = generate_signals(ctx, cfg, test_indices, vol_ratio)
    log = simulate_signals(ctx, signals, sl_atr=w["sl_atr"], tp_atr=w["tp_atr"], hold_bars=w["hold"])
    if not len(log): return None
    log['open_dt'] = pd.to_datetime(log['open_time'])
    log['date'] = log['open_dt'].dt.normalize()
    log['day_atr'] = log['date'].map(da)
    log = log[log['day_atr'].fillna(0) >= GATE_PIPS].reset_index(drop=True)
    if not len(log): return None
    log["strategy"] = w["name"]; log["bucket"] = "10s"
    return log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy()


def run_combined(start, end):
    logs = []
    for w in STRATEGIES:
        log = run_strategy_in_window(w, start, end)
        if log is not None and len(log):
            logs.append(log)
    if not logs: return None
    return combine_strategies(logs, deposit=1000.0, max_concurrent_positions=5, overlap_penalty=0.5)


def main():
    print(f"=== 10-strategy walk-forward ===")
    print(f"\nfold                       trades   PF    WR    DD%  Sharpe   PnL$")
    print('-' * 75)
    rows = []
    for s, e in FOLDS:
        m = run_combined(s, e)
        if m is None:
            print(f"  {s} no trades"); continue
        rows.append({"start": s, "end": e, "n_trades": m.n_trades,
                     "pf": m.pf, "wr": m.win_rate, "dd": m.max_dd_pct,
                     "sharpe": m.sharpe_daily, "pnl": m.total_pnl})
        print(f"  {s}->{e}  {m.n_trades:>5}  {m.pf:>6.3f}  {m.win_rate:>4.2f}  {m.max_dd_pct:>5.1f}  {m.sharpe_daily:>6.2f}  {m.total_pnl:>+8.0f}")
    op = RESULTS / "eur_gbp_10strat_walkforward.json"
    op.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    n_pf_ge_1 = sum(1 for r in rows if r['pf'] >= 1.0)
    n_pf_ge_13 = sum(1 for r in rows if r['pf'] >= 1.3)
    print(f"\nSUMMARY: {len(rows)} folds")
    print(f"  PF >= 1.0: {n_pf_ge_1}/{len(rows)}")
    print(f"  PF >= 1.3: {n_pf_ge_13}/{len(rows)}")
    print(f"  median PF: {np.median([r['pf'] for r in rows]):.3f}")
    print(f"  total PnL: ${sum(r['pnl'] for r in rows):+.0f}")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
