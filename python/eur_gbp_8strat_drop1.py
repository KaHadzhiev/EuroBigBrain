#!/usr/bin/env python3
"""Drop-1 robustness on 8-strategy cross-asset champion.

Removes each of the 8 strategies one at a time, reports portfolio metrics with the
remaining 7. Reveals which strategy contributes most / which is most replaceable.
"""
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
         "breakout_range": {"lookback": 12}}

STRATEGIES = [
    {"name": "EUR_M15_momentum_short_NY",  "data": "eurusd_m15_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "EUR_M15_ema_cross_short_NY", "data": "eurusd_m15_2020_2026.parquet", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "EUR_M30_breakout_NY",       "data": "eurusd_m30_2020_2026.parquet", "entry_type": "breakout_range",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "EUR_M30_momentum_short_Lon","data": "eurusd_m30_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "7-13",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "GBP_M15_momentum_long_NY",  "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "momentum_long",   "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12},
    {"name": "GBP_M15_momentum_short_NY", "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12},
    {"name": "GBP_M15_ema_cross_short_NY","data": "gbpusd_m15_2020_2026.parquet", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12},
    {"name": "GBP_M15_ema_cross_long_NY", "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "ema_cross_long",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12},
]
GATE_PIPS = 5
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


def run_one(w):
    df = load_df(w["data"])
    ctx = build_ctx(df)
    n = len(df); test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"], out=np.zeros(n), where=ctx["vol_ma20"]>0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    da = daily_atr(df).to_dict()
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
    log["strategy"] = w["name"]; log["bucket"] = "8s"
    return log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy()


def main():
    print("Computing all 8 strategy logs (cached)...")
    all_logs = {}
    for w in STRATEGIES:
        log = run_one(w)
        all_logs[w["name"]] = log
        n = len(log) if log is not None else 0
        print(f"  {w['name']:35s}  trades={n}")

    print(f"\n{'dropped':<35}{'PF':>7}{'DD%':>7}{'RF':>8}{'Sharpe':>8}{'PnL$':>10}{'tr':>7}")
    print('-' * 80)

    # Baseline: all 8
    logs = [l for l in all_logs.values() if l is not None]
    base = combine_strategies(logs, deposit=1000.0, max_concurrent_positions=4, overlap_penalty=0.5)
    print(f"  {'(none — all 8)':<33}{base.pf:>7.3f}{base.max_dd_pct:>7.1f}{base.recovery_factor:>8.2f}{base.sharpe_daily:>8.2f}{base.total_pnl:>+10.0f}{base.n_trades:>7}")

    # Drop-1 each
    rows = []
    for w in STRATEGIES:
        keep = [l for name, l in all_logs.items() if name != w["name"] and l is not None]
        if not keep: continue
        m = combine_strategies(keep, deposit=1000.0, max_concurrent_positions=4, overlap_penalty=0.5)
        delta_pnl = m.total_pnl - base.total_pnl
        delta_pf = m.pf - base.pf
        rows.append({"dropped": w["name"], "pf": m.pf, "dd_pct": m.max_dd_pct,
                     "rf": m.recovery_factor, "sharpe": m.sharpe_daily,
                     "pnl": m.total_pnl, "trades": m.n_trades,
                     "delta_pnl": delta_pnl, "delta_pf": delta_pf})
        print(f"  {w['name']:<33}{m.pf:>7.3f}{m.max_dd_pct:>7.1f}{m.recovery_factor:>8.2f}{m.sharpe_daily:>8.2f}{m.total_pnl:>+10.0f}{m.n_trades:>7}  (Δpnl={delta_pnl:+.0f}, ΔPF={delta_pf:+.3f})")

    op = RESULTS / "eur_gbp_8strat_drop1.json"
    op.write_text(json.dumps({
        "baseline": {"pf": base.pf, "dd_pct": base.max_dd_pct, "rf": base.recovery_factor,
                     "sharpe": base.sharpe_daily, "pnl": base.total_pnl, "trades": base.n_trades},
        "drop_1": rows,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {op}")

    # Best drop = highest PnL when removed = drop hurts least = most replaceable
    by_delta = sorted(rows, key=lambda r: r['delta_pnl'])
    print(f"\nMost VALUABLE strategies (largest PnL drop when removed):")
    for r in by_delta[:3]:
        print(f"  {r['dropped']:35s}  drops PnL by {r['delta_pnl']:+.0f}")
    print(f"\nLeast valuable (smallest PnL impact, possibly trim?):")
    for r in by_delta[-3:]:
        print(f"  {r['dropped']:35s}  drops PnL by {r['delta_pnl']:+.0f}")


if __name__ == "__main__":
    main()
