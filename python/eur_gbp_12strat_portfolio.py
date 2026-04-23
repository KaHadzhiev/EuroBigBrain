#!/usr/bin/env python3
"""12-strategy portfolio: 10-strat + 2 NEW null-validated BB squeeze strategies."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals
from portfolio.combine import combine_strategies, write_artifacts
from eur_portfolio_run import build_ctx, simulate_signals

EXTRA = {"momentum_short": {"bracket_offset": 0.3},
         "momentum_long": {"bracket_offset": 0.3},
         "ema_cross_short": {"bracket_offset": 0.3},
         "ema_cross_long": {"bracket_offset": 0.3},
         "breakout_range": {"lookback": 12},
         "asian_range": {"max_asian_atr": 6.0},
         "bb_squeeze_long": {"bracket_offset": 0.3},
         "bb_squeeze_short": {"bracket_offset": 0.3}}

STRATEGIES = [
    {"name": "EUR_M15_momentum_short_NY",  "data": "eurusd_m15_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6,  "null_edge": 12.13},
    {"name": "EUR_M15_ema_cross_short_NY", "data": "eurusd_m15_2020_2026.parquet", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6,  "null_edge": 8.20},
    {"name": "EUR_M30_breakout_NY",       "data": "eurusd_m30_2020_2026.parquet", "entry_type": "breakout_range",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6,  "null_edge": 29.68},
    {"name": "EUR_M30_momentum_short_Lon","data": "eurusd_m30_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "7-13",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6,  "null_edge": 7.07},
    {"name": "GBP_M15_momentum_long_NY",  "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "momentum_long",   "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12, "null_edge": 22.22},
    {"name": "GBP_M15_momentum_short_NY", "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12, "null_edge": 15.65},
    {"name": "GBP_M15_ema_cross_short_NY","data": "gbpusd_m15_2020_2026.parquet", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12, "null_edge": 8.74},
    {"name": "GBP_M15_ema_cross_long_NY", "data": "gbpusd_m15_2020_2026.parquet", "entry_type": "ema_cross_long",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 12, "null_edge": 6.57},
    {"name": "GBP_M30_breakout_FullDay", "data": "gbpusd_m30_2020_2026.parquet", "entry_type": "breakout_range",  "sess": "7-20",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 1.5, "hold": 24, "null_edge": 17.22},
    {"name": "GBP_M30_asian_range_London","data": "gbpusd_m30_2020_2026.parquet", "entry_type": "asian_range",    "sess": "7-13",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 24, "null_edge": 13.35},
    # NEW (2026-04-23 BB squeeze)
    {"name": "EUR_M15_bb_squeeze_short_FullDay", "data": "eurusd_m15_2020_2026.parquet", "entry_type": "bb_squeeze_short", "sess": "7-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 1.5, "hold": 6, "null_edge": float('inf')},
    {"name": "EUR_M30_bb_squeeze_long_FullDay",  "data": "eurusd_m30_2020_2026.parquet", "entry_type": "bb_squeeze_long",  "sess": "7-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6, "null_edge": float('inf')},
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
    log["strategy"] = w["name"]
    log["bucket"] = w["data"][:3].upper()
    return log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy()


def main():
    tag = "eur_gbp_12strat_v1"
    logs = []
    print("=== 12-strategy cross-asset portfolio ===")
    for w in STRATEGIES:
        log = run_one(w)
        if log is None or not len(log):
            print(f"  {w['name']:42s}  no trades")
            continue
        gp = log.loc[log["pnl"] > 0, "pnl"].sum()
        gl = abs(log.loc[log["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0
        net = log["pnl"].sum()
        print(f"  {w['name']:42s}  trades={len(log):4d}  PF={pf:.3f}  net=${net:+.0f}")
        log.to_csv(RESULTS / f"{tag}__{w['name']}__trades.csv", index=False)
        logs.append(log)

    print("\n[combine 12-strat]")
    m = combine_strategies(logs, deposit=1000.0, max_concurrent_positions=6, overlap_penalty=0.5)
    print(f"  {m.summary_line()}")
    print(f"  WR={m.win_rate:.3f}  trades/mo={m.trades_per_month:.1f}")
    print("  per-year:")
    for y, n_t, pf, pn in m.per_year:
        print(f"    {y}  trades={n_t:5d}  PF={pf:.3f}  pnl=${pn:+.0f}")
    if m.correlation_matrix is not None:
        mat = m.correlation_matrix.values
        mean_off = mat[~np.eye(len(mat), dtype=bool)].mean()
        print(f"  mean off-diag corr = {mean_off:.3f}")
    artifacts = write_artifacts(m, RESULTS, tag=tag)
    op = RESULTS / f"{tag}_summary.json"
    op.write_text(json.dumps({"tag": tag, "strategies": STRATEGIES,
        "combined": {"pf": m.pf, "n_trades": m.n_trades, "max_dd_pct": m.max_dd_pct,
                     "sharpe_daily": m.sharpe_daily, "trades_per_month": m.trades_per_month,
                     "win_rate": m.win_rate, "per_year": m.per_year,
                     "ending_equity": m.ending_equity, "return_pct": m.return_pct,
                     "recovery_factor": m.recovery_factor, "total_pnl": m.total_pnl},
        "artifacts": artifacts}, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
