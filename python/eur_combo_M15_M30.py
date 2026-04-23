#!/usr/bin/env python3
"""Cross-timeframe combo: M15 v9 (2 NY shorts) + M30 v2 (breakout NY + momentum_short London).

Hypothesis: M15 and M30 strategies fire on different bars, may give better diversification
than either alone. Total of 4 strategies across 2 timeframes."""
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
         "ema_cross_short": {"bracket_offset": 0.3},
         "breakout_range": {"lookback": 12}}

# M15 v9 strategies
M15_WINNERS = [
    {"name": "v9_momentum_short_NY_M15",  "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6, "tf": "M15"},
    {"name": "v9_ema_cross_short_NY_M15", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6, "tf": "M15"},
]
# M30 v2 strategies
M30_WINNERS = [
    {"name": "m30v2_breakout_NY_M30",          "entry_type": "breakout_range",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6, "tf": "M30"},
    {"name": "m30v2_momentum_short_London_M30", "entry_type": "momentum_short", "sess": "7-13",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6, "tf": "M30"},
]
GATE_PIPS = 5


def daily_atr(df_subset):
    df = df_subset.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return (tr.rolling(14, min_periods=14).mean().resample('D').median() * 10000)


def run_one(df, w):
    ctx = build_ctx(df)
    n = len(df); test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    da = daily_atr(df).to_dict()
    s, e = [int(x) for x in w["sess"].split("-")]
    cfg = {"entry_type": w["entry_type"], "vt": w["vt_vol"],
           "sess_start": s, "sess_end": e,
           **EXTRA.get(w["entry_type"], {})}
    signals = generate_signals(ctx, cfg, test_indices, vol_ratio)
    log = simulate_signals(ctx, signals, sl_atr=w["sl_atr"], tp_atr=w["tp_atr"],
                           hold_bars=w["hold"])
    log['open_dt'] = pd.to_datetime(log['open_time'])
    log['date'] = log['open_dt'].dt.normalize()
    log['day_atr'] = log['date'].map(da)
    log = log[log['day_atr'].fillna(0) >= GATE_PIPS].reset_index(drop=True)
    log["strategy"] = w["name"]; log["bucket"] = w["tf"]
    return log


def main():
    tag = "eur_combo_M15_M30"
    df_m15 = pd.read_parquet(DATA / "eurusd_m15_2020_2026.parquet")
    df_m15['time'] = pd.to_datetime(df_m15['time'])
    df_m15 = df_m15.sort_values('time').reset_index(drop=True)
    df_m30 = pd.read_parquet(DATA / "eurusd_m30_2020_2026.parquet")
    df_m30['time'] = pd.to_datetime(df_m30['time'])
    df_m30 = df_m30.sort_values('time').reset_index(drop=True)
    print(f"M15: {len(df_m15):,}  M30: {len(df_m30):,}")

    logs = []
    for w in M15_WINNERS:
        log = run_one(df_m15, w)
        gp = log.loc[log["pnl"] > 0, "pnl"].sum()
        gl = abs(log.loc[log["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0
        net = log["pnl"].sum()
        print(f"  {w['name']:35s}  trades={len(log):4d}  PF={pf:.3f}  net=${net:+.0f}")
        log.to_csv(RESULTS / f"{tag}__{w['name']}__trades.csv", index=False)
        logs.append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())
    for w in M30_WINNERS:
        log = run_one(df_m30, w)
        gp = log.loc[log["pnl"] > 0, "pnl"].sum()
        gl = abs(log.loc[log["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0
        net = log["pnl"].sum()
        print(f"  {w['name']:35s}  trades={len(log):4d}  PF={pf:.3f}  net=${net:+.0f}")
        log.to_csv(RESULTS / f"{tag}__{w['name']}__trades.csv", index=False)
        logs.append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())

    print("\n[combine cross-timeframe]")
    m = combine_strategies(logs, deposit=1000.0, max_concurrent_positions=3, overlap_penalty=0.5)
    print(f"  {m.summary_line()}")
    print(f"  WR={m.win_rate:.3f}  trades/mo={m.trades_per_month:.1f}")
    print("  per-year:")
    for y, n_t, pf, pn in m.per_year:
        print(f"    {y}  trades={n_t:5d}  PF={pf:.3f}  pnl=${pn:+.0f}")
    if m.correlation_matrix is not None:
        mat = m.correlation_matrix.values
        mean_off = mat[~np.eye(len(mat), dtype=bool)].mean()
        print(f"  mean off-diag corr = {mean_off:.3f}")
        print("  full corr matrix:")
        print(m.correlation_matrix.round(3))
    artifacts = write_artifacts(m, RESULTS, tag=tag)
    op = RESULTS / f"{tag}_summary.json"
    op.write_text(json.dumps({"tag": tag,
        "combined": {"pf": m.pf, "n_trades": m.n_trades, "max_dd_pct": m.max_dd_pct,
                     "sharpe_daily": m.sharpe_daily, "trades_per_month": m.trades_per_month,
                     "win_rate": m.win_rate, "per_year": m.per_year,
                     "ending_equity": m.ending_equity, "return_pct": m.return_pct,
                     "recovery_factor": m.recovery_factor},
        "artifacts": artifacts}, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
