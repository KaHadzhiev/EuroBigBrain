#!/usr/bin/env python3
"""EUR M30 portfolio v2 — only the 2 NULL-VALIDATED strategies (5x gate)."""
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
         "breakout_range": {"lookback": 12}}

WINNERS = [
    {"name": "breakout_range_NY_M30",     "entry_type": "breakout_range",  "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6,  "null_edge": 29.68},
    {"name": "momentum_short_London_M30", "entry_type": "momentum_short",  "sess": "7-13",  "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6,  "null_edge": 7.07},
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


def main():
    tag = "eur_m30_v2_validated"
    df = pd.read_parquet(DATA / "eurusd_m30_2020_2026.parquet")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"loaded {len(df):,} M30 bars")
    ctx = build_ctx(df)
    n = len(df); test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    da = daily_atr(df).to_dict()

    logs = []
    for w in WINNERS:
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
        log["strategy"] = w["name"]; log["bucket"] = w["sess"]
        gp = log.loc[log["pnl"] > 0, "pnl"].sum()
        gl = abs(log.loc[log["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0
        wr = (log["pnl"] > 0).mean()
        net = log["pnl"].sum()
        print(f"  {w['name']:30s}  null_edge={w['null_edge']:5.2f}x  trades={len(log):4d}  PF={pf:.3f}  WR={wr:.3f}  net=${net:+.0f}")
        log.to_csv(RESULTS / f"{tag}__{w['name']}__trades.csv", index=False)
        logs.append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())

    print("\n[combine M30 v2 — null-validated only]")
    m = combine_strategies(logs, deposit=1000.0, max_concurrent_positions=2, overlap_penalty=0.5)
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
    op.write_text(json.dumps({"tag": tag, "winners": WINNERS,
        "combined": {"pf": m.pf, "n_trades": m.n_trades, "max_dd_pct": m.max_dd_pct,
                     "sharpe_daily": m.sharpe_daily, "trades_per_month": m.trades_per_month,
                     "win_rate": m.win_rate, "per_year": m.per_year,
                     "ending_equity": m.ending_equity, "return_pct": m.return_pct,
                     "recovery_factor": m.recovery_factor},
        "artifacts": artifacts}, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
