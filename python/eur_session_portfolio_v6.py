#!/usr/bin/env python3
"""EUR M15 portfolio v6 — null-validated only, drops ema_cross_long FullDay culprit.

Final 4 strategies:
  - momentum_short_NY (12.1x null)
  - ema_cross_short_NY (8.2x null)
  - momentum_long_NY (7.4x null)
  - ema_cross_long_NY (need separate null test, but real PF 1.14)

All 4 in the NY 13-20 GMT block.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from portfolio.combine import combine_strategies, write_artifacts  # noqa: E402
from eur_portfolio_run import build_ctx, simulate_signals  # noqa: E402

EXTRA_PARAMS = {
    "momentum_short": {"bracket_offset": 0.3},
    "momentum_long": {"bracket_offset": 0.3},
    "ema_cross_long": {"bracket_offset": 0.3},
    "ema_cross_short": {"bracket_offset": 0.3},
}

WINNERS = [
    # All 5x null gate PASS strategies
    {"name": "momentum_short_NY",  "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "ema_cross_short_NY", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "momentum_long_NY",   "entry_type": "momentum_long",   "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    # ema_cross_long_NY DROPPED — null edge only 4.13x (gate is 5x)
]


def main():
    tag = "eur_24h_v6"
    print(f"[{tag}] {len(WINNERS)} null-validated NY strategies")
    df = pd.read_parquet(DATA / "eurusd_m15_2020_2026.parquet")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"loaded {len(df):,} M15 bars")
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)

    logs = []
    for w in WINNERS:
        s, e = [int(x) for x in w["sess"].split("-")]
        cfg = {"entry_type": w["entry_type"], "vt": w["vt_vol"],
               "sess_start": s, "sess_end": e,
               **EXTRA_PARAMS.get(w["entry_type"], {})}
        signals = generate_signals(ctx, cfg, test_indices, vol_ratio)
        log = simulate_signals(ctx, signals, sl_atr=w["sl_atr"], tp_atr=w["tp_atr"],
                               hold_bars=w["hold"])
        log["strategy"] = w["name"]; log["bucket"] = "NY"
        gp = log.loc[log["pnl"] > 0, "pnl"].sum()
        gl = abs(log.loc[log["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0.0
        wr = (log["pnl"] > 0).mean()
        net = log["pnl"].sum()
        print(f"  {w['name']:25s} trades={len(log):4d} PF={pf:.3f} WR={wr:.3f} net=${net:+.0f}")
        log.to_csv(RESULTS / f"{tag}__{w['name']}__trades.csv", index=False)
        logs.append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())

    print("\n[combine 24h]")
    m = combine_strategies(logs, deposit=1000.0, max_concurrent_positions=2, overlap_penalty=0.5)
    print(f"  {m.summary_line()}")
    print(f"  win_rate: {m.win_rate:.3f}  trades_per_month: {m.trades_per_month:.1f}")
    print("  per-year:")
    for y, n_t, pf, pn in m.per_year:
        print(f"    {y}  trades={n_t:5d}  PF={pf:.3f}  pnl=${pn:+.0f}")
    if m.correlation_matrix is not None:
        mat = m.correlation_matrix.values
        mean_off = mat[~np.eye(len(mat), dtype=bool)].mean()
        print(f"\n  mean off-diag corr = {mean_off:.3f}")
    artifacts = write_artifacts(m, RESULTS, tag=tag)
    out = {
        "tag": tag, "winners": WINNERS,
        "combined": {
            "pf": m.pf, "n_trades": m.n_trades, "total_pnl": m.total_pnl,
            "ending_equity": m.ending_equity, "return_pct": m.return_pct,
            "max_dd_pct": m.max_dd_pct, "max_dd_usd": m.max_dd_usd,
            "recovery_factor": m.recovery_factor, "sharpe_daily": m.sharpe_daily,
            "trades_per_month": m.trades_per_month, "win_rate": m.win_rate,
            "per_year": m.per_year,
        },
        "artifacts": artifacts,
    }
    op = RESULTS / f"{tag}_summary.json"
    op.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {op}")


if __name__ == "__main__":
    main()
