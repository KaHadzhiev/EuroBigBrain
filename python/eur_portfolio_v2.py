#!/usr/bin/env python3
"""EUR portfolio v2 -- consumes eur_sweep_top.json from eur_strategy_sweep.

For each strategy, picks the highest-PF qualifying config and builds a
multi-session portfolio. Optionally filters by min PF / min trades.

Output: results/portfolio/eur_portfolio_v2_summary.json + equity PNG.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from portfolio.combine import combine_strategies, write_artifacts  # noqa: E402
from eur_portfolio_run import build_ctx, simulate_signals  # noqa: E402

EXTRA_PARAMS_FALLBACK = {
    "asian_range":       {"max_asian_atr": 6.0},
    "breakout_range":    {"lookback": 12},
    "momentum_short":    {"bracket_offset": 0.3},
    "momentum_long":     {"bracket_offset": 0.3},
    "fade_long":         {"bracket_offset": 0.3},
    "fade_short":        {"bracket_offset": 0.3},
    "atr_bracket":       {"bracket_offset": 0.3},
    "vol_spike_bracket": {"bracket_offset": 0.3, "vol_mult": 2.0},
    "ema_cross_long":    {"bracket_offset": 0.3},
    "ema_cross_short":   {"bracket_offset": 0.3},
}


def _load_top():
    p = RESULTS / "eur_sweep_top.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p} -- run eur_strategy_sweep.py first")
    return json.loads(p.read_text(encoding="utf-8"))


def _pick_winners(top: dict, min_pf=1.10, min_trades=100) -> list:
    """For each strategy, take its top-1 config that meets gates."""
    winners = []
    for entry_type, candidates in top.items():
        if not candidates:
            continue
        # candidates already sorted by PF desc in sweep output
        for c in candidates:
            try:
                pf = float(c["pf"])
                tr = int(c["trades"])
            except Exception:
                continue
            if pf >= min_pf and tr >= min_trades:
                winners.append({
                    "name": f"{entry_type}__sess{c['sess']}_sl{c['sl_atr']}_tp{c['tp_atr']}_vt{c['vt_vol']}_h{c['hold']}",
                    "entry_type": entry_type,
                    "sess": c["sess"],
                    "sl_atr": float(c["sl_atr"]),
                    "tp_atr": float(c["tp_atr"]),
                    "vt_vol": float(c["vt_vol"]),
                    "hold": int(c["hold"]),
                    "expected_pf": pf,
                    "expected_trades": tr,
                })
                break  # only top-1 per strategy
    return winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="eur_portfolio_v2")
    ap.add_argument("--from", dest="d_from", default="2020-01-01")
    ap.add_argument("--to", dest="d_to", default="2026-04-13")
    ap.add_argument("--min-pf", type=float, default=1.10)
    ap.add_argument("--min-trades", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    top = _load_top()
    winners = _pick_winners(top, min_pf=args.min_pf, min_trades=args.min_trades)
    if not winners:
        print(f"[fail] no strategies passed PF>={args.min_pf} & trades>={args.min_trades}")
        return 2
    print(f"[winners] {len(winners)} strategies passed gate (PF>={args.min_pf}, trades>={args.min_trades})")
    for w in winners:
        print(f"  {w['name']}  expected PF={w['expected_pf']:.3f} trades={w['expected_trades']}")

    print(f"\n[load] {DATA / 'eurusd_m5_2020_2026.parquet'}")
    df = pd.read_parquet(DATA / "eurusd_m5_2020_2026.parquet")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[(df["time"] >= args.d_from) & (df["time"] <= args.d_to)].reset_index(drop=True)
    print(f"  rows: {len(df):,}")

    print("[build] ctx")
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)

    per_strategy_logs = []
    per_strategy_summary = []
    for w in winners:
        s0, s1 = [int(x) for x in w["sess"].split("-")]
        cfg = {
            "entry_type": w["entry_type"],
            "vt": w["vt_vol"],
            "sess_start": s0, "sess_end": s1,
            **EXTRA_PARAMS_FALLBACK.get(w["entry_type"], {}),
        }
        ts = time.time()
        signals = generate_signals(ctx, cfg, test_indices, vol_ratio)
        log = simulate_signals(ctx, signals, sl_atr=w["sl_atr"], tp_atr=w["tp_atr"],
                               hold_bars=w["hold"])
        log["strategy"] = w["name"]
        if not len(log):
            print(f"  {w['name']}: 0 trades on full 6yr -- skipping")
            continue
        gp = log.loc[log["pnl"] > 0, "pnl"].sum()
        gl = abs(log.loc[log["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0.0
        wr = (log["pnl"] > 0).mean()
        net = log["pnl"].sum()
        print(f"  {w['name']}: trades={len(log)} PF={pf:.3f} WR={wr:.3f} net=${net:+.0f} ({time.time()-ts:.1f}s)")
        per_strategy_summary.append({
            **w, "actual_trades": int(len(log)), "actual_pf": float(pf),
            "actual_wr": float(wr), "actual_net": float(net),
        })
        log.to_csv(RESULTS / f"{args.tag}__{w['name']}__trades.csv", index=False)
        per_strategy_logs.append(log[["open_time", "close_time", "pnl", "strategy"]].copy())

    if not per_strategy_logs:
        print("ERROR: no winner produced trades on the full period")
        return 1

    print(f"\n[combine] {len(per_strategy_logs)} strategies")
    metrics = combine_strategies(per_strategy_logs, deposit=1000.0,
                                 max_concurrent_positions=2,
                                 overlap_penalty=0.5)
    print(f"  {metrics.summary_line()}")
    print(f"  win_rate: {metrics.win_rate:.3f}  trades_per_month: {metrics.trades_per_month:.1f}")
    print("  per-year:")
    for y, n_t, pf, pn in metrics.per_year:
        print(f"    {y}  trades={n_t:5d}  PF={pf:.3f}  pnl=${pn:+.0f}")
    if metrics.correlation_matrix is not None and not metrics.correlation_matrix.empty:
        print("  correlation:")
        print(metrics.correlation_matrix.round(3))

    artifacts = write_artifacts(metrics, RESULTS, tag=args.tag)
    summary_path = RESULTS / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps({
        "tag": args.tag, "winners": per_strategy_summary,
        "combined": {
            "pf": metrics.pf, "n_trades": metrics.n_trades,
            "total_pnl": metrics.total_pnl, "ending_equity": metrics.ending_equity,
            "return_pct": metrics.return_pct, "max_dd_pct": metrics.max_dd_pct,
            "max_dd_usd": metrics.max_dd_usd, "recovery_factor": metrics.recovery_factor,
            "sharpe_daily": metrics.sharpe_daily, "trades_per_month": metrics.trades_per_month,
            "win_rate": metrics.win_rate, "overlap_hits": metrics.overlap_hit_count,
            "per_year": metrics.per_year,
        },
        "artifacts": artifacts,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] -> {summary_path}")
    print(f"[elapsed] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
