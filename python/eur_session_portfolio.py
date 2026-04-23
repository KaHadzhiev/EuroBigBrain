#!/usr/bin/env python3
"""EUR M15 24-hour session portfolio (Hatton-style).

Reads sweep_top.json, picks PF>=1.10 cfgs, GROUPS by session, builds
per-session portfolios (London 7-13, NY 13-20, Asian 20-7), then
combines into a 24h portfolio.

Outputs:
  results/portfolio/eur_24h__<session>__trades.csv  (per session combined)
  results/portfolio/eur_24h__per_session_summary.json
  results/portfolio/eur_24h_combined__summary.json
  results/portfolio/eur_24h_combined__equity.png
"""
from __future__ import annotations

import argparse
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
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from portfolio.combine import combine_strategies, write_artifacts  # noqa: E402
from eur_portfolio_run import build_ctx, simulate_signals  # noqa: E402

EXTRA_PARAMS = {
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

# How to assign a sess "7-13" to a session bucket
SESSION_BUCKETS = {
    "London":     [(7, 13), (8, 12), (7, 11)],
    "NY":         [(13, 20), (13, 19), (14, 21), (15, 20), (13, 22)],
    "Asian":      [(20, 7), (22, 6), (21, 5), (20, 0), (22, 2)],
    "FullDay":    [(7, 20), (8, 20), (0, 23)],
    "LateNY":     [(20, 23), (20, 22)],
}


def _bucket_of(sess_str):
    s, e = [int(x) for x in sess_str.split("-")]
    for bucket, ranges in SESSION_BUCKETS.items():
        if (s, e) in ranges:
            return bucket
    return "Other"


def _load_top(tag_prefix):
    p = RESULTS / f"{tag_prefix}_top.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-source", default="eur_m15_sweep_mac",
                    help="prefix of sweep_top.json source")
    ap.add_argument("--tag-out", default="eur_24h",
                    help="output prefix")
    ap.add_argument("--data", default="eurusd_m15_2020_2026.parquet")
    ap.add_argument("--min-pf", type=float, default=1.10)
    ap.add_argument("--min-trades", type=int, default=50)
    ap.add_argument("--max-per-session", type=int, default=4,
                    help="cap strategies per session bucket (avoid over-correlation)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[load] sweep top from tag={args.tag_source}")
    top = _load_top(args.tag_source)

    # Pick top configs per (entry_type, session bucket) from sweep
    candidates = []  # list of dicts
    for entry_type, lst in top.items():
        for c in lst:
            try:
                pf = float(c["pf"]); tr = int(c["trades"])
            except Exception:
                continue
            if pf < args.min_pf or tr < args.min_trades:
                continue
            bucket = _bucket_of(c["sess"])
            candidates.append({
                "entry_type": entry_type,
                "sess": c["sess"], "bucket": bucket,
                "sl_atr": float(c["sl_atr"]),
                "tp_atr": float(c["tp_atr"]),
                "vt_vol": float(c["vt_vol"]),
                "hold": int(c["hold"]),
                "expected_pf": pf,
                "expected_trades": tr,
            })

    print(f"  {len(candidates)} candidate cfgs pass PF>={args.min_pf} & tr>={args.min_trades}")

    # Per-bucket: take top max_per_session by expected PF, dedup by entry_type
    by_bucket = defaultdict(list)
    for c in sorted(candidates, key=lambda x: -x["expected_pf"]):
        bucket = c["bucket"]
        # dedup: only one cfg per (entry_type, bucket) — the top one
        if any(x["entry_type"] == c["entry_type"] for x in by_bucket[bucket]):
            continue
        if len(by_bucket[bucket]) >= args.max_per_session:
            continue
        by_bucket[bucket].append(c)

    print("\n[per-session-bucket picks]")
    for bucket, lst in by_bucket.items():
        print(f"  {bucket}: {len(lst)} strategies")
        for c in lst:
            print(f"    {c['entry_type']:18s} sess={c['sess']:>5s} sl={c['sl_atr']} tp={c['tp_atr']} vt={c['vt_vol']} h={c['hold']}  expPF={c['expected_pf']:.3f}")

    print(f"\n[load] {DATA / args.data}")
    df = pd.read_parquet(DATA / args.data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  rows: {len(df):,}")

    print("[build] ctx")
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)

    # Run each candidate, group logs by bucket
    bucket_logs = defaultdict(list)  # bucket -> list of trade DataFrames
    bucket_summaries = defaultdict(list)
    for c in [item for sublist in by_bucket.values() for item in sublist]:
        s, e = [int(x) for x in c["sess"].split("-")]
        cfg = {"entry_type": c["entry_type"], "vt": c["vt_vol"],
               "sess_start": s, "sess_end": e,
               **EXTRA_PARAMS.get(c["entry_type"], {})}
        signals = generate_signals(ctx, cfg, test_indices, vol_ratio)
        log = simulate_signals(ctx, signals, sl_atr=c["sl_atr"], tp_atr=c["tp_atr"],
                               hold_bars=c["hold"])
        if not len(log):
            continue
        name = f"{c['entry_type']}__{c['sess']}_sl{c['sl_atr']}_tp{c['tp_atr']}_vt{c['vt_vol']}_h{c['hold']}"
        log["strategy"] = name
        log["bucket"] = c["bucket"]
        gp = log.loc[log["pnl"] > 0, "pnl"].sum()
        gl = abs(log.loc[log["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0.0
        wr = (log["pnl"] > 0).mean()
        net = log["pnl"].sum()
        print(f"  {c['bucket']:8s} {name[:60]:60s}  trades={len(log):4d}  PF={pf:.3f}  net=${net:+7.0f}")
        bucket_summaries[c["bucket"]].append({
            **c, "actual_trades": int(len(log)), "actual_pf": float(pf),
            "actual_wr": float(wr), "actual_net": float(net),
        })
        bucket_logs[c["bucket"]].append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())

    # Per-session combined portfolios
    print("\n[combine per session]")
    per_session_metrics = {}
    all_strategy_logs = []
    for bucket, logs in bucket_logs.items():
        if not logs:
            continue
        m = combine_strategies(logs, deposit=1000.0,
                               max_concurrent_positions=2,
                               overlap_penalty=0.5)
        print(f"  [{bucket}] {m.summary_line()}")
        # Save bucket-level combined trade log
        cb = pd.concat([l for l in logs], ignore_index=True).sort_values("open_time").reset_index(drop=True)
        cb.to_csv(RESULTS / f"{args.tag_out}__{bucket}__trades.csv", index=False)
        per_session_metrics[bucket] = {
            "pf": m.pf, "n_trades": m.n_trades, "total_pnl": m.total_pnl,
            "max_dd_pct": m.max_dd_pct, "max_dd_usd": m.max_dd_usd,
            "sharpe_daily": m.sharpe_daily, "trades_per_month": m.trades_per_month,
            "win_rate": m.win_rate,
            "n_strategies": len(logs),
            "per_year": m.per_year,
        }
        all_strategy_logs.extend(logs)

    # 24h combined portfolio (everything together)
    print("\n[combine 24h]")
    if all_strategy_logs:
        m24 = combine_strategies(all_strategy_logs, deposit=1000.0,
                                 max_concurrent_positions=3,
                                 overlap_penalty=0.5)
        print(f"  COMBINED 24h: {m24.summary_line()}")
        print(f"  win_rate: {m24.win_rate:.3f}  trades_per_month: {m24.trades_per_month:.1f}")
        print("  per-year:")
        for y, n_t, pf, pn in m24.per_year:
            print(f"    {y}  trades={n_t:5d}  PF={pf:.3f}  pnl=${pn:+.0f}")
        if m24.correlation_matrix is not None and not m24.correlation_matrix.empty:
            print(f"\n  correlation matrix shape={m24.correlation_matrix.shape}; mean off-diag = {m24.correlation_matrix.values[~np.eye(len(m24.correlation_matrix), dtype=bool)].mean():.3f}")
        artifacts = write_artifacts(m24, RESULTS, tag=args.tag_out)
        out_combined = {
            "tag": args.tag_out,
            "per_session": per_session_metrics,
            "combined": {
                "pf": m24.pf, "n_trades": m24.n_trades, "total_pnl": m24.total_pnl,
                "ending_equity": m24.ending_equity, "return_pct": m24.return_pct,
                "max_dd_pct": m24.max_dd_pct, "max_dd_usd": m24.max_dd_usd,
                "recovery_factor": m24.recovery_factor, "sharpe_daily": m24.sharpe_daily,
                "trades_per_month": m24.trades_per_month, "win_rate": m24.win_rate,
                "overlap_hits": m24.overlap_hit_count,
                "per_year": m24.per_year,
            },
            "n_strategies_total": len(all_strategy_logs),
            "bucket_strategy_counts": {b: len(by_bucket[b]) for b in by_bucket},
            "artifacts": artifacts,
        }
        out_path = RESULTS / f"{args.tag_out}_summary.json"
        out_path.write_text(json.dumps(out_combined, indent=2, default=str), encoding="utf-8")
        print(f"\n[saved] {out_path}")
    print(f"\n[elapsed] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
