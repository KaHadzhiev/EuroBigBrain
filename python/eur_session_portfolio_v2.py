#!/usr/bin/env python3
"""EUR M15 24h portfolio v2 — drops breakout_range (DD culprit per v1 postmortem).

Selects top per (strategy, session) excluding any 'breakout_range' configs.
Otherwise identical to eur_session_portfolio.py.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from portfolio.combine import combine_strategies, write_artifacts  # noqa: E402
from eur_portfolio_run import build_ctx, simulate_signals  # noqa: E402
from eur_session_portfolio import EXTRA_PARAMS, _bucket_of, _load_top  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-source", default="eur_m15_sweep_mac")
    ap.add_argument("--tag-out", default="eur_24h_v2")
    ap.add_argument("--data", default="eurusd_m15_2020_2026.parquet")
    ap.add_argument("--min-pf", type=float, default=1.10)
    ap.add_argument("--min-trades", type=int, default=50)
    ap.add_argument("--max-per-session", type=int, default=4)
    ap.add_argument("--exclude", default="breakout_range",
                    help="comma-separated entry_types to exclude")
    args = ap.parse_args()

    excluded = set(args.exclude.split(","))
    print(f"[exclude] {excluded}")

    t0 = time.time()
    top = _load_top(args.tag_source)
    candidates = []
    for entry_type, lst in top.items():
        if entry_type in excluded:
            print(f"  SKIPPING all {entry_type} configs ({len(lst)} qualifiers)")
            continue
        for c in lst:
            try:
                pf = float(c["pf"]); tr = int(c["trades"])
            except Exception:
                continue
            if pf < args.min_pf or tr < args.min_trades:
                continue
            bucket = _bucket_of(c["sess"])
            candidates.append({
                "entry_type": entry_type, "sess": c["sess"], "bucket": bucket,
                "sl_atr": float(c["sl_atr"]), "tp_atr": float(c["tp_atr"]),
                "vt_vol": float(c["vt_vol"]), "hold": int(c["hold"]),
                "expected_pf": pf, "expected_trades": tr,
            })
    print(f"  {len(candidates)} candidate cfgs after exclude+gates")

    by_bucket = defaultdict(list)
    for c in sorted(candidates, key=lambda x: -x["expected_pf"]):
        if any(x["entry_type"] == c["entry_type"] for x in by_bucket[c["bucket"]]):
            continue
        if len(by_bucket[c["bucket"]]) >= args.max_per_session:
            continue
        by_bucket[c["bucket"]].append(c)

    print("\n[picks]")
    for bucket, lst in by_bucket.items():
        print(f"  {bucket}: {len(lst)}")
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

    bucket_logs = defaultdict(list)
    summaries = []
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
        log["strategy"] = name; log["bucket"] = c["bucket"]
        gp = log.loc[log["pnl"] > 0, "pnl"].sum()
        gl = abs(log.loc[log["pnl"] <= 0, "pnl"].sum())
        pf = gp / gl if gl > 0 else 0.0
        wr = (log["pnl"] > 0).mean()
        net = log["pnl"].sum()
        print(f"  {c['bucket']:8s} {name[:60]:60s}  trades={len(log):4d}  PF={pf:.3f}  net=${net:+7.0f}")
        summaries.append({**c, "actual_trades": int(len(log)), "actual_pf": float(pf),
                          "actual_wr": float(wr), "actual_net": float(net)})
        bucket_logs[c["bucket"]].append(log[["open_time", "close_time", "pnl", "strategy", "bucket"]].copy())

    per_session = {}
    all_logs = []
    print("\n[per session]")
    for bucket, logs in bucket_logs.items():
        if not logs: continue
        m = combine_strategies(logs, deposit=1000.0, max_concurrent_positions=2, overlap_penalty=0.5)
        print(f"  [{bucket}] {m.summary_line()}")
        cb = pd.concat(logs, ignore_index=True).sort_values("open_time").reset_index(drop=True)
        cb.to_csv(RESULTS / f"{args.tag_out}__{bucket}__trades.csv", index=False)
        per_session[bucket] = {
            "pf": m.pf, "n_trades": m.n_trades, "total_pnl": m.total_pnl,
            "max_dd_pct": m.max_dd_pct, "max_dd_usd": m.max_dd_usd,
            "sharpe_daily": m.sharpe_daily, "trades_per_month": m.trades_per_month,
            "win_rate": m.win_rate, "n_strategies": len(logs),
            "per_year": m.per_year,
        }
        all_logs.extend(logs)

    print("\n[combine 24h]")
    if all_logs:
        m24 = combine_strategies(all_logs, deposit=1000.0, max_concurrent_positions=3, overlap_penalty=0.5)
        print(f"  COMBINED: {m24.summary_line()}")
        print(f"  WR={m24.win_rate:.3f}  trades/mo={m24.trades_per_month:.1f}")
        print("  per-year:")
        for y, n_t, pf, pn in m24.per_year:
            print(f"    {y}  trades={n_t:5d}  PF={pf:.3f}  pnl=${pn:+.0f}")
        if m24.correlation_matrix is not None and not m24.correlation_matrix.empty:
            mat = m24.correlation_matrix.values
            mean_off = mat[~np.eye(len(mat), dtype=bool)].mean()
            print(f"\n  mean off-diag corr = {mean_off:.3f}")
        artifacts = write_artifacts(m24, RESULTS, tag=args.tag_out)
        out = {
            "tag": args.tag_out, "excluded": list(excluded),
            "per_session": per_session,
            "combined": {
                "pf": m24.pf, "n_trades": m24.n_trades, "total_pnl": m24.total_pnl,
                "ending_equity": m24.ending_equity, "return_pct": m24.return_pct,
                "max_dd_pct": m24.max_dd_pct, "max_dd_usd": m24.max_dd_usd,
                "recovery_factor": m24.recovery_factor, "sharpe_daily": m24.sharpe_daily,
                "trades_per_month": m24.trades_per_month, "win_rate": m24.win_rate,
                "overlap_hits": m24.overlap_hit_count, "per_year": m24.per_year,
            },
            "n_strategies_total": len(all_logs),
            "strategies": summaries,
            "artifacts": artifacts,
        }
        op = RESULTS / f"{args.tag_out}_summary.json"
        op.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\n[saved] {op}")
    print(f"\n[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
