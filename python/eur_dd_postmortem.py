#!/usr/bin/env python3
"""EUR portfolio drawdown postmortem.

Per user instruction (2026-04-23): after the portfolio is built, find the
worst DD periods, identify root causes, propose fixes.

Inputs:
  results/portfolio/eur_portfolio_v2_summary.json
  results/portfolio/eur_portfolio_v2__<strategy>__trades.csv  (per strat)

Outputs:
  results/portfolio/eur_dd_postmortem.json
  results/portfolio/eur_dd_postmortem.md  (human-readable writeup)

Steps:
  1. Reconstruct combined equity curve
  2. Find top-K drawdown events (peak -> trough)
  3. For each DD: list trades, by strategy, by hour, by month, by exit-reason
  4. Cluster patterns (regime / time / strategy concentration / direction)
  5. Propose mitigations and quantify (e.g. "remove Friday 17h trades -> +3% PF")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "portfolio"


def _load_combined(tag: str) -> pd.DataFrame:
    """Load + concatenate all per-strategy trade CSVs for the tag."""
    pattern = f"{tag}__*__trades.csv"
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no per-strategy trades found for tag={tag}")
    dfs = []
    for p in files:
        d = pd.read_csv(p)
        d["open_time"] = pd.to_datetime(d["open_time"])
        d["close_time"] = pd.to_datetime(d["close_time"])
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).sort_values("open_time").reset_index(drop=True)
    return df


def _equity_curve(df: pd.DataFrame, deposit=1000.0):
    eq = deposit + df["pnl"].cumsum().to_numpy()
    peaks = np.maximum.accumulate(eq)
    dd = peaks - eq
    dd_pct = (peaks - eq) / np.maximum(peaks, 1e-9)
    return eq, peaks, dd, dd_pct


def _find_dd_events(df, eq, peaks, dd_pct, top_k=5, min_pct=0.03):
    """Find the top-K disjoint drawdown events (peak -> trough)."""
    n = len(eq)
    events = []
    used = np.zeros(n, dtype=bool)
    order = np.argsort(-dd_pct)  # deepest first
    for trough in order:
        if used[trough]:
            continue
        if dd_pct[trough] < min_pct:
            continue
        # find peak before trough: walk back from trough-1 while eq < peak_val
        peak_val = peaks[trough]
        peak_idx = trough - 1
        while peak_idx > 0 and eq[peak_idx] < peak_val - 1e-9:
            peak_idx -= 1
        # find recovery: first index after trough where eq returns to peak_val
        recov_idx = None
        for j in range(trough + 1, n):
            if eq[j] >= peak_val - 1e-9:
                recov_idx = j
                break
        # mark used range
        end = recov_idx if recov_idx is not None else n - 1
        used[peak_idx:end + 1] = True
        events.append({
            "peak_idx": int(peak_idx),
            "trough_idx": int(trough),
            "recov_idx": int(recov_idx) if recov_idx is not None else None,
            "peak_eq": float(eq[peak_idx]),
            "trough_eq": float(eq[trough]),
            "dd_usd": float(eq[peak_idx] - eq[trough]),
            "dd_pct": float(dd_pct[trough] * 100),
            "peak_time": str(df["open_time"].iloc[peak_idx]),
            "trough_time": str(df["open_time"].iloc[trough]),
            "trough_close_time": str(df["close_time"].iloc[trough]),
            "recov_time": (str(df["open_time"].iloc[recov_idx]) if recov_idx is not None else None),
            "duration_trades": int(trough - peak_idx),
        })
        if len(events) >= top_k:
            break
    return events


def _analyse_dd_window(df: pd.DataFrame, evt: dict) -> dict:
    """For one DD event: collect stats on what was happening."""
    sl = slice(evt["peak_idx"], evt["trough_idx"] + 1)
    sub = df.iloc[sl]
    if not len(sub):
        return {}
    by_strat = sub.groupby("strategy")["pnl"].agg(["count", "sum", "mean"]).to_dict()
    by_hour = sub.assign(hour=sub["open_time"].dt.hour).groupby("hour")["pnl"].sum().to_dict()
    by_dow = sub.assign(dow=sub["open_time"].dt.day_name()).groupby("dow")["pnl"].sum().to_dict()
    by_month = sub.assign(month=sub["open_time"].dt.to_period("M").astype(str)).groupby("month")["pnl"].sum().to_dict()
    by_dir = sub.groupby("direction")["pnl"].sum().to_dict() if "direction" in sub.columns else {}
    by_reason = sub.groupby("exit_reason")["pnl"].agg(["count", "sum"]).to_dict() if "exit_reason" in sub.columns else {}
    return {
        "n_trades": int(len(sub)),
        "wins": int((sub["pnl"] > 0).sum()),
        "losses": int((sub["pnl"] <= 0).sum()),
        "by_strategy": by_strat,
        "by_hour": {int(k): float(v) for k, v in by_hour.items()},
        "by_day_of_week": {str(k): float(v) for k, v in by_dow.items()},
        "by_month": {str(k): float(v) for k, v in by_month.items()},
        "by_direction": {str(k): float(v) for k, v in by_dir.items()},
        "by_exit_reason": {str(k): v for k, v in by_reason.items()} if by_reason else {},
    }


def _aggregate_patterns(events_with_analysis):
    """Across all DDs, look for repeated culprits (strategy / hour / month / dir)."""
    counter_strat: dict = {}
    counter_hour: dict = {}
    counter_dow: dict = {}
    for evt in events_with_analysis:
        a = evt.get("analysis", {})
        for s, sumv in a.get("by_strategy", {}).get("sum", {}).items():
            counter_strat[s] = counter_strat.get(s, 0.0) + sumv
        for h, v in a.get("by_hour", {}).items():
            counter_hour[h] = counter_hour.get(h, 0.0) + v
        for d, v in a.get("by_day_of_week", {}).items():
            counter_dow[d] = counter_dow.get(d, 0.0) + v
    return {
        "loss_by_strategy": dict(sorted(counter_strat.items(), key=lambda x: x[1])),
        "loss_by_hour":      dict(sorted(counter_hour.items(), key=lambda x: x[1])),
        "loss_by_day_of_week": dict(sorted(counter_dow.items(), key=lambda x: x[1])),
    }


def _proposed_fixes(patterns: dict) -> list:
    """Heuristic suggestions based on the pattern aggregation."""
    fixes = []
    # Hour blacklist: if 1-2 hours dominate losses, suggest removing
    hours_sorted = sorted(patterns["loss_by_hour"].items(), key=lambda x: x[1])
    if hours_sorted:
        worst_hour, worst_loss = hours_sorted[0]
        if worst_loss < -50:  # arbitrary; meaningful loss
            fixes.append(f"Hour blacklist: drop signals at hour={worst_hour} (DD-loss=${worst_loss:.0f}). Caveat: re-test, do NOT retro-filter winners.")
    # Strategy concentration: if one strategy contributes >50% of DD loss
    strat_sorted = sorted(patterns["loss_by_strategy"].items(), key=lambda x: x[1])
    if strat_sorted:
        worst_strat, worst_pl = strat_sorted[0]
        total_loss = sum(v for v in patterns["loss_by_strategy"].values() if v < 0)
        if total_loss < 0 and worst_pl / total_loss > 0.5:
            fixes.append(f"Strategy concentration: '{worst_strat}' = {100*worst_pl/total_loss:.0f}% of DD loss. Consider half-weight.")
    dow_sorted = sorted(patterns["loss_by_day_of_week"].items(), key=lambda x: x[1])
    if dow_sorted:
        worst_dow, worst_pl_dow = dow_sorted[0]
        if worst_pl_dow < -30:
            fixes.append(f"Day-of-week filter: avoid {worst_dow} (DD-loss=${worst_pl_dow:.0f}).")
    if not fixes:
        fixes.append("No single dominant culprit -- DD is broadly distributed. Recommend: lower per-trade risk, add daily-loss cap.")
    return fixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="eur_portfolio_v2")
    ap.add_argument("--top-dd", type=int, default=5)
    ap.add_argument("--min-dd-pct", type=float, default=0.03,
                    help="ignore DDs smaller than this (3%% default)")
    args = ap.parse_args()

    print(f"[load] tag={args.tag}")
    df = _load_combined(args.tag)
    print(f"  trades: {len(df):,}  span={df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}")

    eq, peaks, dd, dd_pct = _equity_curve(df)
    print(f"[equity] start=$1000  end=${eq[-1]:.0f}  net=${eq[-1]-1000:+.0f}")
    print(f"[dd] max=${dd.max():.0f}  ({dd_pct.max()*100:.1f}%)")

    events = _find_dd_events(df, eq, peaks, dd_pct,
                             top_k=args.top_dd, min_pct=args.min_dd_pct)
    print(f"[events] {len(events)} DD events >= {args.min_dd_pct*100:.0f}%")

    enriched = []
    for i, e in enumerate(events):
        a = _analyse_dd_window(df, e)
        e["analysis"] = a
        enriched.append(e)
        print(f"  DD#{i+1}: {e['dd_pct']:.1f}% (${e['dd_usd']:.0f}) "
              f"{e['peak_time'][:10]} -> {e['trough_time'][:10]}  "
              f"({e['duration_trades']} trades)")

    patterns = _aggregate_patterns(enriched)
    fixes = _proposed_fixes(patterns)

    out = {
        "tag": args.tag,
        "summary": {
            "n_trades": int(len(df)),
            "ending_equity": float(eq[-1]),
            "max_dd_pct": float(dd_pct.max() * 100),
            "max_dd_usd": float(dd.max()),
        },
        "events": enriched,
        "patterns": patterns,
        "proposed_fixes": fixes,
    }
    out_json = RESULTS / f"{args.tag}__dd_postmortem.json"
    out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    # Markdown writeup
    md_lines = [f"# {args.tag} -- DD Postmortem\n"]
    md_lines.append(f"- trades: {len(df):,}")
    md_lines.append(f"- ending_eq: ${eq[-1]:.0f}  (start $1000)")
    md_lines.append(f"- max_dd: {dd_pct.max()*100:.2f}% (${dd.max():.0f})\n")
    md_lines.append("## Top DD events\n")
    for i, e in enumerate(enriched, 1):
        md_lines.append(f"### DD#{i}: {e['dd_pct']:.1f}% (${e['dd_usd']:.0f})")
        md_lines.append(f"  - peak: {e['peak_time']}  trough: {e['trough_time']}")
        md_lines.append(f"  - recov: {e.get('recov_time') or 'NEVER (still in DD)'}")
        md_lines.append(f"  - duration: {e['duration_trades']} trades")
        a = e.get("analysis", {})
        if a:
            top_strats = sorted(a.get("by_strategy", {}).get("sum", {}).items(), key=lambda x: x[1])[:3]
            md_lines.append(f"  - worst-strats: {top_strats}")
            top_hours = sorted(a.get("by_hour", {}).items(), key=lambda x: x[1])[:3]
            md_lines.append(f"  - worst-hours: {top_hours}\n")
    md_lines.append("## Aggregate patterns\n")
    md_lines.append("### Loss by strategy (sum across all DDs)\n")
    for k, v in patterns["loss_by_strategy"].items():
        md_lines.append(f"  - {k}: ${v:.0f}")
    md_lines.append("\n### Loss by hour\n")
    for k, v in patterns["loss_by_hour"].items():
        md_lines.append(f"  - h={k}: ${v:.0f}")
    md_lines.append("\n### Loss by day of week\n")
    for k, v in patterns["loss_by_day_of_week"].items():
        md_lines.append(f"  - {k}: ${v:.0f}")
    md_lines.append("\n## Proposed mitigations\n")
    for f in fixes:
        md_lines.append(f"  - {f}")
    out_md = RESULTS / f"{args.tag}__dd_postmortem.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
