#!/usr/bin/env python3
"""Deep DD postmortem on EUR 24h portfolio — patterns + root causes.

Goes beyond eur_dd_postmortem.py. For each top-K DD event, breaks down by:
  - Hour of day (which session leaked)
  - Day of week
  - Year + month + quarter
  - Strategy concentration (which strategies drove it)
  - Direction concentration (long vs short)
  - Exit-reason mix (SL / TP / timeout)
  - Regime context (vol regime, trend regime, ATR percentile)
  - Correlation spike (did normally-uncorrelated strategies all lose together?)

Aggregate "patterns across ALL DDs" then proposes specific mitigations.

Outputs:
  results/portfolio/<tag>__dd_deep.json
  results/portfolio/<tag>__dd_deep.md  (human-readable writeup)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"


def _load_combined(tag):
    """Load + concat all per-strategy/per-bucket trade CSVs for tag."""
    files = sorted(RESULTS.glob(f"{tag}__*__trades.csv"))
    if not files:
        raise FileNotFoundError(f"no trade CSVs for tag={tag}")
    dfs = []
    for p in files:
        d = pd.read_csv(p)
        d["open_time"] = pd.to_datetime(d["open_time"])
        d["close_time"] = pd.to_datetime(d["close_time"])
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True).sort_values("open_time").reset_index(drop=True)


def _equity(df, dep=1000.0):
    eq = dep + df["pnl"].cumsum().to_numpy()
    peaks = np.maximum.accumulate(eq)
    dd_pct = (peaks - eq) / np.maximum(peaks, 1e-9)
    return eq, peaks, dd_pct


def _find_dd_events(df, eq, peaks, dd_pct, top_k=10, min_pct=0.03):
    n = len(eq)
    events = []
    used = np.zeros(n, dtype=bool)
    order = np.argsort(-dd_pct)
    for trough in order:
        if used[trough] or dd_pct[trough] < min_pct:
            continue
        peak_val = peaks[trough]
        peak_idx = trough - 1
        while peak_idx > 0 and eq[peak_idx] < peak_val - 1e-9:
            peak_idx -= 1
        recov_idx = None
        for j in range(trough + 1, n):
            if eq[j] >= peak_val - 1e-9:
                recov_idx = j
                break
        end = recov_idx if recov_idx is not None else n - 1
        used[peak_idx:end + 1] = True
        events.append({
            "rank": len(events) + 1,
            "peak_idx": int(peak_idx), "trough_idx": int(trough),
            "recov_idx": int(recov_idx) if recov_idx is not None else None,
            "peak_eq": float(eq[peak_idx]),
            "trough_eq": float(eq[trough]),
            "dd_usd": float(eq[peak_idx] - eq[trough]),
            "dd_pct": float(dd_pct[trough] * 100),
            "peak_time": str(df["open_time"].iloc[peak_idx]),
            "trough_time": str(df["close_time"].iloc[trough]),
            "recov_time": (str(df["open_time"].iloc[recov_idx]) if recov_idx is not None else None),
            "duration_trades": int(trough - peak_idx),
            "duration_days": int((df["close_time"].iloc[trough] - df["open_time"].iloc[peak_idx]).days),
        })
        if len(events) >= top_k:
            break
    return events


def _vol_regime(df_m15):
    """Bin daily ATR into 4 quartiles -> low / mid / high / extreme."""
    h = df_m15['high']; l = df_m15['low']; c = df_m15['close']
    pc = np.r_[c.iloc[0], c.iloc[:-1].values]
    tr = np.maximum.reduce([h-l, np.abs(h-pc), np.abs(l-pc)])
    df_m15 = df_m15.assign(atr=pd.Series(tr, index=df_m15.index).rolling(96).mean())
    daily = df_m15['atr'].resample('1D').mean().dropna()
    quartiles = pd.qcut(daily, 4, labels=['low', 'mid', 'high', 'extreme'])
    return quartiles


def _analyse_event(df, evt, vol_regime=None):
    sub = df.iloc[evt["peak_idx"]:evt["trough_idx"] + 1]
    if not len(sub):
        return {}
    sub = sub.copy()
    sub["hour"] = sub["open_time"].dt.hour
    sub["dow"] = sub["open_time"].dt.day_name()
    sub["month"] = sub["open_time"].dt.to_period("M").astype(str)
    sub["year"] = sub["open_time"].dt.year
    sub["date"] = sub["open_time"].dt.date

    a = {
        "n_trades": int(len(sub)),
        "wins": int((sub["pnl"] > 0).sum()),
        "losses": int((sub["pnl"] <= 0).sum()),
        "win_rate": float((sub["pnl"] > 0).mean()),
        "by_strategy": sub.groupby("strategy")["pnl"].agg(["count", "sum", "mean"]).to_dict("index"),
        "by_bucket": sub.groupby("bucket")["pnl"].agg(["count", "sum"]).to_dict("index") if "bucket" in sub.columns else {},
        "by_hour": {int(k): float(v) for k, v in sub.groupby("hour")["pnl"].sum().to_dict().items()},
        "by_dow": {str(k): float(v) for k, v in sub.groupby("dow")["pnl"].sum().to_dict().items()},
        "by_month": {str(k): float(v) for k, v in sub.groupby("month")["pnl"].sum().to_dict().items()},
        "by_year": {int(k): float(v) for k, v in sub.groupby("year")["pnl"].sum().to_dict().items()},
        "by_direction": ({str(k): float(v) for k, v in sub.groupby("direction")["pnl"].sum().to_dict().items()}
                          if "direction" in sub.columns else {}),
        "by_exit_reason": ({str(k): {"count": int(g.shape[0]), "sum": float(g["pnl"].sum())}
                             for k, g in sub.groupby("exit_reason")}
                            if "exit_reason" in sub.columns else {}),
    }
    # Strategy concentration: % of DD-loss attributable to top-1 strategy
    losses_only = sub[sub["pnl"] < 0]
    if len(losses_only):
        per_strat = losses_only.groupby("strategy")["pnl"].sum().sort_values()
        worst_strat = per_strat.index[0]
        worst_sum = float(per_strat.iloc[0])
        total_loss = float(losses_only["pnl"].sum())
        a["worst_strategy"] = worst_strat
        a["worst_strategy_loss_share"] = float(worst_sum / total_loss) if total_loss != 0 else 0.0
    # Vol regime during DD
    if vol_regime is not None:
        regimes_in_dd = vol_regime.reindex(pd.to_datetime(sub["date"]).unique()).dropna()
        if len(regimes_in_dd):
            cnt = Counter(str(x) for x in regimes_in_dd.values)
            a["vol_regime_share"] = {k: v / sum(cnt.values()) for k, v in cnt.items()}
    return a


def _aggregate_patterns(events):
    """Across all DDs, look for repeated culprits."""
    counter_strat = defaultdict(float); counter_strat_count = Counter()
    counter_hour = defaultdict(float); counter_hour_count = Counter()
    counter_dow = defaultdict(float)
    counter_month = defaultdict(float)
    counter_year = defaultdict(float)
    counter_dir = defaultdict(float)
    counter_reason = defaultdict(float)
    counter_bucket = defaultdict(float)
    worst_strats = Counter()
    for e in events:
        a = e.get("analysis", {})
        worst_strats[a.get("worst_strategy", "?")] += 1
        for s, info in a.get("by_strategy", {}).items():
            counter_strat[s] += info.get("sum", 0.0)
            counter_strat_count[s] += info.get("count", 0)
        for h, v in a.get("by_hour", {}).items():
            counter_hour[h] += v
            counter_hour_count[h] += 1
        for d, v in a.get("by_dow", {}).items():
            counter_dow[d] += v
        for m, v in a.get("by_month", {}).items():
            counter_month[m] += v
        for y, v in a.get("by_year", {}).items():
            counter_year[y] += v
        for d, v in a.get("by_direction", {}).items():
            counter_dir[d] += v
        for r, info in a.get("by_exit_reason", {}).items():
            counter_reason[r] += info.get("sum", 0.0)
        for b, info in a.get("by_bucket", {}).items():
            counter_bucket[b] += info.get("sum", 0.0)
    return {
        "loss_by_strategy": dict(sorted(counter_strat.items(), key=lambda x: x[1])),
        "trades_per_strategy_in_dd": dict(counter_strat_count),
        "loss_by_hour": dict(sorted(counter_hour.items(), key=lambda x: x[1])),
        "loss_by_dow": dict(sorted(counter_dow.items(), key=lambda x: x[1])),
        "loss_by_month_top10_worst": dict(sorted(counter_month.items(), key=lambda x: x[1])[:10]),
        "loss_by_year": dict(sorted(counter_year.items(), key=lambda x: x[1])),
        "loss_by_direction": dict(counter_dir),
        "loss_by_exit_reason": dict(counter_reason),
        "loss_by_bucket": dict(sorted(counter_bucket.items(), key=lambda x: x[1])),
        "most_often_worst_strategy": dict(worst_strats.most_common()),
    }


def _propose_fixes(patterns, n_dds):
    fixes = []
    # Bucket — if one bucket dominates losses
    bucket_losses = patterns["loss_by_bucket"]
    if bucket_losses:
        worst_b, worst_b_v = next(iter(bucket_losses.items()))
        total_b = sum(v for v in bucket_losses.values() if v < 0)
        if total_b < 0 and worst_b_v / total_b > 0.5:
            fixes.append(f"Session bucket '{worst_b}' = {100*worst_b_v/total_b:.0f}% of DD loss across {n_dds} events. Consider reducing weight or pausing this bucket.")
    # Hour
    hours = patterns["loss_by_hour"]
    worst_h = list(hours.keys())[:2] if hours else []
    if worst_h:
        worst_2 = sum(hours[h] for h in worst_h)
        total_h = sum(v for v in hours.values() if v < 0)
        if total_h < 0 and worst_2 / total_h > 0.4:
            fixes.append(f"Hours {worst_h} = {100*worst_2/total_h:.0f}% of DD loss. News-window blackout candidates.")
    # Direction
    dirs = patterns["loss_by_direction"]
    if dirs and len(dirs) == 2:
        long_l = dirs.get("long", 0.0)
        short_l = dirs.get("short", 0.0)
        if long_l < 0 and short_l < 0:
            ratio = abs(long_l) / max(abs(long_l) + abs(short_l), 1)
            if ratio > 0.7:
                fixes.append(f"Long-side losses dominate ({100*ratio:.0f}% of DD loss). Half-weight long strategies?")
            elif ratio < 0.3:
                fixes.append(f"Short-side losses dominate ({100*(1-ratio):.0f}%). Half-weight short strategies?")
    # Exit reason
    er = patterns["loss_by_exit_reason"]
    if er:
        if "sl" in er and abs(er["sl"]) > sum(abs(v) for v in er.values() if v < 0) * 0.7:
            fixes.append(f"SL exits = {100*abs(er['sl'])/sum(abs(v) for v in er.values() if v<0):.0f}% of DD loss. Tighter trailing stop or wider SL with smaller position?")
    # Year
    years = patterns["loss_by_year"]
    if years:
        worst_y, worst_yv = next(iter(years.items()))
        if worst_yv < -100:  # arbitrary
            fixes.append(f"Year {worst_y}: net DD-loss=${worst_yv:.0f}. Investigate regime — Fed pivot/EM crisis/COVID etc.")
    # Most-often worst strategy
    most_worst = patterns["most_often_worst_strategy"]
    if most_worst:
        s, n = next(iter(most_worst.items()))
        if n >= 3:
            fixes.append(f"Strategy '{s}' was 'worst contributor' in {n}/{n_dds} DD events. Consider drop-test.")
    if not fixes:
        fixes.append("DDs broadly distributed — no single dominant culprit. Lower per-trade risk, add daily-loss cap.")
    return fixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="eur_24h")
    ap.add_argument("--top-dd", type=int, default=10)
    ap.add_argument("--min-dd-pct", type=float, default=0.03)
    ap.add_argument("--data", default="eurusd_m15_2020_2026.parquet")
    args = ap.parse_args()

    print(f"[load] tag={args.tag}")
    df = _load_combined(args.tag)
    print(f"  trades: {len(df):,}  span={df['open_time'].iloc[0]} -> {df['open_time'].iloc[-1]}")

    eq, peaks, dd_pct = _equity(df)
    print(f"[equity] start=$1000  end=${eq[-1]:.0f}  net=${eq[-1]-1000:+.0f}  maxDD={dd_pct.max()*100:.2f}%")

    df_m15 = pd.read_parquet(DATA / args.data)
    df_m15["time"] = pd.to_datetime(df_m15["time"])
    df_m15 = df_m15.set_index("time").sort_index()
    vol_reg = _vol_regime(df_m15)

    events = _find_dd_events(df, eq, peaks, dd_pct, top_k=args.top_dd, min_pct=args.min_dd_pct)
    print(f"[events] {len(events)} DD events >= {args.min_dd_pct*100:.0f}%")
    enriched = []
    for e in events:
        e["analysis"] = _analyse_event(df, e, vol_regime=vol_reg)
        enriched.append(e)
        print(f"  DD#{e['rank']}: {e['dd_pct']:.1f}% (${e['dd_usd']:.0f})  "
              f"{e['peak_time'][:10]} -> {e['trough_time'][:10]}  "
              f"({e['duration_trades']} trades, {e['duration_days']}d)  "
              f"WR={e['analysis']['win_rate']:.2f}  worst={e['analysis'].get('worst_strategy', '?')[:35]}")

    patterns = _aggregate_patterns(enriched)
    fixes = _propose_fixes(patterns, n_dds=len(enriched))

    out = {
        "tag": args.tag,
        "summary": {
            "n_trades": int(len(df)),
            "ending_equity": float(eq[-1]),
            "max_dd_pct": float(dd_pct.max() * 100),
            "max_dd_usd": float((peaks - eq).max()),
        },
        "events": enriched,
        "patterns": patterns,
        "proposed_fixes": fixes,
    }
    out_json = RESULTS / f"{args.tag}__dd_deep.json"
    out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    # MD writeup
    md = [f"# {args.tag} -- Deep DD Postmortem\n"]
    md.append(f"- trades: {len(df):,}")
    md.append(f"- ending eq: ${eq[-1]:.0f}  (start $1000)")
    md.append(f"- max DD: {dd_pct.max()*100:.2f}% (${(peaks-eq).max():.0f})\n")
    md.append("## Top DD events (rank by depth)\n")
    for e in enriched:
        md.append(f"### DD#{e['rank']}: {e['dd_pct']:.1f}% (${e['dd_usd']:.0f})  duration={e['duration_days']}d / {e['duration_trades']} trades")
        md.append(f"- peak: {e['peak_time']}  trough: {e['trough_time']}  recov: {e.get('recov_time') or 'NEVER'}")
        a = e["analysis"]
        if a:
            md.append(f"- WR during DD: {a['win_rate']:.2f}  ({a['wins']}W / {a['losses']}L)")
            md.append(f"- worst strategy: `{a.get('worst_strategy', '?')}` ({100*a.get('worst_strategy_loss_share',0):.0f}% of loss)")
            top_h = sorted(a.get("by_hour", {}).items(), key=lambda x: x[1])[:3]
            md.append(f"- worst hours: {top_h}")
            if a.get("vol_regime_share"):
                md.append(f"- vol regime: {a['vol_regime_share']}")
            if a.get("by_direction"):
                md.append(f"- by direction: {a['by_direction']}")
        md.append("")

    md.append("## Aggregate patterns across all DDs\n")
    md.append("### Loss by session bucket (sum)\n")
    for k, v in patterns["loss_by_bucket"].items():
        md.append(f"  - {k}: ${v:.0f}")
    md.append("\n### Loss by hour (worst first)\n")
    for k, v in patterns["loss_by_hour"].items():
        md.append(f"  - h={k}: ${v:.0f}")
    md.append("\n### Loss by year\n")
    for k, v in patterns["loss_by_year"].items():
        md.append(f"  - {k}: ${v:.0f}")
    md.append("\n### Loss by exit reason\n")
    for k, v in patterns["loss_by_exit_reason"].items():
        md.append(f"  - {k}: ${v:.0f}")
    md.append("\n### Loss by direction\n")
    for k, v in patterns["loss_by_direction"].items():
        md.append(f"  - {k}: ${v:.0f}")
    md.append("\n### 'Worst contributor' frequency across DDs\n")
    for k, v in patterns["most_often_worst_strategy"].items():
        md.append(f"  - {k}: {v} times")
    md.append("\n## Proposed mitigations\n")
    for f in fixes:
        md.append(f"  - {f}")
    out_md = RESULTS / f"{args.tag}__dd_deep.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
