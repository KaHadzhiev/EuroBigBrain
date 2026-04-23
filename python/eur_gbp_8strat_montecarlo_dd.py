#!/usr/bin/env python3
"""Monte Carlo DD on 8-strat champion: 1000 random orderings of the same trades.

Tells us the p50/p95/p99 worst-case DD given the same trade pool — important
pre-deploy: live trading might experience worse DD than the historical sequence.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "portfolio"


def main():
    # Load the 8-strat trade log (sum of all 8 strategy trade CSVs)
    files = sorted(RESULTS.glob("eur_gbp_cross_asset_v1__*__trades.csv"))
    print(f"loading {len(files)} per-strat CSVs")
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  total trades: {len(df):,}")
    pnls = df['pnl'].to_numpy()
    print(f"  total PnL: ${pnls.sum():+.0f}")

    rng = np.random.default_rng(42)
    n_trials = 1000
    dd_pcts = np.zeros(n_trials)
    dd_dollars = np.zeros(n_trials)
    final_eqs = np.zeros(n_trials)
    deposit = 1000.0

    print(f"\nRunning {n_trials} Monte Carlo trials...")
    for i in range(n_trials):
        order = rng.permutation(len(pnls))
        eq = deposit + np.cumsum(pnls[order])
        peaks = np.maximum.accumulate(eq)
        dd_d = (peaks - eq).max()
        dd_p = ((peaks - eq) / np.maximum(peaks, 1e-9)).max() * 100
        dd_pcts[i] = dd_p
        dd_dollars[i] = dd_d
        final_eqs[i] = eq[-1]
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_trials}")

    p50 = np.percentile(dd_pcts, 50)
    p95 = np.percentile(dd_pcts, 95)
    p99 = np.percentile(dd_pcts, 99)
    p_max = dd_pcts.max()
    print(f"\nDD% distribution (over {n_trials} random orderings):")
    print(f"  p50  (typical):  {p50:.2f}%")
    print(f"  p95  (bad tail): {p95:.2f}%")
    print(f"  p99  (very bad): {p99:.2f}%")
    print(f"  worst seen:      {p_max:.2f}%")
    print(f"\nHistorical actual DD: 11.4% (the order in the data; some MC orderings worse, some better)")
    print(f"\nFinal equity distribution:")
    print(f"  mean ending equity: ${final_eqs.mean():.0f}")
    print(f"  std:                ${final_eqs.std():.0f}")

    out = {
        "n_trials": n_trials, "n_trades": int(len(pnls)),
        "total_pnl": float(pnls.sum()),
        "dd_pct_p50": float(p50), "dd_pct_p95": float(p95),
        "dd_pct_p99": float(p99), "dd_pct_worst": float(p_max),
        "ending_equity_mean": float(final_eqs.mean()),
        "ending_equity_std": float(final_eqs.std()),
        "all_dd_pcts": dd_pcts.tolist()[:200],  # truncate to keep JSON small
    }
    op = RESULTS / "eur_gbp_8strat_montecarlo_dd.json"
    op.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
