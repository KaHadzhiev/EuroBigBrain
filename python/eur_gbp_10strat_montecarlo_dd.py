#!/usr/bin/env python3
"""Monte Carlo DD on 10-strat — 1000 random orderings."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "portfolio"


def main():
    files = sorted(RESULTS.glob("eur_gbp_10strat_v1__*__trades.csv"))
    print(f"loading {len(files)} per-strat CSVs")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  total trades: {len(df):,}")
    pnls = df['pnl'].to_numpy()
    print(f"  total PnL: ${pnls.sum():+.0f}")

    rng = np.random.default_rng(42)
    n_trials = 1000
    dd_pcts = np.zeros(n_trials)
    final_eqs = np.zeros(n_trials)
    deposit = 1000.0

    print(f"\nRunning {n_trials} MC trials...")
    for i in range(n_trials):
        order = rng.permutation(len(pnls))
        eq = deposit + np.cumsum(pnls[order])
        peaks = np.maximum.accumulate(eq)
        dd_pcts[i] = ((peaks - eq) / np.maximum(peaks, 1e-9)).max() * 100
        final_eqs[i] = eq[-1]

    p50 = np.percentile(dd_pcts, 50)
    p95 = np.percentile(dd_pcts, 95)
    p99 = np.percentile(dd_pcts, 99)
    p_max = dd_pcts.max()
    print(f"\nDD% distribution (1000 trials):")
    print(f"  p50:    {p50:.2f}%")
    print(f"  p95:    {p95:.2f}%")
    print(f"  p99:    {p99:.2f}%")
    print(f"  worst:  {p_max:.2f}%")
    print(f"\nHistorical DD: 9.2% (10-strat in actual order)")
    print(f"\nFinal equity: mean=${final_eqs.mean():.0f}  std=${final_eqs.std():.0f}")

    out = {"n_trials": n_trials, "n_trades": int(len(pnls)),
           "total_pnl": float(pnls.sum()),
           "dd_pct_p50": float(p50), "dd_pct_p95": float(p95),
           "dd_pct_p99": float(p99), "dd_pct_worst": float(p_max),
           "ending_equity_mean": float(final_eqs.mean())}
    op = RESULTS / "eur_gbp_10strat_montecarlo_dd.json"
    op.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
