# Deflated Sharpe Ratio — Design Note

**Module:** `python/validation/deflated_sharpe.py`
**Pipeline role:** Gate-2 selection-bias filter (WG3 validation pipeline)
**Reference:** Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"

## What problem does this solve?

A backtest sweep of N configurations is a multiple-hypothesis test. The best-of-N observed Sharpe will always look impressive even when every candidate is pure noise — this is why GoldBigBrain's cfg74 (PF=1.41, from a ~1500-config sweep) read as a winner yet later failed an every-tick null test.

The Deflated Sharpe Ratio (DSR) corrects for this by asking: *given N trials were conducted, is the observed SR high enough to reject the hypothesis that it's merely the top-of-sample noise?* It deflates the raw SR by the expected maximum SR under the null, scaled by the sampling variance of SR.

## Mathematical core

The module implements three quantities, each in annualised SR units:

1. **Observed SR**: from a return series, or derived from (PF, win rate, trades/year) via `pf_to_sharpe`.
2. **Expected max SR under N IID trials** (Bailey closed form):
   `E[max] = sqrt(V) · [(1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))]`, γ = Euler–Mascheroni (0.5772), V = variance of SRs across the N trials (default 1.0, Bailey's conservative choice).
3. **Asymptotic std of SR** (Lo 2002, Mertens 2002), with the non-normality correction Bailey/LdP prescribe: `σ(SR) = sqrt((1 − γ3·SR + (γ4 − 1)/4 · SR²) / (T − 1))`, scaled by `sqrt(annualisation)`.

DSR is then `Φ((SR_observed − E[max]) / σ(SR))`. A strategy passes Gate 2 iff DSR > 0.95 (p < 0.05). The `min_pf_for_dsr_pass` helper binary-searches the minimum PF that clears the threshold given a sweep size — used for pre-registering p-hacking budgets.

## Where DSR runs in the pipeline

DSR is the **second gate**, immediately after in-sample sanity. Concretely:

1. **Gate 1** (in-sample sanity) — cheap filter: PF ≥ 1.30, trades ≥ 500, DD ≤ 15%, duration reasonable. Fail here = abort.
2. **Gate 2a** (block bootstrap) — stationary block bootstrap p-value on per-trade returns, null hypothesis μ = 0.
3. **Gate 2b (this module)** — Deflated Sharpe on the same candidate, where `N` is the *cumulative* trial count from `validation_ledger.parquet`, **not** just the current sweep. This enforces the project-wide p-hacking budget (WG3: 200 lifetime backtests).
4. **Gate 3+** — archetype-specific null, walk-forward, per-year stability, cost stress, robustness, multi-instrument, regime, held-out, live monitor.

DSR runs **before** any deployment discussion and **before** any MT5 every-tick validation. If a candidate fails DSR, it is logged to the ledger (incrementing the trial counter) and discarded. It may not be revived with parameter tweaks — that's how p-hacking comes in through the back door.

## What NOT to do with DSR

- **Do not** run DSR on a single favorite config and call it validation; N must include every sibling config tried during the candidate selection, including exploratory ones that were never written up.
- **Do not** reduce N post-hoc "because some of those trials weren't really serious attempts". The trial counter in the ledger is append-only.
- **Do not** substitute PF-derived SR for actual-returns SR when you have the returns. `pf_to_sharpe` assumes IID per-trade returns with uniform win size; real trades have fat tails. Prefer passing `returns` directly when possible (the `skew`/`kurt` correction kicks in).

## Cross-checks against published values

Tests verify the closed-form `E[max SR | N, V=1]` matches Bailey 2014 Table 1 within 3%: N=10 → 1.57, N=100 → 2.53, N=1000 → 3.26, N=10000 → 3.86. Implied `min_pf_for_dsr_pass` for 2000 trades over 7 years matches the WG3 p-hacking table (N=1000 → 1.62) within a tenth of a PF.

## Limitations

The closed-form assumes IID trial SRs with variance V. Trials in a grid search are correlated (nearby configs share signal paths), which makes the true E[max] *lower* than the formula — meaning DSR is **conservative**. WG3 treats this as a feature: better to reject borderline candidates than deploy an overfit.

For highly correlated sweeps (e.g. a 7-dim hyperparameter grid) the effective N is closer to `N / autocorrelation`, but we do not estimate that automatically. When in doubt, use the raw sweep count — the conservative direction.

## Interfaces

```python
from validation.deflated_sharpe import deflated_sharpe, pf_to_sharpe, min_pf_for_dsr_pass

res = deflated_sharpe(returns=daily_pnl, n_trials=ledger.total_trials(), annualization=252)
if not res.passes:
    ledger.reject(cfg, reason=f"DSR={res.dsr:.3f} p={res.p_value:.3f}")
```

CLI for ad-hoc checks: `python -m validation.deflated_sharpe --pf 1.41 --win-rate 0.28 --trades 1700 --trials 1500 --years 6`.
