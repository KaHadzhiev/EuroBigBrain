# EuroBigBrain — Portfolio Combine Design

**Author:** WG2 portfolio-engineering sub-group
**Date:** 2026-04-21
**Status:** initial implementation

## Purpose

WG2's archetype catalog concluded that a **$1k EUR/USD retail account** is better served by running **2-3 uncorrelated PF=1.35 edges** than a single PF=1.7 edge. The binding constraint is drawdown (DD), not peak profit factor: anything above ~15% DD crosses the psychological and broker-liquidation cliff before compounding can carry the account into safety.

The London Beasts project proved the principle on gold (fg + rsisl = PF 1.87 / 13.9% DD on $2k, risk-adjusted winner over any solo variant). This module brings the same architecture to EUR/USD.

## Scope of this module

Two Python files under `python/portfolio/`:

- `combine.py` — takes a list of per-strategy MT5/sim trade logs and produces a joint equity curve, PF, DD, recovery factor, daily-PnL Sharpe, concurrency histogram, and per-year breakdown. Outputs CSVs and a PNG equity-curve plot on demand.
- `correlation_test.py` — the mandatory **gate** that must return `passed=True` before combining. Computes pairwise daily-PnL Pearson correlation, fails on any pair above threshold, and also fails on insufficient overlap (<60 shared trading days).

Both are <300 lines. No external deps beyond pandas / numpy / matplotlib (matplotlib is optional; missing = skip PNG).

## Public API

```python
from portfolio.combine import combine_strategies, write_artifacts
from portfolio.correlation_test import validate_uncorrelated

report = validate_uncorrelated([df_a, df_b, df_c], threshold=0.30)
assert report.passed, report.summary()
metrics = combine_strategies([df_a, df_b, df_c], deposit=1000, max_concurrent_positions=2)
paths = write_artifacts(metrics, out_dir="runs/portfolio/v1", tag="ny_asia_ml")
```

Each input DataFrame must have `open_time` and `pnl`; `close_time` and `strategy` are recommended (and required for correlation labelling).

## Key design decisions

### Position-overlap penalty

Unlike the gold `portfolio_math.py` (which chronologically merges trades and trusts MT5's already-compounded PnL), EuroBigBrain's combiner introduces an **explicit overlap penalty**. When >N positions are open simultaneously, the *excess* trade's PnL is multiplied by `overlap_penalty` (default 0.5).

Rationale: a $1k account holding 3 positions at once cannot bear 3 full-size losses before the broker liquidates; margin doubles (then triples); the PnL earned by the third simultaneous position is not risk-free capacity — it carries hidden tail exposure. Haircutting the excess PnL symmetrically (both wins and losses) is a conservative acknowledgement that the account was operating outside its design envelope during those bars.

A future hardening would switch this from a flat 50% haircut to an exposure-based risk scalar (shrink position size when margin utilisation > 40%), but that requires lot-size data the sim doesn't always emit, so we start with the simpler, more honest penalty.

### Daily-PnL as the correlation unit

WG2's target is 0.30 correlation between legs. **Time-overlap** (trades firing at the same clock minute) is *not* what we gate on — gold's fg and rsisl time-overlap 96% yet their combined DD is lower than pure addition. The relevant quantity is whether their *daily* PnL moves together. A leg that fires 10 profitable trades at 16:00 UTC while another leg fires 10 losing trades at 17:00 UTC has very low daily correlation despite high time-overlap. We compute daily correlation on the union of trading days (missing days = 0 PnL).

### Threshold = 0.30 (hard default)

Why 0.30 and not 0.50 or 0.70?

- WG2 spec: "NY Reversal Fade vs Asian Breakout <0.3 (different sessions, opposite logic); ML vs either <0.4".
- Empirically, two FX strategies with daily-PnL correlation >0.30 offer little DD-smoothing: combined variance = var_A + var_B + 2*rho*sqrt(var_A*var_B), and at rho=0.5 the cross-term is ~40% of variance reduction you'd get vs rho=0. So above 0.30-0.35 the "2 edges > 1 edge" thesis breaks down.
- We keep 0.30 as the default. Callers mixing an ML meta-gate with a session strategy may explicitly relax to 0.40 (documented limit, not silent).

Above 0.40 we have strong prior that the legs are redundant fits of the same underlying factor — combining them doubles position sizing without adding independent information. Gate always fails in that regime.

### Outputs that survive to MT5 deploy

Every combine run writes to disk:

- `<tag>_equity.csv` — trade-indexed equity walk
- `<tag>_daily_pnl.csv` — daily bucketed PnL for Sharpe / regime analysis
- `<tag>_correlation.csv` — pairwise Pearson matrix
- `<tag>_equity.png` — equity + drawdown dual-axis plot

These feed directly into the auditor's review and the validation memo before any live deploy request.

## Differences vs London Beasts `portfolio_math.py`

| Aspect | London Beasts (gold) | EuroBigBrain |
|---|---|---|
| Shape | CLI script with hardcoded paths | Importable module with clean API |
| Overlap handling | implicit (trust MT5 compounding) | explicit 50% haircut beyond cap |
| Correlation gate | post-hoc only (printed) | mandatory pre-combine gate |
| Daily-PnL series | not computed | first-class output |
| Sharpe | per-trade only | per-trade **and** annualised daily |
| Recovery factor | not computed | first-class output |
| Plot output | no | PNG equity+DD dual panel |

## Next steps

Once the first 2-3 EBB strategy candidates pass their individual null tests (per WG2 top 5: NY Reversal Fade, ML Direction, Asian Breakout), run `validate_uncorrelated` on their 2020-2026 trade logs. Any pair >0.30 triggers a re-design before combining. Surviving triples feed `combine_strategies` with `deposit=1000, max_concurrent_positions=2` (conservative for $1k). Target: combined PF>=1.4, DD<=12%, 20-30 trades/month, 6/7+ years green.

No live deploy until user greenlights and walk-forward 6/6 folds PASS on the **combined** logs, not just per-leg.
