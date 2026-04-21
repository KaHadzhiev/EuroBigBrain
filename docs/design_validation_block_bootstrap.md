# Design — Block Bootstrap + Walk-Forward Validation

**Status:** implemented 2026-04-21. Gates 2 and 4 of the WG3 validation pipeline.
**Modules:** `python/validation/block_bootstrap.py`, `python/validation/walk_forward.py`.

## Motivation

The GoldBigBrain reversal taught us two lessons. First, **session-shuffle nulls test
the wrong hypothesis** for reversion strategies — we rejected cfg74 for four days before
a proper bootstrap gave p<0.0001 and we reinstated it. Second, the **iid bootstrap used
in that rescue is optimistic**: per-trade PnL is not iid because trades cluster in time,
share the same regime, and inherit volatility auto-correlation. An iid resample breaks
these correlations and produces artificially-tight CIs.

EuroBigBrain Gate 2 therefore mandates the **stationary block bootstrap** (Politis &
Romano 1994) with automatic block length via Politis & White (2004). Gate 4 adds
walk-forward OOS validation to catch strategies that look great in-sample but do not
generalise forward.

## block_bootstrap.py

### API

```python
block_bootstrap_pf(daily_returns, n_iterations=10_000, block_length=None, seed=...)
  -> BootstrapResult
# .as_tuple() -> (mean_pf, p_value, ci_95)
```

### Algorithm

1. **Block length auto-select.** If `block_length is None`, call
   `arch.bootstrap.optimal_block_length(returns)` and take the `stationary`
   column (Politis/Romano variant). Clamp to `[2, n/4]` to prevent degenerate
   block lengths on short series.
2. **Stationary bootstrap.** Use `arch.bootstrap.StationaryBootstrap(bl, arr)`.
   Each resample starts a new block with probability `1/bl`; otherwise extends
   the previous block by one index. This preserves short-range autocorrelation
   while still producing asymptotically iid samples for the PF statistic.
3. **Statistic.** Compute PF = gross_profit / gross_loss on each resample.
4. **Output.** Return observed PF, mean/median of the bootstrap distribution,
   95% percentile CI, and p-value = fraction of resampled PFs ≤ 1.0
   (right-tailed test of null "true PF ≤ 1.0").

### Why stationary vs circular

The circular block bootstrap wraps the series on block boundaries, which works
only if the series is periodic. Financial returns are not periodic, so circular
biases the variance. The stationary bootstrap's geometric block-length
distribution makes resamples genuinely stationary — the right choice here.

### Fallback

If `arch` is unavailable, a pure-numpy stationary bootstrap is used. 10× slower
but correct. The arch path is the default because it ships a vectorised
implementation and the optimal-block-length routine.

### cfg74 reproduction

On cfg74's 1674-trade log with `n_iterations=10_000, seed=20260421`:

| Series | Block length | Observed PF | CI95 | p-value |
|---|---|---|---|---|
| Per-trade PnL | 2.0 (floor) | 1.4086 | [1.21, 1.63] | 0.0000 |
| Daily PnL | 2.30 | 1.6077 | [1.31, 1.98] | 0.0000 |

Both confirm the memory note "bootstrap p<0.0001". The per-trade block length
floors at 2 because cfg74 PnLs are close to serially independent at the trade
level (fresh signal per setup). The daily series has mild autocorrelation,
giving bl≈2.3.

## walk_forward.py

### API

```python
walk_forward(strategy_runner, full_period_data,
             train_years=3, test_months=6, slide_months=6,
             ...)
  -> pd.DataFrame   # one row per fold
# df.attrs['verdict'] -> WalkForwardVerdict
```

### Two modes

- **Frozen trade log mode.** `strategy_runner=None`, `full_period_data` is a
  trade-log DataFrame. The harness slices by date and scores PF/DD on the
  frozen trades. This is the mode used for cfg74 verification — the strategy
  was already run end-to-end; we just re-score per fold.
- **Full retrain mode.** `strategy_runner` is a callable
  `(data_slice, params) -> trade_log` and `full_period_data` exposes
  `slice_by_date(start, end)`. The harness rebuilds training trades AND test
  trades per fold. This catches strategies whose "edge" vanishes when the
  training window shifts.

### PASS criteria (WG3)

Per WG3 §4: **≥ 5/6 folds PF ≥ 1.0** and **no fold DD > 10%**. These are
looser than the final Gate-4 thresholds (PF≥1.2, DD≤25%) — they catch "runs
of losses" rather than "insufficient alpha", allowing better separation when
a strategy trades thinly in some periods.

### cfg74 walk-forward result

6 folds (2020-01 → 2026-01 with 3yr train / 6mo test / 6mo slide):

```
fold  test_end     tePF   teDD    verdict
 1    2023-07-03   1.48    8.84%  PASS
 2    2024-01-03   1.60    6.07%  PASS
 3    2024-07-03   1.26    9.34%  PASS
 4    2025-01-03   1.09   17.47%  FAIL (DD)
 5    2025-07-03   1.29   19.25%  FAIL (DD)
 6    2026-01-03   0.97   28.92%  FAIL (DD + PF)
```

5/6 have PF≥1.0 but only 3/6 clear the 10% DD gate. Overall FAIL. This is
consistent with the known memory `project_beast_drawdowns_wg_2026_04_21.md` —
cfg74/London-beast family has a documented DD issue in 2024-2026 that is
exactly what walk-forward was designed to catch. The module is working.

## Testing

`tests/test_block_bootstrap.py` unit-tests:
1. Known-positive synthetic: mean=+0.5/trade, 1000 samples → p≈0.
2. Known-zero synthetic: mean=0, symmetric → p≈0.5.
3. Fixed-seed reproducibility.
4. cfg74 regression: PF=1.4086 exact.

`tests/test_walk_forward.py`:
1. Synthetic trade log → correct fold count.
2. Known PASS series → verdict.passes True.
3. Known FAIL (negative drift) → False.

## Dependencies

- `arch>=7.0` (for `optimal_block_length` and `StationaryBootstrap`).
- `numpy`, `pandas`, `scipy` (already in EBB requirements).

## Next steps

1. Hook into Gate-2 sweep runner so every candidate config logs its
   `BootstrapResult` into `validation_ledger.parquet`.
2. Parallelise walk-forward folds across Mac prefixes when in full-retrain
   mode (trivially embarrassingly parallel; each fold is independent).
3. Add a Politis/White block-length sanity plot to `findings/` for visual
   inspection — if optimal_block_length returns 200+ on a series, something
   is wrong with the series, not with the bootstrap.
