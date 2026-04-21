"""Stationary block bootstrap for autocorrelation-aware PF significance.

Implements Gate-2 of the WG3 validation pipeline. Use this (not iid) bootstrap
on any per-period returns series (daily PnL, per-trade PnL) when the series has
autocorrelation (almost always true for financial strategies).

Method
------
1. Compute optimal block length via Politis & White (2004) [arch.bootstrap.
   optimal_block_length]. The "stationary" column is the one we want (matches
   Politis & Romano 1994 stationary bootstrap).
2. Resample block-wise with geometric block-length distribution and wrap-around
   to preserve short-range serial dependence.
3. For each resample, compute Profit Factor.
4. Report:
     - observed PF
     - mean bootstrap PF
     - 95% CI (2.5, 97.5 percentiles)
     - p-value for null H0: "true PF <= 1.0"
       (equivalent to testing whether mean return > 0 via PF framing)

Design notes
------------
- For PF, we bootstrap the *returns* vector directly (PF is scale-invariant
  under block resampling, so no need to bootstrap signed residuals).
- The null "PF <= 1.0" is equivalent to "gross_profit <= gross_loss". We
  approximate p-value as the fraction of resampled PFs that fall at or below
  1.0 (a right-tailed test). Under standard identification (strategy has
  positive expectancy), most resamples will have PF > 1 and p will be small.
- When `daily_returns` has few points (e.g. <100), optimal_block_length can be
  unstable. We clamp to [2, len/4].

CLI
---
    python -m validation.block_bootstrap --trades cfg74_trades.csv
        --daily --iters 10000

References
----------
- Politis & Romano (1994). "The Stationary Bootstrap." JASA 89(428).
- Politis & White (2004). "Automatic Block-Length Selection for the Dependent
  Bootstrap."
- arch 7.x docs: arch.bootstrap.optimal_block_length,
  arch.bootstrap.StationaryBootstrap.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from arch.bootstrap import StationaryBootstrap, optimal_block_length
    ARCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    ARCH_AVAILABLE = False


# -- types ---------------------------------------------------------------------

@dataclass
class BootstrapResult:
    pf_observed: float
    mean_pf: float
    median_pf: float
    ci_low: float
    ci_high: float
    p_value: float              # P(PF_boot <= 1.0)
    block_length: float
    n_iterations: int
    n_obs: int

    def as_tuple(self) -> Tuple[float, float, Tuple[float, float]]:
        """WG3-specified return: (mean_pf, p_value, ci_95)."""
        return self.mean_pf, self.p_value, (self.ci_low, self.ci_high)


# -- core ----------------------------------------------------------------------

def _pf(returns: np.ndarray) -> float:
    """Profit Factor = gross_profit / gross_loss."""
    r = np.asarray(returns, dtype=float)
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)


def _auto_block_length(returns: np.ndarray) -> float:
    """Use Politis-White (2004) optimal block length, clamp to sanity range."""
    if not ARCH_AVAILABLE:
        raise RuntimeError(
            "arch package not installed. `pip install arch` to enable "
            "optimal_block_length. Pass block_length explicitly to skip."
        )
    n = len(returns)
    if n < 20:
        return max(2.0, float(n) ** 0.5)
    try:
        tbl = optimal_block_length(returns)
        # column 'stationary' = Politis/Romano stationary bootstrap
        bl = float(tbl["stationary"].iloc[0])
    except Exception:
        bl = float(n) ** (1.0 / 3.0)
    # clamp: at least 2, at most quarter of sample
    bl = max(2.0, min(bl, n / 4.0))
    return bl


def block_bootstrap_pf(
    daily_returns: Union[Sequence[float], np.ndarray, pd.Series],
    n_iterations: int = 10_000,
    block_length: Optional[float] = None,
    seed: int = 20260421,
) -> BootstrapResult:
    """Stationary block-bootstrap PF with autocorrelation preservation.

    Parameters
    ----------
    daily_returns : array-like
        Per-period returns (daily PnL preferred; per-trade PnL also ok when
        trade-interarrival times are not themselves an edge signal).
    n_iterations : int, default 10_000
        Number of bootstrap resamples.
    block_length : float, optional
        Expected block length for the stationary bootstrap. If None, auto
        via Politis & White (2004).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    BootstrapResult. `.as_tuple()` returns (mean_pf, p_value, ci_95) per WG3.
    """
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 30:
        raise ValueError(
            f"need >=30 observations for block bootstrap, got {arr.size}"
        )
    n = arr.size

    if block_length is None:
        block_length = _auto_block_length(arr)

    if not ARCH_AVAILABLE:
        # pure-numpy fallback: stationary bootstrap with geometric block length
        pf_boot = _numpy_stationary_bootstrap_pf(
            arr, n_iter=n_iterations, avg_block=block_length, seed=seed
        )
    else:
        rs = np.random.default_rng(seed)
        # arch's StationaryBootstrap expects integer RandomState/int seed
        bs = StationaryBootstrap(block_length, arr, seed=int(rs.integers(0, 2**31 - 1)))
        pf_boot = np.empty(n_iterations, dtype=float)
        for i, data in enumerate(bs.bootstrap(n_iterations)):
            sample = data[0][0]
            pf_boot[i] = _pf(sample)

    pf_obs = _pf(arr)
    finite = pf_boot[np.isfinite(pf_boot)]
    if finite.size == 0:
        raise RuntimeError("bootstrap produced no finite PFs")
    mean_pf = float(finite.mean())
    median_pf = float(np.median(finite))
    ci_lo, ci_hi = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
    # right-tailed: null is "true PF <= 1.0"
    p_value = float((finite <= 1.0).mean())

    return BootstrapResult(
        pf_observed=pf_obs,
        mean_pf=mean_pf,
        median_pf=median_pf,
        ci_low=ci_lo,
        ci_high=ci_hi,
        p_value=p_value,
        block_length=float(block_length),
        n_iterations=int(n_iterations),
        n_obs=int(n),
    )


# -- numpy fallback ------------------------------------------------------------

def _numpy_stationary_bootstrap_pf(
    arr: np.ndarray, n_iter: int, avg_block: float, seed: int
) -> np.ndarray:
    """Stationary bootstrap (Politis-Romano 1994), pure-numpy PF.

    Each new bar either starts a new block (prob p=1/avg_block) or continues
    the previous block by one step (prob 1-p). Wrap-around at array end.
    """
    rng = np.random.default_rng(seed)
    n = arr.size
    p = 1.0 / max(avg_block, 1.0)
    pfs = np.empty(n_iter, dtype=float)

    for it in range(n_iter):
        indices = np.empty(n, dtype=np.int64)
        indices[0] = rng.integers(0, n)
        u = rng.random(n - 1)
        new_block = u < p
        for k in range(1, n):
            if new_block[k - 1]:
                indices[k] = rng.integers(0, n)
            else:
                indices[k] = (indices[k - 1] + 1) % n
        sample = arr[indices]
        pfs[it] = _pf(sample)
    return pfs


# -- daily aggregation helper --------------------------------------------------

def trades_to_daily_pnl(trades: pd.DataFrame, time_col: str = "close_time",
                       pnl_col: str = "pnl") -> np.ndarray:
    """Aggregate a trade log into daily PnL totals."""
    df = trades.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["__d"] = df[time_col].dt.date
    daily = df.groupby("__d")[pnl_col].sum()
    return daily.to_numpy(dtype=float)


# -- CLI -----------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(
        description="Stationary block bootstrap PF (WG3 Gate 2)"
    )
    ap.add_argument("--trades", type=Path, required=True,
                    help="path to trade log CSV (needs pnl + close_time cols)")
    ap.add_argument("--daily", action="store_true",
                    help="aggregate to daily PnL before bootstrapping")
    ap.add_argument("--pnl-col", default="pnl")
    ap.add_argument("--time-col", default="close_time")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--block", type=float, default=None,
                    help="block length; auto if omitted")
    ap.add_argument("--seed", type=int, default=20260421)
    args = ap.parse_args()

    df = pd.read_csv(args.trades)
    if args.time_col not in df.columns:
        # Goldie-style log uses close_time; tolerate open_time fallback
        for cand in ("close_time", "open_time", "time"):
            if cand in df.columns:
                args.time_col = cand
                break
    if args.daily:
        returns = trades_to_daily_pnl(df, time_col=args.time_col, pnl_col=args.pnl_col)
    else:
        returns = df[args.pnl_col].to_numpy(dtype=float)

    res = block_bootstrap_pf(
        returns,
        n_iterations=args.iters,
        block_length=args.block,
        seed=args.seed,
    )
    print(f"input              : {args.trades}  (daily={args.daily})")
    print(f"n_obs              : {res.n_obs}")
    print(f"block_length       : {res.block_length:.3f}")
    print(f"observed PF        : {res.pf_observed:.4f}")
    print(f"bootstrap mean PF  : {res.mean_pf:.4f}")
    print(f"bootstrap median   : {res.median_pf:.4f}")
    print(f"95% CI             : [{res.ci_low:.4f}, {res.ci_high:.4f}]")
    print(f"P(PF_boot <= 1.0)  : {res.p_value:.6f}")
    verdict = "PASS" if res.p_value < 0.01 else "FAIL"
    print(f"Gate-2 verdict     : {verdict} (threshold p < 0.01)")
    return 0 if res.p_value < 0.01 else 1


if __name__ == "__main__":
    raise SystemExit(_main())
