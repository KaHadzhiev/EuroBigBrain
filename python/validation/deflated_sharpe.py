"""Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014).

Implements the DSR gatekeeper for EuroBigBrain's Gate-2 validation.
Protects against selection bias when many strategy configurations are
tried: with N=1000 trials, raw PF ~1.60 is needed to clear DSR.

References
----------
- Bailey, D. H., & Lopez de Prado, M. (2014). "The Deflated Sharpe
  Ratio: Correcting for Selection Bias, Backtest Overfitting and
  Non-Normality." Journal of Portfolio Management, 40(5).
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey, Borwein, Lopez de Prado, Zhu (2014). "The Probability of
  Backtest Overfitting."
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

Formulas
--------
sigma_SR = sqrt((1 - gamma3*SR + ((gamma4 - 1)/4) * SR^2) / (T - 1))

E[max SR_null | N trials, V=Var across trial SRs]
  approx= sqrt(V) * ((1 - gamma_em) * Phi^-1(1 - 1/N)
                     + gamma_em      * Phi^-1(1 - 1/(N*e)))
  gamma_em = 0.5772156649 (Euler-Mascheroni)
  default V = 1.0 (unit variance of non-deflated SRs; conservative when
  the actual spread of trial SRs is unknown).

DSR = Phi( (SR_obs - E[max SR_null]) / sigma_SR )
Pass iff DSR > 0.95 (i.e. p-value = 1 - DSR < 0.05).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


@dataclass
class DSRResult:
    sr_observed: float
    sr_threshold: float           # E[max SR | N trials]
    sigma_sr: float
    dsr: float                    # probability real edge in (0,1)
    p_value: float                # 1 - dsr
    n_trials: int
    n_obs: int
    passes: bool                  # dsr > 0.95

    def as_tuple(self):
        return self.dsr, self.p_value, self.sr_threshold


# -- helpers -------------------------------------------------------------------

def _expected_max_sr(n_trials: int, var_trial_sr: float = 1.0) -> float:
    """Bailey/LdP 2014 closed-form expected max of N iid SRs.

    E[max] = sqrt(V) * ((1-g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N*e)))
    """
    if n_trials < 2:
        return 0.0
    g = EULER_MASCHERONI
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(max(var_trial_sr, 1e-12)) * ((1.0 - g) * q1 + g * q2)


def _sigma_sr(
    sr_per_period: float,
    n_obs: int,
    skew: float,
    kurt: float,
    annualization: float = 1.0,
) -> float:
    """Asymptotic std of the estimated SR (Lo 2002, Mertens 2002).

    Input SR is in per-period units. Output is std of SR at
    sqrt(annualization)-scaled frequency:
      sigma_ann_SR = sqrt(annualization) * sigma_perperiod_SR
    """
    if n_obs < 2:
        return float("inf")
    # Bailey/LdP formula uses raw kurtosis gamma4 (Pearson 3=normal);
    # accept either raw (>=1) or excess (=gamma4-3) kurtosis.
    gamma4 = kurt if kurt >= 1.0 else 3.0 + kurt
    sr = sr_per_period
    var = (1.0 - skew * sr + ((gamma4 - 1.0) / 4.0) * sr * sr) / (n_obs - 1.0)
    return math.sqrt(max(var, 1e-12) * max(annualization, 1.0))


# -- public API ----------------------------------------------------------------

def deflated_sharpe(
    returns: Sequence[float] | np.ndarray | None,
    n_trials: int,
    *,
    sr_observed: Optional[float] = None,
    sr_per_period: Optional[float] = None,
    n_obs: Optional[int] = None,
    skew: Optional[float] = None,
    kurt: Optional[float] = None,
    var_trial_sr: float = 1.0,
    annualization: Optional[float] = None,
) -> DSRResult:
    """Compute the Deflated Sharpe Ratio (Bailey/LdP 2014).

    Either pass `returns` (per-period) or `sr_observed` + `n_obs`.
    `annualization` scales SR and sigma to annual units (e.g. 252 for
    daily). `var_trial_sr` is the variance of SRs across the N trials
    (1.0 = Bailey/LdP conservative default).
    Returns DSRResult whose .as_tuple() is (DSR, p_value, SR_threshold).
    """
    ann = float(annualization) if annualization is not None else 1.0
    if returns is not None:
        arr = np.asarray(returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            raise ValueError("need >=2 finite returns")
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1))
        if sd <= 0:
            raise ValueError("zero variance in returns")
        sr_pp = mu / sd
        sr_ann = sr_pp * math.sqrt(ann)
        n_obs = arr.size
        if skew is None:
            skew = float(stats.skew(arr, bias=False))
        if kurt is None:
            kurt = float(stats.kurtosis(arr, fisher=False, bias=False))
    else:
        if sr_observed is None or n_obs is None:
            raise ValueError("pass returns OR (sr_observed + n_obs)")
        sr_ann = float(sr_observed)
        if sr_per_period is not None:
            sr_pp = float(sr_per_period)
        else:
            sr_pp = sr_ann / math.sqrt(ann) if ann > 1.0 else sr_ann
        if skew is None:
            skew = 0.0
        if kurt is None:
            kurt = 3.0

    e_max = _expected_max_sr(n_trials, var_trial_sr=var_trial_sr)
    sig = _sigma_sr(sr_pp, n_obs, skew, kurt, annualization=ann)
    z = (sr_ann - e_max) / sig if sig > 0 else 0.0
    dsr = float(stats.norm.cdf(z))
    p = 1.0 - dsr
    return DSRResult(
        sr_observed=sr_ann,
        sr_threshold=e_max,
        sigma_sr=sig,
        dsr=dsr,
        p_value=p,
        n_trials=n_trials,
        n_obs=int(n_obs),
        passes=dsr > 0.95,
    )


def pf_to_sharpe(
    pf: float,
    win_rate: float,
    n_trades: int,
    *,
    avg_win_loss_ratio: Optional[float] = None,
    trades_per_year: Optional[float] = None,
) -> float:
    """PF + win rate -> annualised Sharpe.

    Unit-loss parametrisation: L=1, W = PF*(1-w)/w, mu_t = (1-w)*(PF-1),
    Var_t = w*W^2 + (1-w) - mu_t^2. Annualise with sqrt(trades_per_year)
    (default: n_trades/7 assuming a 7-year backtest horizon).
    """
    if not (0.0 < win_rate < 1.0):
        raise ValueError("win_rate must be in (0,1)")
    if pf <= 0:
        raise ValueError("pf must be > 0")
    w = win_rate
    if avg_win_loss_ratio is not None:
        wl = avg_win_loss_ratio
    else:
        wl = pf * (1.0 - w) / w            # W / L
    # unit loss magnitude
    W = wl
    L = 1.0
    mu = w * W - (1.0 - w) * L
    second = w * W * W + (1.0 - w) * L * L
    var = second - mu * mu
    if var <= 0:
        return 0.0
    sr_per_trade = mu / math.sqrt(var)
    if trades_per_year is None:
        trades_per_year = max(n_trades / 7.0, 1.0)  # default: 7yr horizon
    return sr_per_trade * math.sqrt(trades_per_year)


def min_pf_for_dsr_pass(
    n_trials: int,
    n_trades: int,
    win_rate: float = 0.5,
    *,
    threshold: float = 0.95,
    skew: float = 0.0,
    kurt: float = 3.0,
    var_trial_sr: float = 1.0,
    trades_per_year: Optional[float] = None,
) -> float:
    """Binary-search the minimum raw PF whose DSR just exceeds `threshold`.

    Assumes iid per-trade returns parametrised by (pf, win_rate).
    """
    tpy = trades_per_year if trades_per_year is not None else max(n_trades / 7.0, 1.0)
    lo, hi = 1.0, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        sr_ann = pf_to_sharpe(mid, win_rate, n_trades, trades_per_year=tpy)
        res = deflated_sharpe(
            None,
            n_trials,
            sr_observed=sr_ann,
            n_obs=n_trades,
            skew=skew,
            kurt=kurt,
            var_trial_sr=var_trial_sr,
            annualization=tpy,
        )
        if res.dsr >= threshold:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-4:
            break
    return hi


# -- CLI -----------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Deflated Sharpe Ratio (Bailey/LdP 2014)")
    ap.add_argument("--pf", type=float, required=True, help="profit factor")
    ap.add_argument("--win-rate", type=float, required=True, help="win rate in (0,1)")
    ap.add_argument("--trades", type=int, required=True, help="number of trades")
    ap.add_argument("--trials", type=int, required=True, help="number of configs tried (sweep size)")
    ap.add_argument("--years", type=float, default=7.0, help="sample horizon (years)")
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--kurt", type=float, default=3.0, help="raw kurtosis (3=normal)")
    ap.add_argument("--var-trial-sr", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()

    tpy = args.trades / max(args.years, 0.25)
    sr_ann = pf_to_sharpe(args.pf, args.win_rate, args.trades, trades_per_year=tpy)
    res = deflated_sharpe(
        None,
        args.trials,
        sr_observed=sr_ann,
        n_obs=args.trades,
        skew=args.skew,
        kurt=args.kurt,
        var_trial_sr=args.var_trial_sr,
        annualization=tpy,
    )
    min_pf = min_pf_for_dsr_pass(
        args.trials, args.trades, args.win_rate,
        threshold=args.threshold, skew=args.skew, kurt=args.kurt,
        var_trial_sr=args.var_trial_sr,
        trades_per_year=tpy,
    )
    verdict = "PASS" if res.passes else "FAIL"
    print(f"PF={args.pf:.3f}  WR={args.win_rate:.3f}  trades={args.trades}  trials={args.trials}  years={args.years}")
    print(f"  trades/year              = {tpy:.1f}")
    print(f"  SR_annualised            = {res.sr_observed:.4f}")
    print(f"  E[max SR | N={args.trials}]  = {res.sr_threshold:.4f}")
    print(f"  sigma_SR (annualised)    = {res.sigma_sr:.4f}")
    print(f"  DSR (1 - p)              = {res.dsr:.4f}")
    print(f"  p-value                  = {res.p_value:.4f}")
    print(f"  min PF to DSR>{args.threshold:.2f}     = {min_pf:.3f}")
    print(f"  verdict                  = {verdict}")
    return 0 if res.passes else 1


if __name__ == "__main__":
    raise SystemExit(_main())
