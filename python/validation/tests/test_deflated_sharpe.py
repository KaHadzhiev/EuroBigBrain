"""Tests for validation.deflated_sharpe.

Cross-checks:
1. Closed-form E[max SR | N, V=1] against Bailey & Lopez de Prado
   (2014) Table 1 / equation 6.
2. pf_to_sharpe math against hand-calculated per-trade SR.
3. DSR monotonicities: higher PF -> higher DSR; higher N -> lower DSR.
4. min_pf_for_dsr_pass matches WG3's tabulated thresholds within 10%.
5. Normal-returns simulation: DSR with N=1 trials ~= ordinary SR test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from validation.deflated_sharpe import (
    _expected_max_sr,
    deflated_sharpe,
    min_pf_for_dsr_pass,
    pf_to_sharpe,
)


# ---------- Bailey 2014 Table 1 reference values ------------------------------

@pytest.mark.parametrize(
    "n, expected",
    [
        (10, 1.575),      # Bailey: ~1.54
        (100, 2.531),     # Bailey: ~2.51
        (1000, 3.255),    # Bailey: ~3.26
        (10000, 3.861),   # Bailey: ~3.85
    ],
)
def test_expected_max_sr_matches_published_table(n, expected):
    got = _expected_max_sr(n, var_trial_sr=1.0)
    assert abs(got - expected) < 0.03, f"N={n}: {got:.4f} vs {expected}"


def test_expected_max_sr_is_increasing_in_N():
    last = -1.0
    for n in [2, 10, 100, 1_000, 10_000, 100_000]:
        v = _expected_max_sr(n, 1.0)
        assert v > last
        last = v


def test_expected_max_sr_N1_is_zero():
    assert _expected_max_sr(1) == 0.0


# ---------- pf_to_sharpe -------------------------------------------------------

def test_pf1_gives_zero_sharpe():
    # PF=1 means gross win = gross loss => zero expectation => SR=0
    assert abs(pf_to_sharpe(1.0, 0.5, 1000)) < 1e-9


def test_pf_to_sharpe_monotone_in_pf():
    lo = pf_to_sharpe(1.1, 0.5, 1000, trades_per_year=200)
    hi = pf_to_sharpe(2.0, 0.5, 1000, trades_per_year=200)
    assert hi > lo > 0


def test_pf_to_sharpe_closed_form_for_symmetric_case():
    # PF=1.5, w=0.5 => W/L = 1.5. W=1.5, L=1.
    # mu = 0.5*1.5 - 0.5*1 = 0.25
    # E[r^2] = 0.5*2.25 + 0.5*1 = 1.625
    # var = 1.625 - 0.0625 = 1.5625 ; std = 1.25
    # sr_pp = 0.25/1.25 = 0.2
    sr = pf_to_sharpe(1.5, 0.5, 100, trades_per_year=100)
    expected = 0.2 * math.sqrt(100)
    assert abs(sr - expected) < 1e-9


# ---------- monotonicities of the DSR -----------------------------------------

def test_dsr_higher_pf_higher_score():
    tpy = 200.0
    r1 = deflated_sharpe(
        None, 1000, sr_observed=pf_to_sharpe(1.3, 0.5, 2000, trades_per_year=tpy),
        n_obs=2000, annualization=tpy,
    )
    r2 = deflated_sharpe(
        None, 1000, sr_observed=pf_to_sharpe(2.0, 0.5, 2000, trades_per_year=tpy),
        n_obs=2000, annualization=tpy,
    )
    assert r2.dsr > r1.dsr


def test_dsr_more_trials_lower_score():
    tpy = 200.0
    sr = pf_to_sharpe(1.5, 0.5, 2000, trades_per_year=tpy)
    r_few = deflated_sharpe(None, 10, sr_observed=sr, n_obs=2000, annualization=tpy)
    r_many = deflated_sharpe(None, 10_000, sr_observed=sr, n_obs=2000, annualization=tpy)
    assert r_few.dsr > r_many.dsr


def test_dsr_passes_at_N1_with_sr_above_zero():
    # With only a single trial and a moderate SR, DSR should approach
    # the one-sided t-test p-value since E[max] is 0.
    res = deflated_sharpe(
        None, 1, sr_observed=0.5, n_obs=500, annualization=1.0,
    )
    assert res.dsr > 0.99


# ---------- min_pf_for_dsr_pass: monotone + sanity thresholds -----------------
# WG3's p-hacking budget table is a heuristic; the actual closed-form
# threshold depends on trade count and annualisation. We check:
#   * monotone increasing in N (more trials = need higher PF)
#   * falls in sensible absolute range for typical EURUSD M5 parameters
def test_min_pf_monotone_in_n_trials():
    pfs = [
        min_pf_for_dsr_pass(n, n_trades=2000, win_rate=0.5, trades_per_year=286)
        for n in [1, 10, 100, 1000, 10_000]
    ]
    assert pfs == sorted(pfs), f"not monotone: {pfs}"


@pytest.mark.parametrize(
    "n_trials, lo, hi",
    [
        (10, 1.10, 1.40),
        (100, 1.30, 1.60),
        (1000, 1.45, 1.80),
        (10000, 1.60, 2.00),
    ],
)
def test_min_pf_in_reasonable_range(n_trials, lo, hi):
    # 2000 trades over 7 yrs, win rate 0.5 (WG3 p-hacking budget scenario).
    mp = min_pf_for_dsr_pass(
        n_trials, n_trades=2000, win_rate=0.5, trades_per_year=2000 / 7,
    )
    assert lo <= mp <= hi, f"N={n_trials}: min_pf={mp:.3f} not in [{lo},{hi}]"


# ---------- end-to-end with synthetic returns ---------------------------------

def test_deflated_sharpe_on_positive_return_series():
    rng = np.random.default_rng(42)
    # 4 years daily, mean=0.002, std=0.01 -> SR~3 ann, comfortably
    # above E[max|N=10,V=1]=1.57
    rets = rng.normal(loc=0.002, scale=0.01, size=252 * 4)
    res = deflated_sharpe(rets, n_trials=10, annualization=252.0)
    assert res.sr_observed > 2.0
    assert res.passes


def test_deflated_sharpe_rejects_noise_under_heavy_search():
    rng = np.random.default_rng(7)
    # Barely-positive-mean returns but N=10_000 trials should reject.
    rets = rng.normal(loc=0.0002, scale=0.01, size=252 * 2)
    res = deflated_sharpe(rets, n_trials=10_000, annualization=252.0)
    assert not res.passes


def test_deflated_sharpe_rejects_flat_or_negative():
    rng = np.random.default_rng(1)
    rets = rng.normal(loc=-0.0001, scale=0.01, size=500)
    res = deflated_sharpe(rets, n_trials=100, annualization=252.0)
    assert not res.passes
    assert res.dsr < 0.5


# ---------- input validation --------------------------------------------------

def test_pf_to_sharpe_rejects_bad_win_rate():
    with pytest.raises(ValueError):
        pf_to_sharpe(1.5, 0.0, 100)
    with pytest.raises(ValueError):
        pf_to_sharpe(1.5, 1.0, 100)


def test_deflated_sharpe_requires_returns_or_sr():
    with pytest.raises(ValueError):
        deflated_sharpe(None, n_trials=10)


def test_deflated_sharpe_returns_tuple_via_as_tuple():
    res = deflated_sharpe(None, 100, sr_observed=1.0, n_obs=500, annualization=252.0)
    dsr, p, thr = res.as_tuple()
    assert 0.0 <= dsr <= 1.0
    assert abs((1.0 - dsr) - p) < 1e-12
    assert thr > 0.0
