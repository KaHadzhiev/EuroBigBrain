"""Unit tests for validation/block_bootstrap.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.block_bootstrap import (  # noqa: E402
    _pf,
    block_bootstrap_pf,
    trades_to_daily_pnl,
)


def test_pf_basic():
    assert _pf(np.array([1.0, 2.0, -1.0])) == pytest.approx(3.0)
    assert _pf(np.array([-1.0, -2.0])) == 0.0
    assert _pf(np.array([1.0, 2.0])) == float("inf")


def test_bootstrap_positive_series():
    rng = np.random.default_rng(7)
    # highly positive-expectancy synthetic: PF ~= 3
    pos = rng.exponential(scale=1.0, size=400)
    neg = -rng.exponential(scale=0.33, size=300)
    series = np.concatenate([pos, neg])
    rng.shuffle(series)
    res = block_bootstrap_pf(series, n_iterations=1000, seed=42)
    assert res.pf_observed > 2.0
    assert res.p_value < 0.01
    assert res.ci_low > 1.0


def test_bootstrap_zero_expectation():
    rng = np.random.default_rng(9)
    # mean-zero symmetric → PF should straddle 1, p ~0.5
    series = rng.standard_normal(400)
    res = block_bootstrap_pf(series, n_iterations=1000, seed=42)
    assert 0.3 < res.p_value < 0.7


def test_bootstrap_reproducibility():
    rng = np.random.default_rng(1)
    series = rng.normal(loc=0.1, size=500)
    r1 = block_bootstrap_pf(series, n_iterations=500, seed=111)
    r2 = block_bootstrap_pf(series, n_iterations=500, seed=111)
    assert r1.mean_pf == r2.mean_pf
    assert r1.p_value == r2.p_value


def test_block_length_autoselect():
    rng = np.random.default_rng(3)
    series = rng.normal(size=1000)
    res = block_bootstrap_pf(series, n_iterations=200, seed=5)
    # Clamp rule: 2 <= bl <= n/4
    assert 2.0 <= res.block_length <= 250.0


def test_explicit_block_length():
    rng = np.random.default_rng(4)
    series = rng.normal(size=500)
    res = block_bootstrap_pf(series, n_iterations=200, block_length=7.0, seed=5)
    assert res.block_length == 7.0


def test_trades_to_daily():
    df = pd.DataFrame({
        "close_time": ["2024-01-01 09:00", "2024-01-01 14:00", "2024-01-02 10:00"],
        "pnl": [1.0, 2.0, -0.5],
    })
    daily = trades_to_daily_pnl(df)
    assert len(daily) == 2
    assert daily[0] == 3.0
    assert daily[1] == -0.5


def test_cfg74_regression():
    """cfg74 should reproduce PF=1.4086 and p=0.0000 from memory note."""
    gold = Path(r"C:\Users\kahad\IdeaProjects\GoldBigBrain\results\news_probe\cfg74_trades.csv")
    if not gold.exists():
        pytest.skip("cfg74 gold log not available on this machine")
    df = pd.read_csv(gold)
    res = block_bootstrap_pf(df["pnl"].to_numpy(), n_iterations=2000, seed=20260421)
    assert res.pf_observed == pytest.approx(1.4086, abs=1e-3)
    assert res.p_value == pytest.approx(0.0, abs=1e-3)
    assert res.ci_low > 1.0
