"""Unit tests for validation/walk_forward.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.walk_forward import (  # noqa: E402
    _max_dd_pct,
    _pf,
    walk_forward,
    WalkForwardVerdict,
)


def _make_synthetic_log(seed: int, n_years: int = 6, trades_per_day: int = 2,
                       mean: float = 1.0, std: float = 8.0,
                       start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    n_days = 252 * n_years
    rows = []
    for d in range(n_days):
        dt = start_ts + pd.Timedelta(days=d * (365.25 / 252.0))
        for k in range(trades_per_day):
            rows.append({
                "open_time": dt + pd.Timedelta(hours=9 + k),
                "pnl": rng.normal(loc=mean, scale=std),
            })
    return pd.DataFrame(rows)


def test_max_dd_monotonic_up():
    pnls = np.array([1.0, 1.0, 1.0, 1.0])
    assert _max_dd_pct(pnls) == 0.0


def test_max_dd_negative():
    pnls = np.array([1.0, -500.0, 1.0])  # starting 1000 -> 501 -> -99% territory
    assert _max_dd_pct(pnls) > 40.0


def test_pf_helper():
    assert _pf(np.array([1.0, -0.5, 2.0])) == pytest.approx(6.0)


def test_walk_forward_6_folds():
    df = _make_synthetic_log(seed=1, n_years=6)
    out = walk_forward(None, df, train_years=3, test_months=6, slide_months=6)
    assert len(out) >= 5
    assert "verdict" in out.attrs
    v = out.attrs["verdict"]
    assert isinstance(v, WalkForwardVerdict)


def test_walk_forward_known_pass():
    df = _make_synthetic_log(seed=2, n_years=6, mean=4.0, std=4.0)
    out = walk_forward(None, df, train_years=3, test_months=6, slide_months=6)
    v = out.attrs["verdict"]
    # Strong positive drift should pass fold-PF gate on most/all folds
    assert v.folds_pass >= 5


def test_walk_forward_known_fail():
    df = _make_synthetic_log(seed=3, n_years=6, mean=-2.0, std=5.0)
    out = walk_forward(None, df, train_years=3, test_months=6, slide_months=6)
    v = out.attrs["verdict"]
    assert v.passes is False
    # negative drift should have tePF < 1 on most folds
    assert v.folds_pass <= 1


def test_walk_forward_custom_time_col():
    df = _make_synthetic_log(seed=4, n_years=6).rename(columns={"open_time": "entry_ts"})
    out = walk_forward(None, df, time_col="entry_ts",
                       train_years=3, test_months=6, slide_months=6)
    assert not out.empty


def test_walk_forward_cfg74_regression():
    """cfg74 gold log: 6 folds expected, matches original walk_forward script."""
    gold = Path(r"C:\Users\kahad\IdeaProjects\GoldBigBrain\results\news_probe\cfg74_trades.csv")
    if not gold.exists():
        pytest.skip("cfg74 gold log not available")
    df = pd.read_csv(gold)
    out = walk_forward(None, df, train_years=3, test_months=6, slide_months=6)
    assert len(out) == 6
    # first fold should be profitable (documented: all years positive 2020-2023)
    assert out.iloc[0]["test_pf"] >= 1.0
