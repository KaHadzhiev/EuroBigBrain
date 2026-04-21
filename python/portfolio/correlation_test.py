#!/usr/bin/env python3
"""Correlation gate for EuroBigBrain portfolio.

Hard rule from WG2: 2-3 uncorrelated edges > 1 strong edge on $1k retail — but
"uncorrelated" must actually be uncorrelated. Before calling `combine_strategies`,
run `validate_uncorrelated(logs, threshold=0.3)`. If pairwise daily-PnL Pearson
correlation exceeds `threshold`, the gate FAILS with the offending pair and value.

Threshold rationale (0.30 default):
  - NY Reversal Fade (session 15-17 UTC) vs Asian Range Breakout (07-10 UTC)
    share zero bars of exposure — empirical corr should be near 0.
  - ML-gate on top of either is cross-sectional meta, corr target <0.4.
  - WG2 spec says "<0.3 (different sessions, opposite logic)" for session-orthogonal
    legs, "<0.4" for ML-gated meta. We keep the stricter 0.30 as default and let
    callers relax to 0.40 when mixing ML meta.

Historical reference (gold): fg vs rsisl have 96% TIME-overlap but combined DD
still came in below simple addition — ergo time-overlap is NOT the same as daily-PnL
correlation. We only gate on *daily PnL* corr here. Strategies can fire at the same
hour and still decorrelate at the day level if directional PnL differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class CorrelationReport:
    passed: bool
    threshold: float
    max_pair: tuple[str, str] | None
    max_value: float
    matrix: pd.DataFrame
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            return (
                f"PASS (threshold {self.threshold:.2f}) — "
                f"worst pair {self.max_pair} = {self.max_value:.3f}"
            )
        return (
            f"FAIL (threshold {self.threshold:.2f}) — "
            f"{self.max_pair[0]} vs {self.max_pair[1]} = {self.max_value:.3f}. {self.reason}"
        )


def _daily_pnl(df: pd.DataFrame) -> pd.Series:
    if "open_time" not in df.columns or "pnl" not in df.columns:
        raise ValueError("trade log missing open_time/pnl columns")
    dt = pd.to_datetime(df["open_time"]).dt.floor("D")
    s = df.assign(_d=dt).groupby("_d")["pnl"].sum()
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def _names(trade_logs: Sequence[pd.DataFrame]) -> list[str]:
    out = []
    for i, df in enumerate(trade_logs):
        if "strategy" in df.columns and len(df):
            out.append(str(df["strategy"].iloc[0]))
        else:
            out.append(f"strat{i}")
    # de-dup if two logs share the same label
    seen: dict[str, int] = {}
    result = []
    for n in out:
        if n in seen:
            seen[n] += 1
            result.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            result.append(n)
    return result


def pairwise_daily_correlation(trade_logs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Return Pearson daily-PnL correlation matrix over the union of trading days."""
    if not trade_logs:
        return pd.DataFrame()
    names = _names(trade_logs)
    series = [_daily_pnl(df) for df in trade_logs]
    idx = sorted(set().union(*(s.index for s in series)))
    if not idx:
        return pd.DataFrame()
    frame = pd.DataFrame({n: s.reindex(idx).fillna(0.0) for n, s in zip(names, series)})
    return frame.corr()


def validate_uncorrelated(
    trade_logs: Sequence[pd.DataFrame],
    threshold: float = 0.3,
    min_overlap_days: int = 60,
) -> CorrelationReport:
    """Gate: fail if any off-diagonal |corr| exceeds threshold.

    Args:
      trade_logs: list of trade-log DataFrames (open_time, pnl, [strategy]).
      threshold: max absolute pairwise daily-PnL Pearson correlation (default 0.30).
      min_overlap_days: minimum shared trading days per pair, else FAIL as
        "insufficient overlap to validate correlation".
    """
    if len(trade_logs) < 2:
        return CorrelationReport(
            passed=False, threshold=threshold, max_pair=None, max_value=0.0,
            matrix=pd.DataFrame(),
            reason="need >=2 strategies to test correlation",
        )

    names = _names(trade_logs)
    series = [_daily_pnl(df) for df in trade_logs]

    # overlap check
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            shared = series[i].index.intersection(series[j].index)
            if len(shared) < min_overlap_days:
                return CorrelationReport(
                    passed=False,
                    threshold=threshold,
                    max_pair=(names[i], names[j]),
                    max_value=float("nan"),
                    matrix=pd.DataFrame(),
                    reason=(
                        f"only {len(shared)} shared trading days "
                        f"(need >={min_overlap_days})"
                    ),
                )

    matrix = pairwise_daily_correlation(trade_logs)
    # off-diagonal abs max
    abs_m = matrix.abs().copy()
    for i in range(len(abs_m)):
        abs_m.iat[i, i] = 0.0
    max_val = float(abs_m.values.max()) if abs_m.size else 0.0
    ij = np.unravel_index(int(abs_m.values.argmax()), abs_m.shape) if abs_m.size else (0, 0)
    max_pair = (matrix.index[ij[0]], matrix.columns[ij[1]])

    passed = max_val <= threshold
    reason = "" if passed else (
        f"pair correlation {max_val:.3f} exceeds {threshold:.2f} — legs are too similar; "
        f"combining will NOT yield DD-smoothing benefit. Re-check session/logic orthogonality."
    )
    return CorrelationReport(
        passed=passed,
        threshold=threshold,
        max_pair=max_pair,
        max_value=max_val,
        matrix=matrix,
        reason=reason,
    )


# ---------- self-test ----------

def _synth(seed: int, n: int, base_ret: float = 0.4, shared_signal: np.ndarray | None = None,
           shared_weight: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = np.cumsum(rng.integers(60, 480, size=n))
    opens = pd.Timestamp("2024-01-02") + pd.to_timedelta(minutes, unit="m")
    closes = opens + pd.to_timedelta(rng.integers(20, 120, size=n), unit="m")
    noise = rng.normal(loc=base_ret, scale=5.0, size=n)
    if shared_signal is not None:
        # inject partial shared daily drift to drive up correlation
        day_idx = (opens.normalize() - opens.normalize().min()).days.to_numpy()
        day_idx = np.clip(day_idx, 0, len(shared_signal) - 1)
        pnls = (1 - shared_weight) * noise + shared_weight * shared_signal[day_idx] * 5.0
    else:
        pnls = noise
    return pd.DataFrame(
        {"open_time": opens, "close_time": closes, "pnl": pnls, "strategy": f"strat{seed}"}
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    shared = rng.standard_normal(800)

    print("== Uncorrelated legs ==")
    uncor = [_synth(1, 400), _synth(2, 350), _synth(3, 420)]
    rep = validate_uncorrelated(uncor, threshold=0.3)
    print(rep.summary())
    print(rep.matrix.round(3))

    print("\n== Correlated legs (forced 70% shared drift) ==")
    cor = [_synth(10, 400, shared_signal=shared, shared_weight=0.7),
           _synth(11, 350, shared_signal=shared, shared_weight=0.7),
           _synth(12, 420)]
    rep2 = validate_uncorrelated(cor, threshold=0.3)
    print(rep2.summary())
    print(rep2.matrix.round(3))
