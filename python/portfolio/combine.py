#!/usr/bin/env python3
"""EuroBigBrain portfolio combiner — 2-3 uncorrelated EUR/USD strategies on one account.

Portfolio thesis (WG2): 2-3 uncorrelated PF=1.35 edges outperform one PF=1.7 edge
for $1k retail accounts because the capital-efficiency gate is DD, not peak PF.

Differences vs GoldBigBrain `portfolio_math.py`:
  - Module API (reusable function), not CLI-only script.
  - Explicit position-overlap penalty: when >N strategies hold simultaneously,
    account margin/exposure multiplies — we down-weight overlapped PnL to reflect
    the reduced room for a fresh loss before hitting a broker-level margin call.
  - Daily-PnL series + correlation matrix as first-class outputs.
  - CSV + PNG equity-curve artifacts.

Expected trade-log columns: open_time (str/dt), close_time (optional str/dt),
  pnl (float, USD on 1k account), strategy (optional str).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


@dataclass
class PortfolioMetrics:
    """Combined portfolio summary."""

    n_trades: int
    pf: float
    total_pnl: float
    ending_equity: float
    starting_equity: float
    return_pct: float
    max_dd_pct: float
    max_dd_usd: float
    recovery_factor: float  # total_pnl / max_dd_usd
    sharpe_daily: float  # daily-pnl sharpe, annualised
    sharpe_per_trade: float  # per-trade sharpe (mean/std * sqrt(n))
    trades_per_month: float
    win_rate: float
    max_concurrent: int
    overlap_hit_count: int  # number of moments >= max_concurrent_positions was hit
    concurrent_hist: dict = field(default_factory=dict)
    correlation_matrix: pd.DataFrame | None = None
    daily_pnl: pd.Series | None = None
    equity_curve: pd.Series | None = None
    per_year: list = field(default_factory=list)

    def summary_line(self) -> str:
        return (
            f"n={self.n_trades} PF={self.pf:.3f} return={self.return_pct:+.1f}% "
            f"DD={self.max_dd_pct:.1f}% RF={self.recovery_factor:.2f} "
            f"Sharpe={self.sharpe_daily:.2f} maxConc={self.max_concurrent} "
            f"overlapHits={self.overlap_hit_count}"
        )


# ---------- helpers ----------

def _pf(pnls: np.ndarray) -> float:
    gp = pnls[pnls > 0].sum()
    gl = float(abs(pnls[pnls <= 0].sum()))
    return float(gp / gl) if gl > 0 else 0.0


def _normalise(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    out = df.copy()
    if "open_time" not in out.columns:
        raise ValueError(f"strategy '{strategy_name}': missing 'open_time' column")
    if "pnl" not in out.columns:
        raise ValueError(f"strategy '{strategy_name}': missing 'pnl' column")
    out["open_dt"] = pd.to_datetime(out["open_time"])
    if "close_time" in out.columns:
        out["close_dt"] = pd.to_datetime(out["close_time"])
    else:
        # assume ~60-min hold if close_time missing; lets us still compute overlap
        out["close_dt"] = out["open_dt"] + pd.Timedelta(minutes=60)
    out["strategy"] = strategy_name
    return out[["open_dt", "close_dt", "pnl", "strategy"]].sort_values("open_dt").reset_index(drop=True)


def _concurrency_series(df_all: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Return the concurrency count at each trade-open instant + a histogram."""
    opens = df_all["open_dt"].to_numpy().astype("datetime64[ns]")
    closes = df_all["close_dt"].to_numpy().astype("datetime64[ns]")
    n = len(opens)
    concurrent_at_open = np.ones(n, dtype=int)
    # O(n^2) is fine for portfolio sizes (<100k trades). For larger, swap to
    # sweepline with heap.
    for i in range(n):
        mask = (opens <= opens[i]) & (closes > opens[i])
        concurrent_at_open[i] = int(mask.sum())
    hist: dict[int, int] = {}
    for v in concurrent_at_open:
        hist[int(v)] = hist.get(int(v), 0) + 1
    return concurrent_at_open, hist


def _daily_pnl(df: pd.DataFrame) -> pd.Series:
    s = df.assign(date=df["open_dt"].dt.floor("D")).groupby("date")["pnl"].sum()
    s.index = pd.DatetimeIndex(s.index)
    return s.sort_index()


def _correlation_matrix(trade_logs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Pairwise daily-PnL correlation per strategy."""
    daily = {}
    for df in trade_logs:
        name = df["strategy"].iloc[0] if "strategy" in df.columns and len(df) else "unnamed"
        daily[name] = _daily_pnl(df)
    if not daily:
        return pd.DataFrame()
    idx = sorted(set().union(*(s.index for s in daily.values())))
    frame = pd.DataFrame({k: v.reindex(idx).fillna(0.0) for k, v in daily.items()})
    return frame.corr()


def _per_year(df: pd.DataFrame) -> list:
    df = df.assign(year=df["open_dt"].dt.year)
    rows = []
    for y, g in df.groupby("year"):
        p = g["pnl"].to_numpy()
        rows.append((int(y), len(p), _pf(p), float(p.sum())))
    return rows


# ---------- core API ----------

def combine_strategies(
    trade_logs: list[pd.DataFrame],
    deposit: float = 1000.0,
    max_concurrent_positions: int = 2,
    overlap_penalty: float = 0.5,
) -> PortfolioMetrics:
    """Combine trade logs, walk chronological equity curve, compute joint metrics.

    Args:
      trade_logs: list of per-strategy DataFrames (need `open_time`, `pnl`; `strategy`
        and `close_time` recommended).
      deposit: starting balance in USD (EBB default $1k).
      max_concurrent_positions: soft cap on simultaneous positions. When exceeded,
        the *excess* position's PnL is multiplied by `overlap_penalty` to reflect
        (a) halved usable margin, (b) broker auto-liquidation risk, and (c) the fact
        that two strategies both firing at the same instant have correlated beta
        that we shouldn't double-count as independent return.
      overlap_penalty: multiplier in (0, 1] applied to overlapped trades beyond the cap.
        0.5 by default = haircut half of the excess PnL (symmetric on wins/losses).

    Returns:
      PortfolioMetrics with combined PF/DD/RF/Sharpe, correlation matrix, daily PnL.
    """
    if not trade_logs:
        raise ValueError("trade_logs is empty")
    if not (0.0 < overlap_penalty <= 1.0):
        raise ValueError("overlap_penalty must be in (0, 1]")

    normalised: list[pd.DataFrame] = []
    for i, df in enumerate(trade_logs):
        strat = df["strategy"].iloc[0] if ("strategy" in df.columns and len(df)) else f"strat{i}"
        normalised.append(_normalise(df, strat))
    combined = pd.concat(normalised, ignore_index=True).sort_values("open_dt").reset_index(drop=True)

    # concurrency + overlap penalty
    concurrent, hist = _concurrency_series(combined)
    combined["concurrent"] = concurrent
    excess_mask = concurrent > max_concurrent_positions
    combined["pnl_adj"] = np.where(
        excess_mask,
        combined["pnl"] * overlap_penalty,
        combined["pnl"],
    )
    overlap_hits = int(excess_mask.sum())

    # equity walk (trade-by-trade)
    pnls = combined["pnl_adj"].to_numpy()
    equity = deposit + np.cumsum(pnls)
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    i_worst = int(np.argmin(dd)) if len(dd) else 0
    max_dd_pct = float(abs(dd.min()) * 100) if len(dd) else 0.0
    max_dd_usd = float(peaks[i_worst] - equity[i_worst]) if len(dd) else 0.0
    ending = float(equity[-1]) if len(equity) else deposit

    total = float(pnls.sum())
    n = int(len(pnls))
    pf = _pf(pnls)
    wr = float((pnls > 0).mean()) if n else 0.0
    span_days = (combined["open_dt"].max() - combined["open_dt"].min()).days if n else 0
    tpm = n / (span_days / 30.4375) if span_days > 0 else 0.0

    # Sharpe-per-trade and daily annualised Sharpe
    sharpe_trade = float(pnls.mean() / pnls.std() * np.sqrt(n)) if n and pnls.std() > 0 else 0.0
    daily = _daily_pnl(combined.assign(pnl=combined["pnl_adj"]))
    if daily.std() > 0 and len(daily) > 1:
        sharpe_daily = float(daily.mean() / daily.std() * np.sqrt(252))
    else:
        sharpe_daily = 0.0

    rf = float(total / max_dd_usd) if max_dd_usd > 0 else float("inf")

    corr = _correlation_matrix(normalised)

    metrics = PortfolioMetrics(
        n_trades=n,
        pf=pf,
        total_pnl=total,
        ending_equity=ending,
        starting_equity=deposit,
        return_pct=100 * (ending / deposit - 1) if deposit else 0.0,
        max_dd_pct=max_dd_pct,
        max_dd_usd=max_dd_usd,
        recovery_factor=rf,
        sharpe_daily=sharpe_daily,
        sharpe_per_trade=sharpe_trade,
        trades_per_month=tpm,
        win_rate=wr,
        max_concurrent=int(concurrent.max()) if len(concurrent) else 0,
        overlap_hit_count=overlap_hits,
        concurrent_hist=hist,
        correlation_matrix=corr,
        daily_pnl=daily,
        equity_curve=pd.Series(equity, index=combined["open_dt"]),
        per_year=_per_year(combined.assign(pnl=combined["pnl_adj"])),
    )
    return metrics


# ---------- I/O helpers ----------

def write_artifacts(metrics: PortfolioMetrics, out_dir: str | Path, tag: str = "portfolio") -> dict:
    """Dump equity CSV + PNG plot + correlation CSV. Returns paths dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    eq = metrics.equity_curve
    if eq is not None:
        p = out / f"{tag}_equity.csv"
        eq.to_csv(p, header=["equity"])
        paths["equity_csv"] = str(p)

    if metrics.correlation_matrix is not None and not metrics.correlation_matrix.empty:
        p = out / f"{tag}_correlation.csv"
        metrics.correlation_matrix.to_csv(p)
        paths["correlation_csv"] = str(p)

    if metrics.daily_pnl is not None:
        p = out / f"{tag}_daily_pnl.csv"
        metrics.daily_pnl.to_csv(p, header=["pnl"])
        paths["daily_csv"] = str(p)

    if _HAVE_MPL and eq is not None and len(eq):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax1.plot(eq.index, eq.values, lw=1.0)
        ax1.axhline(metrics.starting_equity, color="grey", ls="--", lw=0.7)
        ax1.set_ylabel("Equity ($)")
        ax1.set_title(
            f"{tag} | PF={metrics.pf:.2f} DD={metrics.max_dd_pct:.1f}% "
            f"RF={metrics.recovery_factor:.2f} Sharpe={metrics.sharpe_daily:.2f}"
        )
        ax1.grid(alpha=0.3)
        peaks = np.maximum.accumulate(eq.values)
        ddpct = 100 * (eq.values - peaks) / peaks
        ax2.fill_between(eq.index, ddpct, 0, color="red", alpha=0.4)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(alpha=0.3)
        fig.autofmt_xdate()
        p = out / f"{tag}_equity.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths["equity_png"] = str(p)

    return paths


# ---------- self-test ----------

def _synth_log(seed: int, n: int, start: str = "2024-01-02", bias: float = 0.4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    minutes = np.cumsum(rng.integers(30, 480, size=n))
    opens = pd.Timestamp(start) + pd.to_timedelta(minutes, unit="m")
    closes = opens + pd.to_timedelta(rng.integers(20, 90, size=n), unit="m")
    pnls = rng.normal(loc=bias, scale=5.0, size=n)
    return pd.DataFrame(
        {"open_time": opens, "close_time": closes, "pnl": pnls, "strategy": f"strat{seed}"}
    )


if __name__ == "__main__":
    logs = [_synth_log(1, 400, bias=0.35), _synth_log(2, 350, bias=0.30), _synth_log(3, 420, bias=0.32)]
    m = combine_strategies(logs, deposit=1000.0, max_concurrent_positions=2)
    print("[smoke]", m.summary_line())
    print("[corr]\n", m.correlation_matrix.round(3))
    print("[concurrent hist]", m.concurrent_hist)
    out = Path(__file__).resolve().parent / "_smoke_out"
    p = write_artifacts(m, out, tag="smoke")
    print("[artifacts]", p)
