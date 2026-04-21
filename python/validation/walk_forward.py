"""Walk-forward (rolling IS/OOS) validation harness.

Implements Gate-4 of the WG3 validation pipeline, symbol-agnostic.

Protocol (defaults match the GoldBigBrain fg+rsisl gold script)
---------------------------------------------------------------
- Train window: 3 years (36 months)
- Test window : 6 months
- Slide step  : 6 months
- Minimum 30 test trades required to score the fold; otherwise slide

Usage patterns
--------------
1. **Frozen trade log** (strategy does NOT re-tune per fold):
     pass a pd.DataFrame with columns ['open_time', 'pnl'] as `full_period_data`
     and `strategy_runner=None`. The harness slices by date.

2. **Full retrain** (strategy DOES re-tune per fold):
     pass raw market data as `full_period_data` (any object the runner understands)
     and a callable `strategy_runner(data_slice, params)` that returns a trade
     log DataFrame with at least ['open_time', 'pnl'].

PASS criteria
-------------
- >= 5/6 folds with fold-PF >= 1.0
- No fold with DD worse than 10% of starting equity

Returns
-------
pandas DataFrame with one row per fold:
    train_start, train_end, test_end, train_n, test_n,
    train_pf, test_pf, train_pnl, test_pnl, test_dd, verdict.
An aggregate summary line is logged at INFO level.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# -- public types --------------------------------------------------------------

class StrategyRunner(Protocol):
    """Callable: (data_slice, params) -> trade_log DataFrame."""
    def __call__(self, data: Any, params: Optional[dict]) -> pd.DataFrame: ...


@dataclass
class WalkForwardVerdict:
    folds_total: int
    folds_pass: int              # test_pf >= 1.0
    folds_dd_pass: int           # test_dd <= 10%
    mean_test_pf: float
    median_test_pf: float
    total_test_pnl: float
    worst_test_dd: float
    passes: bool                 # >= 5/6 folds PF>=1 AND no DD > 10%


# -- helpers -------------------------------------------------------------------

def _pf(pnls: np.ndarray) -> float:
    gp = pnls[pnls > 0].sum()
    gl = -pnls[pnls < 0].sum()
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)


def _max_dd_pct(pnls: np.ndarray, starting: float = 1000.0) -> float:
    """Percent max drawdown on cumulative PnL equity curve (starting=$1000)."""
    if pnls.size == 0:
        return 0.0
    eq = starting + np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(abs(dd.min()) * 100.0)


def _fold_row(
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_end: pd.Timestamp,
    train_trades: pd.DataFrame,
    test_trades: pd.DataFrame,
    pnl_col: str,
    fold_pf_gate: float,
    fold_dd_gate: float,
) -> dict:
    train_pnl = train_trades[pnl_col].to_numpy(dtype=float)
    test_pnl = test_trades[pnl_col].to_numpy(dtype=float)
    train_pf = _pf(train_pnl) if train_pnl.size else 0.0
    test_pf = _pf(test_pnl) if test_pnl.size else 0.0
    test_dd = _max_dd_pct(test_pnl)
    verdict = (
        "PASS" if test_pf >= fold_pf_gate and test_dd <= fold_dd_gate
        else ("MARGINAL" if test_pf >= 1.0 and test_dd <= fold_dd_gate else "FAIL")
    )
    return {
        "train_start": train_start.date(),
        "train_end": train_end.date(),
        "test_end": test_end.date(),
        "train_n": int(train_pnl.size),
        "test_n": int(test_pnl.size),
        "train_pf": round(train_pf, 4),
        "test_pf": round(test_pf, 4),
        "train_pnl": round(float(train_pnl.sum()), 2),
        "test_pnl": round(float(test_pnl.sum()), 2),
        "test_dd": round(test_dd, 3),
        "verdict": verdict,
    }


# -- main API ------------------------------------------------------------------

def walk_forward(
    strategy_runner: Optional[StrategyRunner],
    full_period_data: Union[pd.DataFrame, Any],
    *,
    train_years: int = 3,
    test_months: int = 6,
    slide_months: int = 6,
    min_test_trades: int = 30,
    time_col: str = "open_time",
    pnl_col: str = "pnl",
    params: Optional[dict] = None,
    fold_pf_gate: float = 1.0,
    fold_dd_gate_pct: float = 10.0,
    starting_equity: float = 1000.0,
) -> pd.DataFrame:
    """Rolling walk-forward validation.

    Two modes:

    1. `strategy_runner is None`  +  `full_period_data` is a pd.DataFrame
       with `time_col` and `pnl_col` columns (frozen trade log). The harness
       slices by date into train/test and scores each fold on existing PnL.

    2. `strategy_runner` is callable and `full_period_data` is raw data.
       For each fold, we call the runner on the train slice (optional
       in-sample re-fit) AND on the test slice, using the same runner so
       results are comparable. If your runner needs to carry state between
       calls, use a closure.

    Returns DataFrame with columns:
        train_start, train_end, test_end, train_n, test_n,
        train_pf, test_pf, train_pnl, test_pnl, test_dd, verdict.
    A `WalkForwardVerdict` is attached at `df.attrs['verdict']`.
    """
    train_months = train_years * 12
    results: list[dict] = []

    # -- mode 1: frozen trade log -----------------------------------------
    if strategy_runner is None:
        if not isinstance(full_period_data, pd.DataFrame):
            raise TypeError(
                "strategy_runner=None requires full_period_data to be a DataFrame"
            )
        df = full_period_data.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
        t_start = df[time_col].min().normalize()
        t_end = df[time_col].max().normalize()

        cursor = t_start
        while True:
            train_end = cursor + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)
            if test_end > t_end + pd.DateOffset(days=1):
                break
            train = df[(df[time_col] >= cursor) & (df[time_col] < train_end)]
            test = df[(df[time_col] >= train_end) & (df[time_col] < test_end)]
            if len(test) < min_test_trades:
                cursor += pd.DateOffset(months=slide_months)
                continue
            results.append(_fold_row(
                cursor, train_end, test_end, train, test,
                pnl_col, fold_pf_gate, fold_dd_gate_pct,
            ))
            cursor += pd.DateOffset(months=slide_months)

    # -- mode 2: strategy_runner provided ---------------------------------
    else:
        if not hasattr(full_period_data, "slice_by_date"):
            raise TypeError(
                "strategy_runner mode requires full_period_data with a "
                "slice_by_date(start, end)->data method, OR adapt your data "
                "to expose it. Pass frozen trade logs for mode 1."
            )
        t_start = pd.Timestamp(full_period_data.t_start).normalize()
        t_end = pd.Timestamp(full_period_data.t_end).normalize()
        cursor = t_start
        while True:
            train_end = cursor + pd.DateOffset(months=train_months)
            test_end = train_end + pd.DateOffset(months=test_months)
            if test_end > t_end + pd.DateOffset(days=1):
                break
            train_data = full_period_data.slice_by_date(cursor, train_end)
            test_data = full_period_data.slice_by_date(train_end, test_end)
            train_trades = strategy_runner(train_data, params)
            test_trades = strategy_runner(test_data, params)
            if len(test_trades) < min_test_trades:
                cursor += pd.DateOffset(months=slide_months)
                continue
            results.append(_fold_row(
                cursor, train_end, test_end, train_trades, test_trades,
                pnl_col, fold_pf_gate, fold_dd_gate_pct,
            ))
            cursor += pd.DateOffset(months=slide_months)

    out = pd.DataFrame(results)
    if out.empty:
        log.warning("walk_forward produced 0 folds (history too short?)")
        return out

    # verdict
    n_pf_pass = int((out["test_pf"] >= fold_pf_gate).sum())
    n_dd_pass = int((out["test_dd"] <= fold_dd_gate_pct).sum())
    total_folds = len(out)
    # "5/6 folds" semantics — we scale the ratio 5/6 = 0.833 to actual fold count
    pf_ratio = n_pf_pass / total_folds
    passes = pf_ratio >= (5.0 / 6.0) and n_dd_pass == total_folds

    verdict = WalkForwardVerdict(
        folds_total=total_folds,
        folds_pass=n_pf_pass,
        folds_dd_pass=n_dd_pass,
        mean_test_pf=float(out["test_pf"].mean()),
        median_test_pf=float(out["test_pf"].median()),
        total_test_pnl=float(out["test_pnl"].sum()),
        worst_test_dd=float(out["test_dd"].max()),
        passes=passes,
    )
    out.attrs["verdict"] = verdict
    return out


def print_walk_forward_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("(no folds)")
        return
    print(f"{'train_start':<12} {'train_end':<12} {'test_end':<12} "
          f"{'trN':>4} {'tPF':>5} {'tTot':>9}  "
          f"{'teN':>4} {'tePF':>5} {'teTot':>9} {'teDD':>6}  verdict")
    for _, r in df.iterrows():
        print(f"{str(r['train_start']):<12} {str(r['train_end']):<12} "
              f"{str(r['test_end']):<12} "
              f"{r['train_n']:>4} {r['train_pf']:>5.2f} ${r['train_pnl']:>7.2f}  "
              f"{r['test_n']:>4} {r['test_pf']:>5.2f} ${r['test_pnl']:>7.2f} "
              f"{r['test_dd']:>5.2f}%  {r['verdict']}")
    v: WalkForwardVerdict = df.attrs.get("verdict")
    if v is not None:
        tag = "PASS" if v.passes else "FAIL"
        print(
            f"\nFolds: {v.folds_pass}/{v.folds_total} PF>=1.0, "
            f"{v.folds_dd_pass}/{v.folds_total} DD<=10%, "
            f"mean tePF={v.mean_test_pf:.3f}, total tePnL=${v.total_test_pnl:.2f}, "
            f"worst DD={v.worst_test_dd:.2f}%  ==>  {tag}"
        )


# -- CLI -----------------------------------------------------------------------

def _main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward validation on a trade log")
    ap.add_argument("--trades", type=Path, required=True)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--slide-months", type=int, default=6)
    ap.add_argument("--time-col", default="open_time")
    ap.add_argument("--pnl-col", default="pnl")
    args = ap.parse_args()

    df = pd.read_csv(args.trades)
    out = walk_forward(
        strategy_runner=None,
        full_period_data=df,
        train_years=args.train_years,
        test_months=args.test_months,
        slide_months=args.slide_months,
        time_col=args.time_col,
        pnl_col=args.pnl_col,
    )
    print_walk_forward_report(out)
    v = out.attrs.get("verdict")
    return 0 if (v is not None and v.passes) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(_main())
