"""
DXY Synthesizer — compute the ICE U.S. Dollar Index from a six-currency basket.

Formula (ICE U.S. Dollar Index, 1973 base):
    DXY = 50.14348112 * EURUSD^-0.576 * USDJPY^0.136 * GBPUSD^-0.119
                      * USDCAD^0.091  * USDSEK^0.042 * USDCHF^0.036

Negative weight = USD is the QUOTE (EUR, GBP); positive = USD is the BASE.
See: docs/design_dxy_cross_asset.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

DXY_CONSTANT = 50.14348112
DXY_WEIGHTS: Dict[str, float] = {
    "EURUSD": -0.576, "USDJPY": 0.136, "GBPUSD": -0.119,
    "USDCAD":  0.091, "USDSEK": 0.042, "USDCHF":  0.036,
}
REQUIRED_LEGS = tuple(DXY_WEIGHTS.keys())


@dataclass(frozen=True)
class SynthResult:
    """DXY series + per-bar diagnostics."""
    dxy: pd.Series
    staleness_flags: pd.Series
    missing_bars: int
    source_files: Dict[str, str]


# --- loaders -----------------------------------------------------------------

def load_leg_csv(path: str | Path, symbol: str,
                 time_col: str = "time", close_col: str = "close") -> pd.Series:
    """Load an M5 CSV (MT5 export schema), return a UTC-indexed close series."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{symbol} leg CSV not found: {path}")
    df = pd.read_csv(path)
    for col in (time_col, close_col):
        if col not in df.columns:
            raise ValueError(
                f"{symbol}: column '{col}' missing (have {list(df.columns)[:8]})"
            )
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{symbol}: {int(ts.isna().sum())} unparsable timestamps")
    ser = pd.Series(df[close_col].astype(float).values, index=ts, name=symbol)
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    if (ser <= 0).any():
        raise ValueError(f"{symbol}: {int((ser <= 0).sum())} non-positive closes")
    LOG.debug("loaded %s: %d bars, %s..%s",
              symbol, len(ser), ser.index.min(), ser.index.max())
    return ser


# --- alignment and synthesis -------------------------------------------------

def align_legs(legs: Mapping[str, pd.Series],
               ffill_limit: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """Align 6 legs on EUR/USD's timestamp index.

    Forward-fills up to `ffill_limit` bars for thin-liquidity gaps (USDSEK).
    Rows still missing any leg after ffill are dropped.
    Returns (aligned_df, staleness_flag_series).
    """
    missing = [s for s in REQUIRED_LEGS if s not in legs]
    if missing:
        raise ValueError(f"missing required legs: {missing}")
    eur_index = legs["EURUSD"].index
    df = pd.DataFrame(index=eur_index)
    miss_cols = []
    for sym in REQUIRED_LEGS:
        ser = legs[sym].reindex(eur_index)
        was_missing = ser.isna()
        ser = ser.ffill(limit=ffill_limit)
        df[sym] = ser
        col = f"__missing_{sym}"
        df[col] = was_missing & ser.notna()  # True if this bar was ffilled
        miss_cols.append(col)
    staleness = df[miss_cols].any(axis=1)
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_LEGS))
    staleness = staleness.loc[df.index]
    dropped = before - len(df)
    if dropped:
        LOG.warning("dropped %d/%d bars due to unalignable legs", dropped, before)
    return df[list(REQUIRED_LEGS)], staleness


def compute_dxy(aligned: pd.DataFrame,
                weights: Mapping[str, float] = DXY_WEIGHTS,
                constant: float = DXY_CONSTANT) -> pd.Series:
    """Apply the ICE formula in log-space for numerical stability."""
    missing_weights = set(weights) - set(aligned.columns)
    if missing_weights:
        raise ValueError(f"weights reference missing cols: {missing_weights}")
    log_dxy = np.log(constant) + sum(
        weights[sym] * np.log(aligned[sym].values) for sym in weights
    )
    return pd.Series(np.exp(log_dxy), index=aligned.index, name="DXY")


def synthesize(legs: Mapping[str, pd.Series],
               weights: Mapping[str, float] = DXY_WEIGHTS,
               constant: float = DXY_CONSTANT,
               ffill_limit: int = 3) -> SynthResult:
    """Core entry point: 6 close-price Series -> SynthResult."""
    aligned, staleness = align_legs(legs, ffill_limit=ffill_limit)
    dxy = compute_dxy(aligned, weights=weights, constant=constant)
    return SynthResult(
        dxy=dxy,
        staleness_flags=staleness,
        missing_bars=int(len(legs["EURUSD"]) - len(dxy)),
        source_files={k: "<in-memory>" for k in REQUIRED_LEGS},
    )


def synthesize_from_csvs(csv_paths: Mapping[str, str | Path],
                         time_col: str = "time", close_col: str = "close",
                         weights: Mapping[str, float] = DXY_WEIGHTS,
                         constant: float = DXY_CONSTANT,
                         ffill_limit: int = 3) -> SynthResult:
    """Convenience: load 6 CSVs from disk and synthesize."""
    legs: Dict[str, pd.Series] = {}
    paths_seen: Dict[str, str] = {}
    for sym in REQUIRED_LEGS:
        if sym not in csv_paths:
            raise KeyError(f"csv_paths missing required symbol: {sym}")
        p = Path(csv_paths[sym])
        legs[sym] = load_leg_csv(p, sym, time_col=time_col, close_col=close_col)
        paths_seen[sym] = str(p)
    res = synthesize(legs, weights=weights, constant=constant,
                     ffill_limit=ffill_limit)
    return SynthResult(dxy=res.dxy, staleness_flags=res.staleness_flags,
                       missing_bars=res.missing_bars, source_files=paths_seen)


# --- divergence features (for downstream ML / rule-based gates) --------------

def divergence_features(eurusd_close: pd.Series, dxy: pd.Series,
                        window: int = 60) -> pd.DataFrame:
    """Rolling-regression divergence features on M5 bars.

    Columns: r_eur, r_dxy (log-returns); beta, alpha (rolling N=60 regression
    r_eur ~ alpha + beta*r_dxy); pred_r_eur, resid, resid_sigma, resid_z;
    dxy_implied_eur_sign = -sign(r_dxy) (EUR should move opposite to DXY).
    """
    eur = eurusd_close.reindex(dxy.index).dropna()
    dx = dxy.reindex(eur.index)
    r_eur = np.log(eur / eur.shift(1))
    r_dxy = np.log(dx / dx.shift(1))
    cov = r_eur.rolling(window).cov(r_dxy)
    var = r_dxy.rolling(window).var()
    beta = cov / var.replace(0.0, np.nan)
    alpha = r_eur.rolling(window).mean() - beta * r_dxy.rolling(window).mean()
    pred = alpha + beta * r_dxy
    resid = r_eur - pred
    resid_sigma = resid.rolling(window).std()
    resid_z = resid / resid_sigma.replace(0.0, np.nan)
    return pd.DataFrame({
        "r_eur": r_eur, "r_dxy": r_dxy,
        "beta": beta, "alpha": alpha, "pred_r_eur": pred,
        "resid": resid, "resid_sigma": resid_sigma, "resid_z": resid_z,
        "dxy_implied_eur_sign": -np.sign(r_dxy),
    })


# --- sanity checks -----------------------------------------------------------

def validate_against_reference(dxy: pd.Series, reference: pd.Series,
                               max_bp_mean: float = 5.0,
                               max_bp_p99: float = 15.0) -> Dict[str, float]:
    """Compare synthetic DXY to a reference series (e.g. ICE DX futures).

    Returns dict of error stats in basis points; logs a warning if gates
    (mean |err| <= max_bp_mean, p99 |err| <= max_bp_p99) are exceeded.
    """
    ref = reference.reindex(dxy.index).dropna()
    d = dxy.reindex(ref.index)
    if d.empty:
        raise ValueError("no overlapping timestamps between synthetic and reference")
    err_bp = (d - ref) / ref * 10_000.0
    abs_err = err_bp.abs()
    stats = {
        "n": float(len(err_bp)),
        "mean_abs_bp": float(abs_err.mean()),
        "p50_abs_bp":  float(abs_err.quantile(0.50)),
        "p95_abs_bp":  float(abs_err.quantile(0.95)),
        "p99_abs_bp":  float(abs_err.quantile(0.99)),
        "max_abs_bp":  float(abs_err.max()),
        "corr":        float(d.corr(ref)),
    }
    if stats["mean_abs_bp"] > max_bp_mean:
        LOG.warning("synthetic DXY mean |err| = %.2f bp > %.2f bp gate",
                    stats["mean_abs_bp"], max_bp_mean)
    if stats["p99_abs_bp"] > max_bp_p99:
        LOG.warning("synthetic DXY p99 |err| = %.2f bp > %.2f bp gate",
                    stats["p99_abs_bp"], max_bp_p99)
    return stats


# --- CLI ---------------------------------------------------------------------

def _main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Synthesize DXY from 6 currency CSVs.")
    for sym in REQUIRED_LEGS:
        p.add_argument(f"--{sym.lower()}", required=True,
                       help=f"path to {sym} M5 CSV")
    p.add_argument("--out", required=True, help="output CSV path for DXY series")
    p.add_argument("--time-col", default="time")
    p.add_argument("--close-col", default="close")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    csv_paths = {sym: getattr(args, sym.lower()) for sym in REQUIRED_LEGS}
    res = synthesize_from_csvs(
        csv_paths, time_col=args.time_col, close_col=args.close_col
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "time": res.dxy.index,
        "dxy":  res.dxy.values,
        "stale": res.staleness_flags.values.astype(int),
    }).to_csv(out, index=False)
    LOG.info("wrote %d bars to %s (dropped %d bars)",
             len(res.dxy), out, res.missing_bars)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
