"""
EuroBigBrain — M5 parquet data quality auditor.

Bad data silently breaks backtests. This module checks every
parquet produced by fetch_history.py for the common rot:

    - bar count vs. expected FX M5 bar count over the span
    - gap detection  (> 5 min between bars inside market hours)
    - holiday weekend handling (Saturday bars present = broker fault)
    - DST jump anomalies (springs: 55 min gap, falls: 5 min overlap)
    - OHLC sanity (H>=max(O,C,L), L<=min(O,C,H))
    - duplicate timestamps
    - zero-volume bars (often broker-fill placeholder, not real liquidity)

CLI:
    python quality_audit.py data/EURUSD_M5_2020_2026.parquet
    python quality_audit.py data/*.parquet               # multi-symbol report

Outputs:
    data/{symbol}_audit_findings.csv   — one row per anomaly
    data/{symbol}_audit_summary.csv    — one row, headline stats
    stdout                              — concise human summary
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any

import pandas as pd


# FX trades Sun 22:00 UTC -> Fri 22:00 UTC (broker-dependent).
# Allow gap just over 1 weekend (~ 65 hours) without flagging.
MAX_INTRABAR_GAP_MIN = 5       # M5 nominal
FLAG_GAP_MIN = 10              # anything >10m inside a session is a real gap
WEEKEND_GAP_MIN = 60 * 65      # ~65h, covers Fri close -> Sun open
DST_SPRING_MIN = 55            # spring-forward: normal 5 + lost 50
DST_FALL_MIN = -5              # fall-back: one 5m bar repeated


def _expected_m5_bars(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Rough expected M5 bar count for a 24/5 FX symbol.

    ~ trading-day fraction * minutes-per-day / 5.
    Trading-week fraction = 5/7 of total days; daily bars ~ 288 M5.
    """
    span_days = (end - start).total_seconds() / 86400.0
    trading_days = span_days * (5.0 / 7.0)
    return int(trading_days * 288)


def _check_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where OHLC identity is violated."""
    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    out = df.loc[bad, ["time", "open", "high", "low", "close"]].copy()
    out["issue"] = "ohlc_violation"
    return out


def _check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup = df[df["time"].duplicated(keep=False)].copy()
    if dup.empty:
        return dup
    dup = dup[["time", "open", "high", "low", "close"]]
    dup["issue"] = "duplicate_timestamp"
    return dup


def _check_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    # tick_volume == 0 is suspicious; real_volume often 0 for FX (no central exchange).
    bad = df["tick_volume"] == 0
    out = df.loc[bad, ["time", "tick_volume"]].copy()
    out["issue"] = "zero_tick_volume"
    return out


def _check_saturday(df: pd.DataFrame) -> pd.DataFrame:
    # dayofweek: Monday=0 ... Saturday=5, Sunday=6.
    sat = df[df["time"].dt.dayofweek == 5].copy()
    if sat.empty:
        return sat
    sat = sat[["time", "open", "close"]]
    sat["issue"] = "saturday_bar"
    return sat


def _check_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Find inter-bar gaps that are neither the expected 5-min step
    nor a plausible weekend / DST boundary."""
    d = df[["time"]].copy()
    d["delta_min"] = d["time"].diff().dt.total_seconds().div(60.0)
    # Classify each delta
    issues: list[dict[str, Any]] = []
    for row in d.itertuples(index=False):
        dm = row.delta_min
        if pd.isna(dm) or dm == MAX_INTRABAR_GAP_MIN:
            continue
        # Weekend gap: Friday close to Sunday open, roughly 47-65h
        if 47 * 60 <= dm <= WEEKEND_GAP_MIN:
            continue
        # DST spring-forward
        if dm == DST_SPRING_MIN:
            issues.append({"time": row.time, "delta_min": dm,
                           "issue": "dst_spring_ok"})
            continue
        # DST fall-back shows as either extra bar (handled by dup) or
        # zero / negative delta; skip unless negative
        if dm < MAX_INTRABAR_GAP_MIN:
            issues.append({"time": row.time, "delta_min": dm,
                           "issue": "backward_or_zero_delta"})
            continue
        if dm > FLAG_GAP_MIN:
            # Big gap mid-session -> likely data loss
            issues.append({"time": row.time, "delta_min": dm,
                           "issue": "intraday_gap"})
    return pd.DataFrame(issues)


def audit_dataset(parquet_path: str) -> dict[str, Any]:
    """Run all checks on a single parquet; return summary dict."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)
    if "time" not in df.columns:
        raise ValueError(f"{parquet_path} missing 'time' column")
    df = df.sort_values("time").reset_index(drop=True)

    symbol = os.path.basename(parquet_path).split("_")[0]
    out_dir = os.path.dirname(parquet_path)

    findings = pd.concat(
        [
            _check_ohlc(df),
            _check_duplicates(df),
            _check_zero_volume(df),
            _check_saturday(df),
            _check_gaps(df),
        ],
        ignore_index=True,
    )

    start, end = df["time"].iloc[0], df["time"].iloc[-1]
    expected = _expected_m5_bars(start, end)
    completeness = (len(df) / expected) if expected else 0.0

    summary: dict[str, Any] = {
        "symbol": symbol,
        "path": parquet_path,
        "bars": len(df),
        "span_start": start,
        "span_end": end,
        "span_days": round((end - start).total_seconds() / 86400.0, 1),
        "expected_bars": expected,
        "completeness": round(completeness, 4),
        "ohlc_violations": int((findings["issue"] == "ohlc_violation").sum()),
        "duplicates": int((findings["issue"] == "duplicate_timestamp").sum()),
        "zero_vol_bars": int((findings["issue"] == "zero_tick_volume").sum()),
        "saturday_bars": int((findings["issue"] == "saturday_bar").sum()),
        "intraday_gaps": int((findings["issue"] == "intraday_gap").sum()),
        "dst_spring_events": int((findings["issue"] == "dst_spring_ok").sum()),
        "backward_deltas": int((findings["issue"] == "backward_or_zero_delta").sum()),
        "total_issues": len(findings),
    }

    findings_path = os.path.join(out_dir, f"{symbol}_audit_findings.csv")
    summary_path = os.path.join(out_dir, f"{symbol}_audit_summary.csv")
    findings.to_csv(findings_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    summary["findings_csv"] = findings_path
    summary["summary_csv"] = summary_path
    return summary


def _print(summary: dict[str, Any]) -> None:
    s = summary
    status = "OK" if s["completeness"] >= 0.95 and s["ohlc_violations"] == 0 else "REVIEW"
    print(f"[{s['symbol']}] {status}  bars={s['bars']:,}  "
          f"completeness={s['completeness']:.1%}  "
          f"issues={s['total_issues']:,} "
          f"(ohlc={s['ohlc_violations']} dup={s['duplicates']} "
          f"zero_vol={s['zero_vol_bars']} sat={s['saturday_bars']} "
          f"gaps={s['intraday_gaps']} dst={s['dst_spring_events']})")
    print(f"  span: {s['span_start']} .. {s['span_end']}  "
          f"({s['span_days']} days, expected ~{s['expected_bars']:,} bars)")


def main(paths: list[str]) -> int:
    expanded: list[str] = []
    for p in paths:
        expanded.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not expanded:
        print("no parquet files found", file=sys.stderr)
        return 2
    rc = 0
    for path in expanded:
        try:
            summary = audit_dataset(path)
            _print(summary)
            if summary["completeness"] < 0.90 or summary["ohlc_violations"] > 0:
                rc = 1
        except Exception as e:   # noqa: BLE001
            print(f"[{path}] AUDIT FAILED: {e}", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Audit M5 parquet data quality")
    ap.add_argument("paths", nargs="+", help="parquet file(s) or glob(s)")
    args = ap.parse_args()
    sys.exit(main(args.paths))
