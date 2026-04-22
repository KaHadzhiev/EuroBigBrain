#!/usr/bin/env python3
"""Python sim of EBB Triple-Barrier on 6yr OOF predictions, threshold sweep 0.30-0.50.

Inputs:
  data/eurusd_oof_probs_6yr.parquet    (OOF probs from train_eurusd_lightgbm_6yr.py)
  data/eurusd_m5_2020_2026.parquet     (M5 bars for OHLC)

Sim model (matches EBB_TripleBarrier.mq5):
  - On each bar with prob >= threshold, simulate BUY at next-bar OPEN (matches GBB convention)
  - SL = -SL_ATR_Mult * ATR14 from entry, TP = +TP_ATR_Mult * ATR14
  - Time-stop at MaxHoldBars
  - Spread cost: VANTAGE_SPREAD_PIPS pips per round-turn (configurable; default 1.0 pip
    = Dukascopy ECN ~0.7 + retail markup 0.3)
  - Slippage: SLIPPAGE_PIPS per fill (entry+exit) — small for EURUSD market orders

Output:
  results/sim_eurusd_6yr_threshold_sweep.csv   (per-threshold PF/trades/DD/PnL)
  runs/sim_eurusd_6yr.json                     (full per-threshold metrics)
"""
import argparse
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OOF_PATH = REPO / "data" / "eurusd_oof_probs_6yr.parquet"
M5_PATH = REPO / "data" / "eurusd_m5_2020_2026.parquet"
OUT_CSV = REPO / "results" / "sim_eurusd_6yr_threshold_sweep.csv"
OUT_JSON = REPO / "runs" / "sim_eurusd_6yr.json"

# Strategy params (lock to deployed EA values)
H_BARS = 10
SL_MULT = 0.7
TP_MULT = 2.0

# Cost model — EURUSD on Vantage Standard STP (no commission, spread only)
EUR_PIP = 0.0001
# Vantage retail spread > Dukascopy ECN spread. Vantage Standard typical ~1.0 pip on EURUSD
# during London/NY (Dukascopy ECN ~0.2-0.5 pip). Markup over ECN ~0.5-0.8 pip.
VANTAGE_SPREAD_PIPS = 1.3  # corrected from 1.0 after pulling actual EUR Vantage spread (median 13 pts = 1.3 pip)
# Slippage on market entry+exit (pips)
SLIPPAGE_PIPS = 0.2

THRESHOLDS = [0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]


def atr14_series(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Simple rolling-mean of TR over 14, matches build_eurusd_features.py atr()."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        a = high[i] - low[i]
        b = abs(high[i] - close[i - 1])
        c = abs(low[i] - close[i - 1])
        tr[i] = max(a, b, c)
    out = np.full(n, np.nan)
    s = 0.0
    for i in range(n):
        s += tr[i]
        if i >= 14:
            s -= tr[i - 14]
            out[i] = s / 14.0
        elif i == 13:
            out[i] = s / 14.0
    return out


def simulate(probs_at_bar: np.ndarray, open_p: np.ndarray, high_p: np.ndarray,
             low_p: np.ndarray, atr: np.ndarray, threshold: float) -> dict:
    """Run a long-only Triple-Barrier sim. Returns metrics dict.

    Cost model (per round-turn): spread + 2 * slippage, in price terms.
    """
    n = len(probs_at_bar)
    entry_price_arr = []
    pnl_pips = []
    outcomes = []  # 'TP' / 'SL' / 'TIMEOUT'

    cost_pips = VANTAGE_SPREAD_PIPS + 2.0 * SLIPPAGE_PIPS
    cost_price = cost_pips * EUR_PIP

    in_pos_until = -1
    for i in range(n - 1 - H_BARS):
        if i <= in_pos_until:
            continue
        if not np.isfinite(probs_at_bar[i]) or probs_at_bar[i] < threshold:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        # Entry at next bar's OPEN (matches GBB / EBB EA semantics)
        entry_bar = i + 1
        if entry_bar >= n:
            continue
        entry = open_p[entry_bar]
        tp_px = entry + TP_MULT * a
        sl_px = entry - SL_MULT * a

        # Walk forward up to H_BARS bars (inclusive of entry bar high/low)
        outcome = "TIMEOUT"
        exit_price = None
        end_bar = min(entry_bar + H_BARS, n - 1)
        for j in range(entry_bar, end_bar + 1):
            hit_tp = high_p[j] >= tp_px
            hit_sl = low_p[j]  <= sl_px
            if hit_tp and hit_sl:
                # Both touched same bar -> conservative: SL wins (matches MT5 stop priority)
                outcome = "SL"
                exit_price = sl_px
                break
            if hit_sl:
                outcome = "SL"
                exit_price = sl_px
                break
            if hit_tp:
                outcome = "TP"
                exit_price = tp_px
                break
        if exit_price is None:
            # Timeout -> exit at end_bar close (use open of next bar as proxy = close)
            exit_price = open_p[end_bar] if end_bar < n - 1 else open_p[end_bar]

        gross = exit_price - entry
        net = gross - cost_price
        pnl_pips.append(net / EUR_PIP)
        entry_price_arr.append(entry)
        outcomes.append(outcome)
        in_pos_until = end_bar  # block re-entry while position held

    if not pnl_pips:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "pnl_pips": 0.0,
                "max_dd_pips": 0.0, "n_tp": 0, "n_sl": 0, "n_timeout": 0}

    pnl = np.array(pnl_pips)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.size and losses.sum() < 0 else float("inf")
    wr = float((pnl > 0).mean())

    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum)
    max_dd = float(dd.max()) if dd.size else 0.0

    n_tp = sum(1 for o in outcomes if o == "TP")
    n_sl = sum(1 for o in outcomes if o == "SL")
    n_to = sum(1 for o in outcomes if o == "TIMEOUT")

    return {
        "trades": int(len(pnl)),
        "pf": round(float(pf), 4) if np.isfinite(pf) else 999.0,
        "wr": round(wr, 4),
        "pnl_pips": round(float(pnl.sum()), 2),
        "max_dd_pips": round(max_dd, 2),
        "n_tp": n_tp, "n_sl": n_sl, "n_timeout": n_to,
    }


def main() -> int:
    t0 = time.time()
    if not OOF_PATH.exists():
        print(f"FATAL: {OOF_PATH} not found - run train_eurusd_lightgbm_6yr.py first", file=sys.stderr)
        return 1
    if not M5_PATH.exists():
        print(f"FATAL: {M5_PATH} not found", file=sys.stderr)
        return 1

    print(f"Loading {OOF_PATH} + {M5_PATH} ...")
    oof = pd.read_parquet(OOF_PATH)
    oof["time"] = pd.to_datetime(oof["time"])
    oof = oof.set_index("time").sort_index()

    bars = pd.read_parquet(M5_PATH)
    bars["time"] = pd.to_datetime(bars["time"])
    bars = bars.set_index("time").sort_index()

    # Restrict bars to OOF span
    bars = bars.loc[oof.index.min():oof.index.max()]
    common = bars.index.intersection(oof.index)
    bars = bars.loc[common]
    oof = oof.loc[common]
    print(f"  bars in OOF span: {len(bars):,}  span={bars.index.min()} -> {bars.index.max()}")

    # Compute ATR14 (matches EBB feature builder)
    atr = atr14_series(bars["high"].values, bars["low"].values, bars["close"].values)

    open_p = bars["open"].values
    high_p = bars["high"].values
    low_p = bars["low"].values
    probs = oof["prob"].values

    rows = []
    print(f"\n{'thr':>6} {'trades':>7} {'pf':>7} {'wr':>6} {'pnl_pips':>10} {'maxDD_pips':>11} "
          f"{'TP':>5} {'SL':>5} {'TO':>5}")
    for thr in THRESHOLDS:
        m = simulate(probs, open_p, high_p, low_p, atr, thr)
        m["threshold"] = thr
        rows.append(m)
        print(f"{thr:>6.2f} {m['trades']:>7} {m['pf']:>7.3f} {m['wr']:>6.3f} "
              f"{m['pnl_pips']:>10.1f} {m['max_dd_pips']:>11.1f} "
              f"{m['n_tp']:>5} {m['n_sl']:>5} {m['n_timeout']:>5}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "config": {"h": H_BARS, "sl": SL_MULT, "tp": TP_MULT,
                   "spread_pips": VANTAGE_SPREAD_PIPS, "slippage_pips": SLIPPAGE_PIPS,
                   "thresholds": THRESHOLDS},
        "data": {"oof_rows": int(len(oof)), "span_start": str(oof.index.min()),
                 "span_end": str(oof.index.max())},
        "results": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"Saved {OUT_JSON}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    return 0


def smoke_test() -> int:
    """Run sim on synthetic 1000-bar EUR series with mocked OOF probs; verify PF/trades/DD math."""
    print("[smoke] sim_eurusd_6yr")
    rng = np.random.default_rng(2026)
    n = 1000
    # Synthetic EUR-priced OHLC
    close = 1.085 + rng.normal(0, 0.0002, n).cumsum()
    open_p = close + rng.normal(0, 0.0001, n)
    high = np.maximum(open_p, close) + np.abs(rng.normal(0, 0.0003, n))
    low = np.minimum(open_p, close) - np.abs(rng.normal(0, 0.0003, n))
    a = np.full(n, 0.0010)  # 10-pip ATR (realistic EUR M5)
    # Mocked OOF probs centered slightly above 0.5 (some trades will fire @0.50 thr)
    probs = np.clip(rng.normal(0.5, 0.05, n), 0.0, 1.0)
    m = simulate(probs, open_p, high, low, a, threshold=0.50)
    # Sanity: trades > 0 (synthetic noise produces some entries), pf finite, costs deducted
    trades_ok = m["trades"] > 0
    pf_ok = m["pf"] >= 0  # finite, non-neg
    dd_ok = m["max_dd_pips"] >= 0
    counts_ok = (m["n_tp"] + m["n_sl"] + m["n_timeout"]) == m["trades"]
    cost_pips = VANTAGE_SPREAD_PIPS + 2.0 * SLIPPAGE_PIPS  # = 1.4 pips
    cost_ok = abs(cost_pips - 1.4) < 1e-6
    if trades_ok and pf_ok and dd_ok and counts_ok and cost_ok:
        print(f"[smoke] PASS: {m['trades']} trades, PF={m['pf']}, DD={m['max_dd_pips']}p, "
              f"counts(TP/SL/TO)={m['n_tp']}/{m['n_sl']}/{m['n_timeout']}")
        return 0
    print(f"[smoke] FAIL: trades={trades_ok} pf={pf_ok} dd={dd_ok} counts={counts_ok} cost={cost_ok}")
    print(f"  -> {m}")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", help="run smoke test and exit")
    args = ap.parse_args()
    if args.smoke_only:
        sys.exit(smoke_test())
    sys.exit(main())
