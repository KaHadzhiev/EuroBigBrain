#!/usr/bin/env python3
"""EUR per-strategy parameter sweep.

For each strategy in the roster, grid-search SL_ATR x TP_ATR x vol_filter x
session-window. Goal: find the EUR-friendly knobs per strategy before
re-combining into the portfolio.

Saves: results/portfolio/eur_sweep_<strategy>__grid.csv (one row per cfg)
       results/portfolio/eur_sweep_top.json (top-K per strategy)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from eur_portfolio_run import build_ctx, simulate_signals, PIP, USD_PER_PIP  # noqa: E402

# ---- The grid ----
SL_GRID = [0.3, 0.5, 0.7, 1.0]
TP_GRID = [1.0, 1.5, 2.0, 3.0]
VOL_GRID = [0.0, 1.2, 1.5, 2.0]      # min vol/vol_ma20 to accept
HOLD_GRID = [6, 12, 24]

# ---- Strategy x sessions to try ----
STRATEGY_SESSIONS = {
    "asian_range":       [(7, 13), (7, 20)],
    "breakout_range":    [(7, 13), (13, 20), (7, 20)],
    "momentum_short":    [(7, 13), (13, 20), (7, 20)],
    "momentum_long":     [(7, 13), (13, 20), (7, 20)],
    "fade_long":         [(7, 13), (13, 20), (7, 20)],
    "fade_short":        [(7, 13), (13, 20), (7, 20)],
    "atr_bracket":       [(7, 13), (13, 20), (7, 20)],
    "vol_spike_bracket": [(7, 20), (13, 22)],
    "ema_cross_long":    [(13, 20), (7, 20)],
    "ema_cross_short":   [(13, 20), (7, 20)],
    # NEW (2026-04-23)
    "cci_fade_long":     [(7, 13), (13, 20), (7, 20)],
    "cci_fade_short":    [(7, 13), (13, 20), (7, 20)],
    "bb_squeeze_long":   [(7, 13), (13, 20), (7, 20)],
    "bb_squeeze_short":  [(7, 13), (13, 20), (7, 20)],
    "inside_inside_long":  [(7, 13), (13, 20), (7, 20)],
    "inside_inside_short": [(7, 13), (13, 20), (7, 20)],
}

EXTRA_PARAMS = {
    "asian_range":       {"max_asian_atr": 6.0},
    "breakout_range":    {"lookback": 12},
    "momentum_short":    {"bracket_offset": 0.3},
    "momentum_long":     {"bracket_offset": 0.3},
    "fade_long":         {"bracket_offset": 0.3},
    "fade_short":        {"bracket_offset": 0.3},
    "atr_bracket":       {"bracket_offset": 0.3},
    "vol_spike_bracket": {"bracket_offset": 0.3, "vol_mult": 2.0},
    "ema_cross_long":    {"bracket_offset": 0.3},
    "ema_cross_short":   {"bracket_offset": 0.3},
    # NEW
    "cci_fade_long":     {"bracket_offset": 0.3},
    "cci_fade_short":    {"bracket_offset": 0.3},
    "bb_squeeze_long":   {"bracket_offset": 0.3},
    "bb_squeeze_short":  {"bracket_offset": 0.3},
    "inside_inside_long":  {"bracket_offset": 0.0},
    "inside_inside_short": {"bracket_offset": 0.0},
}


def _summary(log: pd.DataFrame) -> dict:
    if not len(log):
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "net": 0.0, "max_dd_usd": 0.0}
    pnls = log["pnl"].to_numpy()
    eq = np.cumsum(pnls)
    peaks = np.maximum.accumulate(eq)
    dd = peaks - eq
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum())
    return {
        "trades": int(len(pnls)),
        "pf": float(gp / gl) if gl > 0 else 0.0,
        "wr": float((pnls > 0).mean()),
        "net": float(pnls.sum()),
        "max_dd_usd": float(dd.max()),
        "tp_count": int((log["exit_reason"] == "tp").sum()),
        "sl_count": int((log["exit_reason"] == "sl").sum()),
        "to_count": int((log["exit_reason"].isin(["timeout", "eod"])).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", default="2020-01-01")
    ap.add_argument("--to", dest="d_to", default="2026-04-13")
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--strategies", default=None,
                    help="comma-separated subset, default=all in STRATEGY_SESSIONS")
    ap.add_argument("--m5-path", default=None,
                    help="override M5 data path (parquet OR csv); default = EUR 6yr Dukascopy")
    ap.add_argument("--tag-prefix", default="eur_sweep",
                    help="output filename prefix; default 'eur_sweep'")
    args = ap.parse_args()

    t0 = time.time()
    m5_path = Path(args.m5_path) if args.m5_path else (DATA / 'eurusd_m5_2020_2026.parquet')
    print(f"[load] {m5_path}")
    if str(m5_path).lower().endswith('.csv'):
        df = pd.read_csv(m5_path)
    else:
        df = pd.read_parquet(m5_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[(df["time"] >= args.d_from) & (df["time"] <= args.d_to)].reset_index(drop=True)
    print(f"  rows: {len(df):,}  span={df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    print("[build] ctx (indicators)")
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)

    # Volume-ratio gate: probs = vol_v / vol_ma20. We then filter via cfg['vt'].
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    # Replace NaN/inf with 0 so generate_signals' `probs[ii] < vt` filter works.
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)

    strat_filter = set(args.strategies.split(",")) if args.strategies else None

    all_top = {}
    for entry_type, sessions in STRATEGY_SESSIONS.items():
        if strat_filter and entry_type not in strat_filter:
            continue
        print(f"\n=== {entry_type} ===  sessions={sessions}")
        rows = []
        n_cfg = (len(SL_GRID) * len(TP_GRID) * len(VOL_GRID) *
                 len(HOLD_GRID) * len(sessions))
        i_cfg = 0
        ts = time.time()
        for sl, tp, vt, hold, (s0, s1) in product(SL_GRID, TP_GRID, VOL_GRID,
                                                    HOLD_GRID, sessions):
            if tp <= sl:
                # require positive RR
                continue
            i_cfg += 1
            cfg = {
                "entry_type": entry_type,
                "vt": vt,
                "sess_start": s0, "sess_end": s1,
                **EXTRA_PARAMS.get(entry_type, {}),
            }
            signals = generate_signals(ctx, cfg, test_indices, vol_ratio)
            log = simulate_signals(ctx, signals, sl_atr=sl, tp_atr=tp,
                                   hold_bars=hold)
            s = _summary(log)
            row = {
                "entry_type": entry_type,
                "sess": f"{s0}-{s1}",
                "sl_atr": sl, "tp_atr": tp,
                "vt_vol": vt, "hold": hold,
                **s,
            }
            rows.append(row)
        grid_df = pd.DataFrame(rows)
        out_csv = RESULTS / f"{args.tag_prefix}_{entry_type}__grid.csv"
        grid_df.to_csv(out_csv, index=False)
        # top-K with min-trades gate
        ok = grid_df[grid_df["trades"] >= args.min_trades].copy()
        ok = ok.sort_values("pf", ascending=False).head(args.top_k)
        print(f"  cfgs={i_cfg}  qualifiers (trades>={args.min_trades}): {len(grid_df[grid_df['trades']>=args.min_trades])}  ({time.time()-ts:.1f}s)")
        if len(ok):
            for _, r in ok.iterrows():
                print(f"  PF={r['pf']:.3f} WR={r['wr']:.3f} trades={r['trades']:4d} net=${r['net']:+7.0f} DD=${r['max_dd_usd']:6.0f} | sess={r['sess']} sl={r['sl_atr']} tp={r['tp_atr']} vt={r['vt_vol']} hold={r['hold']}")
        else:
            print("  no qualifying configs")
        all_top[entry_type] = ok.to_dict(orient="records")

    out_top = RESULTS / f"{args.tag_prefix}_top.json"
    out_top.write_text(json.dumps(all_top, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] top-{args.top_k} per strategy -> {out_top}")
    print(f"[elapsed] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
