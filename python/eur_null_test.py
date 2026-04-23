#!/usr/bin/env python3
"""Null permutation test for one EUR M15 strategy config.

Two null modes:
  - shift: circular-shift the OHLCV by random N bars; re-run strategy + sim
  - shuffle_session: randomly reassign each bar's date to another date (same
    hour); re-run

Saves PF distribution + reports edge ratio (real_PF / null_median_PF).
Per HARD RULE feedback_null_test_before_mt5: real >= 5x null => passes.

Usage:
  python eur_null_test.py --cfg-name 'momentum_short__13-20_sl0.5_tp3.0_vt2.0_h6'
  python eur_null_test.py --cfg-json '{"entry_type":...,"sess":"13-20",...}'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio" / "null_tests"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from eur_portfolio_run import build_ctx, simulate_signals  # noqa: E402


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
}


def _pf(pnls):
    if not len(pnls):
        return 0.0
    gp = pnls[pnls > 0].sum()
    gl = abs(pnls[pnls <= 0].sum())
    return float(gp / gl) if gl > 0 else 0.0


def _real_run(df, cfg):
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    s, e = [int(x) for x in cfg["sess"].split("-")]
    cfg2 = {"entry_type": cfg["entry_type"], "vt": cfg["vt_vol"],
            "sess_start": s, "sess_end": e,
            **EXTRA_PARAMS.get(cfg["entry_type"], {})}
    signals = generate_signals(ctx, cfg2, test_indices, vol_ratio)
    log = simulate_signals(ctx, signals, sl_atr=cfg["sl_atr"], tp_atr=cfg["tp_atr"],
                           hold_bars=cfg["hold"])
    return _pf(log["pnl"].to_numpy()), len(log)


def _null_shift(df, cfg, seed):
    """HOUR-SHUFFLE null: re-stamp every bar with a random hour-of-day, then
    re-sort. This destroys the hour-of-day pattern that strategies key on,
    and is the canonical 'session-shuffle' null per Hatton 5x gate.

    Diagnostic 2026-04-23 showed circular bar-shift is too generous because it
    preserves session-hour patterns the strategy keys on."""
    rng = np.random.default_rng(seed)
    n = len(df)
    df2 = df.copy()
    offsets = rng.integers(0, 96, size=n) * 15  # minutes (M15 grid)
    df2['time'] = pd.to_datetime(df2['time']) + pd.to_timedelta(offsets, unit='m')
    df2 = df2.sort_values('time').reset_index(drop=True)
    return _real_run(df2, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-json", required=True,
                    help='cfg JSON, e.g. {"entry_type":"momentum_short","sess":"13-20","sl_atr":0.5,"tp_atr":3.0,"vt_vol":2.0,"hold":6}')
    ap.add_argument("--cfg-name", default=None,
                    help="Display name for output files; auto-derived if omitted")
    ap.add_argument("--data", default="eurusd_m15_2020_2026.parquet")
    ap.add_argument("--n-nulls", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--gate-edge-ratio", type=float, default=5.0)
    args = ap.parse_args()

    cfg = json.loads(args.cfg_json)
    name = args.cfg_name or f"{cfg['entry_type']}__{cfg['sess']}_sl{cfg['sl_atr']}_tp{cfg['tp_atr']}_vt{cfg['vt_vol']}_h{cfg['hold']}"

    print(f"[null] {name}")
    print(f"  cfg: {cfg}")
    df = pd.read_parquet(DATA / args.data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  data: {len(df):,} M15 bars")

    t0 = time.time()
    real_pf, real_n = _real_run(df, cfg)
    print(f"  REAL: PF={real_pf:.3f}  trades={real_n}  ({time.time()-t0:.1f}s)")

    null_pfs = []
    for i in range(args.n_nulls):
        t = time.time()
        npf, nn = _null_shift(df, cfg, seed=args.seed_base + i)
        null_pfs.append(npf)
        if (i + 1) % 10 == 0 or i == args.n_nulls - 1:
            print(f"  null {i+1}/{args.n_nulls}: PF={npf:.3f} trades={nn} ({time.time()-t:.1f}s)")

    null_pfs = np.array(null_pfs)
    median = float(np.median(null_pfs))
    p5 = float(np.percentile(null_pfs, 5))
    p95 = float(np.percentile(null_pfs, 95))
    p_above = float((null_pfs >= real_pf).mean())
    edge_ratio = real_pf / median if median > 0 else float('inf')
    pass_gate = edge_ratio >= args.gate_edge_ratio

    out = {
        "cfg_name": name, "cfg": cfg,
        "n_nulls": args.n_nulls,
        "real_pf": real_pf, "real_n": real_n,
        "null_pf_median": median, "null_pf_p5": p5, "null_pf_p95": p95,
        "null_pf_min": float(null_pfs.min()), "null_pf_max": float(null_pfs.max()),
        "p_null_ge_real": p_above,
        "edge_ratio": edge_ratio,
        "gate_edge_ratio": args.gate_edge_ratio,
        "PASS": bool(pass_gate),
        "null_pf_dist": null_pfs.tolist(),
        "elapsed_s": time.time() - t0,
    }
    out_path = RESULTS / f"null__{name}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    verdict = "PASS" if pass_gate else "FAIL"
    print(f"\n  RESULT: real={real_pf:.3f}  null_median={median:.3f}  edge_ratio={edge_ratio:.2f}x  p(null>=real)={p_above:.3f}")
    print(f"  GATE: edge_ratio>={args.gate_edge_ratio}x  -> {verdict}")
    print(f"  saved {out_path}")
    return 0 if pass_gate else 1


if __name__ == "__main__":
    sys.exit(main())
