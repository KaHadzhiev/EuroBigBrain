#!/usr/bin/env python3
"""Per-strategy walk-forward on v8 — which of the 3 NY strategies is most robust?

Same 9-fold OOS as eur_v8_walkforward.py but each strategy runs SOLO (no portfolio).
Tells us: which strategy holds up best across OOS folds?
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals
from eur_portfolio_run import build_ctx, simulate_signals

EXTRA = {"momentum_short": {"bracket_offset": 0.3},
         "momentum_long": {"bracket_offset": 0.3},
         "ema_cross_short": {"bracket_offset": 0.3}}

WINNERS = [
    {"name": "momentum_short_NY",  "entry_type": "momentum_short",  "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "ema_cross_short_NY", "entry_type": "ema_cross_short", "sess": "13-20", "sl_atr": 0.5, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
    {"name": "momentum_long_NY",   "entry_type": "momentum_long",   "sess": "13-20", "sl_atr": 0.3, "tp_atr": 3.0, "vt_vol": 2.0, "hold": 6},
]

GATE_PIPS = 5
FOLDS = [
    ("2022-01-01", "2022-07-01"), ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"), ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01"),
]


def daily_atr(df_subset):
    df = df_subset.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    return (atr.resample('D').median() * 10000)


def run_strategy(df, w, gate=GATE_PIPS):
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    vol_ratio = np.divide(ctx["vol_v"], ctx["vol_ma20"],
                          out=np.zeros(n), where=ctx["vol_ma20"] > 0)
    vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 0.0)
    da = daily_atr(df).to_dict()
    s, e = [int(x) for x in w["sess"].split("-")]
    cfg = {"entry_type": w["entry_type"], "vt": w["vt_vol"],
           "sess_start": s, "sess_end": e,
           **EXTRA.get(w["entry_type"], {})}
    signals = generate_signals(ctx, cfg, test_indices, vol_ratio)
    log = simulate_signals(ctx, signals, sl_atr=w["sl_atr"], tp_atr=w["tp_atr"],
                           hold_bars=w["hold"])
    if not len(log):
        return None
    log['open_dt'] = pd.to_datetime(log['open_time'])
    log['date'] = log['open_dt'].dt.normalize()
    log['day_atr'] = log['date'].map(da)
    if gate > 0:
        log = log[log['day_atr'].fillna(0) >= gate].reset_index(drop=True)
    if not len(log):
        return None
    pnls = log['pnl'].to_numpy()
    eq = np.cumsum(pnls); peaks = np.maximum.accumulate(eq); dd = (peaks - eq).max()
    gp = pnls[pnls > 0].sum(); gl = abs(pnls[pnls <= 0].sum())
    return {"trades": int(len(pnls)),
            "pf": float(gp / gl) if gl > 0 else 0,
            "wr": float((pnls > 0).mean()),
            "net": float(pnls.sum()), "dd_usd": float(dd)}


def main():
    df = pd.read_parquet(DATA / "eurusd_m15_2020_2026.parquet")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"loaded {len(df):,} M15 bars")

    print(f"\n{'strategy':<25}", '  '.join(f'{s:<12}' for s, e in FOLDS))
    print('-' * 130)

    all_results = {}
    for w in WINNERS:
        row_str = f"{w['name']:<25}"
        per_fold = []
        for s, e in FOLDS:
            df_win = df[(df['time'] >= s) & (df['time'] < e)].reset_index(drop=True)
            r = run_strategy(df_win, w)
            if r is None:
                row_str += "  noTrades  "
                per_fold.append({"start": s, "end": e, "result": None})
            else:
                row_str += f"  PF{r['pf']:.2f}/n{r['trades']:>3} "
                per_fold.append({"start": s, "end": e, **r})
        print(row_str)
        all_results[w['name']] = per_fold

    print(f"\n=== Robustness summary (folds with PF >= 1.3 / total) ===")
    for name, folds in all_results.items():
        n_pass = sum(1 for f in folds if f.get('pf', 0) >= 1.3)
        n_pos = sum(1 for f in folds if f.get('pf', 0) >= 1.0)
        med_pf = np.median([f.get('pf', 0) for f in folds if f.get('pf', 0) > 0])
        total_net = sum(f.get('net', 0) for f in folds)
        print(f"  {name:<25}  PF>=1.3: {n_pass}/{len(folds)}  PF>=1.0: {n_pos}/{len(folds)}  median PF: {med_pf:.3f}  total net: ${total_net:+.0f}")

    op = RESULTS / "eur_v8_per_strategy_wf.json"
    op.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
