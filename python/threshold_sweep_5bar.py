#!/usr/bin/env python3
"""Convert LightGBM AUC=0.593 into tradeable PF estimates.

Sweep model_prob threshold → trades, win_rate, expected_PF on target_5bar_up_1atr.
Uses out-of-fold predictions (TimeSeriesSplit) so no look-ahead.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/threshold_sweep_5bar.json")

PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
    'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'min_child_samples': 200,
    'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 8,
}

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES)
    feat_cols = [c for c in f.columns if not c.startswith('target_') and c != 'close']

    X = f[feat_cols]
    y = f['target_5bar_up_1atr']
    close = f['close']
    # Realistic exit: forward 5-bar high move
    fwd_high = close.rolling(5).max().shift(-5).reindex(f.index)
    fwd_low  = close.rolling(5).min().shift(-5).reindex(f.index)
    # ATR rebuild from feature
    atr = f['atr14_norm'] * close

    # Build out-of-fold predictions
    n_folds = 5
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=len(X)//(n_folds+2))
    oof_pred = pd.Series(np.nan, index=X.index)
    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        oof_pred.iloc[va_idx] = m.predict(Xva)

    mask = oof_pred.notna() & atr.notna() & fwd_high.notna() & fwd_low.notna()
    oof = oof_pred[mask]
    cl  = close[mask]
    fh  = fwd_high[mask]
    fl  = fwd_low[mask]
    at  = atr[mask]

    print(f"Out-of-fold AUC: {roc_auc_score(y[mask], oof):.4f}")
    print(f"Tradeable rows: {len(oof):,}")

    # Threshold sweep with bracket EA: TP=1×ATR, SL=0.5×ATR, hold=5
    EUR_PIP = 0.0001
    SPREAD = 0.7 * EUR_PIP   # 0.7 pip Vantage spread
    COMMISSION = 0.0  # Vantage Standard STP no commission

    rows = []
    for thr in [0.45, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]:
        for sl_mult, tp_mult in [(0.5, 1.0), (0.5, 1.5), (0.7, 1.0), (0.7, 1.5), (1.0, 2.0)]:
            sig = oof > thr
            n = int(sig.sum())
            if n < 50:
                continue
            entry = cl[sig]
            atr_e = at[sig]
            fh_e  = fh[sig]
            fl_e  = fl[sig]
            tp_px = entry + tp_mult * atr_e
            sl_px = entry - sl_mult * atr_e
            # Long bracket: hits TP if fwd high reaches tp before low reaches sl (approx, no path order)
            tp_hit = (fh_e >= tp_px)
            sl_hit = (fl_e <= sl_px)
            # Conservative: if both hit, assume SL first
            wins = tp_hit & ~sl_hit
            losses = sl_hit
            timeout = ~tp_hit & ~sl_hit
            # PnL per trade (in price units)
            pnl_w = (tp_mult * atr_e[wins]).sum()
            pnl_l = -(sl_mult * atr_e[losses]).sum()
            # Timeout exit at fwd high midpoint (rough)
            pnl_t = ((fh_e[timeout] + fl_e[timeout])/2 - entry[timeout]).sum()
            gross_pnl = pnl_w + pnl_l + pnl_t
            # Spread cost: 1 spread per round-trip
            spread_cost = SPREAD * n
            net_pnl = gross_pnl - spread_cost

            n_w = int(wins.sum())
            n_l = int(losses.sum())
            wr = n_w / n if n else 0
            gross_win = pnl_w
            gross_loss = abs(pnl_l) + spread_cost
            pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')
            avg_trade = net_pnl / n
            rows.append({
                'threshold': thr, 'sl_mult': sl_mult, 'tp_mult': tp_mult,
                'n_trades': n, 'wins': n_w, 'losses': n_l, 'timeouts': int(timeout.sum()),
                'win_rate': round(wr, 4), 'pf_gross': round(pf, 3),
                'net_pnl_price_units': round(net_pnl, 6),
                'avg_per_trade_pips': round(avg_trade / EUR_PIP, 2),
            })

    rows.sort(key=lambda r: r['pf_gross'], reverse=True)
    print(f"\nTop 10 (sorted by PF, after spread):")
    print(f"{'thr':>5} {'sl':>4} {'tp':>4} {'n':>5} {'WR':>6} {'PF':>6} {'pips/tr':>8}")
    for r in rows[:10]:
        print(f"{r['threshold']:>5} {r['sl_mult']:>4} {r['tp_mult']:>4} {r['n_trades']:>5} "
              f"{r['win_rate']:>6.3f} {r['pf_gross']:>6.3f} {r['avg_per_trade_pips']:>8.2f}")

    out = {'oof_auc': float(roc_auc_score(y[mask], oof)), 'tradeable_rows': len(oof), 'sweep': rows}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(out, fh, indent=2)
    print(f"\nSaved {OUT_JSON}")
    print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
