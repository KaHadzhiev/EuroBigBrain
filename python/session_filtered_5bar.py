#!/usr/bin/env python3
"""Apply WG-EUR-2 recommendations: session 07-17 GMT, blackout news+rollover+WMR.
Re-train + threshold sweep on filtered set. Test if AUC 0.593 → 0.62-0.64 prediction holds.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/session_filtered_5bar.json")

PARAMS = {
    'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
    'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'min_child_samples': 200,
    'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 8,
}

def is_blackout(ts):
    """Apply Agent 2 blackouts. ts is a DatetimeIndex (UTC/GMT assumed)."""
    h, m = ts.hour, ts.minute
    dow, dom = ts.dayofweek, ts.day
    bl = pd.Series(False, index=ts)
    # NFP first Friday ±15min around 13:30 GMT
    nfp = (dow == 4) & (dom <= 7) & (((h == 13) & (m >= 15)) | ((h == 13) & (m <= 45)) | ((h == 14) & (m <= 15)))
    bl |= nfp
    # ECB Thursday 12:45 + 13:30 GMT (8 days/yr) — approximated as ANY Thursday during ECB window (over-blackout, safe)
    ecb = (dow == 3) & (((h == 12) & (m >= 30)) | (h == 13) | ((h == 14) & (m <= 30)))
    bl |= ecb
    # FOMC last Wed of month ±15min around 18:00 GMT
    last_wed = (dow == 2) & (dom >= 22)
    fomc = last_wed & (((h == 17) & (m >= 45)) | (h == 18) | ((h == 19) & (m <= 0)))
    bl |= fomc
    # WMR fix month/quarter/year end 15:50-16:10 GMT
    wmr = (dom >= 28) & (((h == 15) & (m >= 50)) | ((h == 16) & (m <= 10)))
    bl |= wmr
    # 17:00 ET rollover ~20:00-22:00 GMT (winter) — broad block
    rollover = (h >= 20) & (h < 22)
    bl |= rollover
    return bl

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES)
    print(f"Original rows: {len(f):,}")

    h = f.index.hour
    sess_mask = (h >= 7) & (h < 17)
    blackout_mask = is_blackout(f.index)
    keep = sess_mask & ~blackout_mask
    f = f[keep]
    print(f"After session 07-17 + blackout filter: {len(f):,} rows ({100*len(f)/97963:.1f}% kept)")

    feat_cols = [c for c in f.columns if not c.startswith('target_') and c != 'close']
    X = f[feat_cols]
    y = f['target_5bar_up_1atr']
    close = f['close']
    fwd_high = close.rolling(5).max().shift(-5).reindex(f.index)
    fwd_low  = close.rolling(5).min().shift(-5).reindex(f.index)
    atr = f['atr14_norm'] * close

    print(f"Target balance: {y.mean():.3f}")

    # Re-train + collect OOF preds
    n_folds = 5
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=len(X)//(n_folds+2))
    oof_pred = pd.Series(np.nan, index=X.index)
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        p = m.predict(Xva)
        oof_pred.iloc[va_idx] = p
        a = roc_auc_score(yva, p)
        fold_aucs.append(a)
        print(f"  Fold {fold}: AUC={a:.4f}")

    valid = oof_pred.notna()
    full_auc = roc_auc_score(y[valid], oof_pred[valid])
    print(f"\n[FILTERED] Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"[FILTERED] OOF pooled AUC: {full_auc:.4f}")
    print(f"[BASELINE] Was: 0.5928 ± 0.0083 → +{np.mean(fold_aucs)-0.5928:+.4f}")

    # Null test on filtered set
    rng = np.random.default_rng(42)
    y_null = pd.Series(rng.permutation(y.values), index=y.index)
    null_aucs = []
    for tr_idx, va_idx in tscv.split(X):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y_null.iloc[tr_idx], y_null.iloc[va_idx]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        null_aucs.append(roc_auc_score(yva, m.predict(Xva)))
    print(f"[NULL] Mean AUC: {np.mean(null_aucs):.4f}, edge = {np.mean(fold_aucs)-np.mean(null_aucs):+.4f} (gate ≥0.03)")

    # Threshold sweep
    EUR_PIP = 0.0001
    SPREAD = 0.7 * EUR_PIP
    cl = close[valid]; oof = oof_pred[valid]; fh = fwd_high[valid]; fl = fwd_low[valid]; at = atr[valid]
    rows = []
    for thr in [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]:
        for sl_mult, tp_mult in [(0.5, 1.0), (0.5, 1.5), (0.7, 1.0), (0.7, 1.5), (1.0, 1.5), (1.0, 2.0)]:
            sig = oof > thr
            n = int(sig.sum())
            if n < 50: continue
            entry = cl[sig]; atr_e = at[sig]; fh_e = fh[sig]; fl_e = fl[sig]
            tp_px = entry + tp_mult * atr_e
            sl_px = entry - sl_mult * atr_e
            tp_hit = (fh_e >= tp_px); sl_hit = (fl_e <= sl_px)
            wins = tp_hit & ~sl_hit; losses = sl_hit; timeout = ~tp_hit & ~sl_hit
            pnl_w = (tp_mult * atr_e[wins]).sum()
            pnl_l = -(sl_mult * atr_e[losses]).sum()
            pnl_t = ((fh_e[timeout] + fl_e[timeout])/2 - entry[timeout]).sum()
            gross_pnl = pnl_w + pnl_l + pnl_t
            spread_cost = SPREAD * n
            net_pnl = gross_pnl - spread_cost
            gross_win = pnl_w
            gross_loss = abs(pnl_l) + spread_cost
            pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')
            rows.append({
                'threshold': thr, 'sl_mult': sl_mult, 'tp_mult': tp_mult,
                'n_trades': n, 'wins': int(wins.sum()), 'losses': int(losses.sum()),
                'win_rate': round(int(wins.sum())/n, 4),
                'pf_net': round(pf, 3),
                'avg_per_trade_pips': round(net_pnl / n / EUR_PIP, 2),
            })
    rows.sort(key=lambda r: r['pf_net'], reverse=True)
    print(f"\nTop 10 (sorted by net PF):")
    print(f"{'thr':>5} {'sl':>4} {'tp':>4} {'n':>5} {'WR':>6} {'PF':>6} {'pips':>6}")
    for r in rows[:10]:
        print(f"{r['threshold']:>5} {r['sl_mult']:>4} {r['tp_mult']:>4} {r['n_trades']:>5} "
              f"{r['win_rate']:>6.3f} {r['pf_net']:>6.3f} {r['avg_per_trade_pips']:>6.2f}")

    out = {
        'rows_kept': len(f), 'rows_total': 97963,
        'mean_fold_auc': float(np.mean(fold_aucs)),
        'std_fold_auc': float(np.std(fold_aucs)),
        'oof_pooled_auc': float(full_auc),
        'null_mean_auc': float(np.mean(null_aucs)),
        'edge': float(np.mean(fold_aucs) - np.mean(null_aucs)),
        'baseline_auc': 0.5928,
        'auc_lift': float(np.mean(fold_aucs) - 0.5928),
        'fold_aucs': [float(a) for a in fold_aucs],
        'null_fold_aucs': [float(a) for a in null_aucs],
        'sweep': rows,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(out, fh, indent=2)
    print(f"\nSaved {OUT_JSON}")
    print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
