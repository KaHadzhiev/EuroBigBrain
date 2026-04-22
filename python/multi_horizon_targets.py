#!/usr/bin/env python3
"""Multi-horizon target sweep + DXY-residualized variant + null per fold.
Tests if 3-bar / 5-bar / 10-bar / 20-bar vol-scaled targets have different AUC.
Also adds purged+embargoed CV to honest-up our AUC.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/multi_horizon_targets.json")

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 2}

def purged_embargo_split(n, n_folds, test_size, embargo):
    """TimeSeriesSplit + embargo gap to prevent label leakage from N-bar overlap."""
    base = TimeSeriesSplit(n_splits=n_folds, test_size=test_size)
    for tr, va in base.split(np.arange(n)):
        # Drop last `embargo` train rows (their labels overlap with start of val)
        tr_purged = tr[:-embargo] if embargo > 0 else tr
        yield tr_purged, va

def build_target(close, high, atr, h_bars):
    """Long: True if forward h_bars high - close > 1×ATR."""
    fwd_max = high.rolling(h_bars).max().shift(-h_bars)
    return ((fwd_max - close) > atr).astype(int)

def evaluate(X, y, n_folds=5, embargo=0):
    aucs = []
    null_aucs = []
    rng = np.random.default_rng(42)
    y_null = pd.Series(rng.permutation(y.values), index=y.index)
    test_size = len(X)//(n_folds+2)
    for tr, va in purged_embargo_split(len(X), n_folds, test_size, embargo):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        aucs.append(roc_auc_score(yva, m.predict(Xva)))

        ytrn, yvan = y_null.iloc[tr], y_null.iloc[va]
        d_trn = lgb.Dataset(Xtr, label=ytrn)
        d_van = lgb.Dataset(Xva, label=yvan, reference=d_trn)
        mn = lgb.train(PARAMS, d_trn, num_boost_round=300, valid_sets=[d_van],
                       callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        null_aucs.append(roc_auc_score(yvan, mn.predict(Xva)))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(null_aucs))

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES)
    print(f"rows={len(f):,}")

    # Need close, high, atr
    close = f['close']
    # high: rebuild from rolling logic — we don't have raw high in features.
    # Approximate: use rolling max of close as a proxy. For better accuracy, we'd need to re-pull from raw OHLCV.
    # For this experiment, target_5bar_up_1atr already uses true high; we trust it as ground truth for h=5.
    atr = f['atr14_norm'] * close

    # Use existing target_5bar_up_1atr as the h=5 baseline
    base_feat_cols = [c for c in f.columns if not c.startswith('target_') and c != 'close']

    # === Variant 1: DXY-residualized return ===
    if 'dxy_proxy' in f.columns:
        # subtract mechanical EUR contribution: residual = dxy - beta*log(EUR)
        dxy = f['dxy_proxy']
        eur_log = np.log(close)
        cov = np.cov(dxy, eur_log)[0, 1]
        var = np.var(eur_log)
        beta = cov / var if var > 0 else 0
        f = f.copy()
        f['dxy_resid'] = dxy - beta * eur_log
        f['dxy_resid_ret_5'] = f['dxy_resid'] - f['dxy_resid'].shift(5)
    feat_cols_resid = base_feat_cols + (['dxy_resid_ret_5'] if 'dxy_resid_ret_5' in f.columns else [])

    results = []

    # === Multi-horizon test (using h=5 as reference; h=3,10,20 we'll need to rebuild) ===
    # For h=5 we use existing target. For 3/10/20 we approximate with rolling close max.
    # Note: this uses close-only proxy for fwd extreme; results ordering should still be correct.
    horizons = [3, 5, 10, 20]
    for h in horizons:
        if h == 5:
            y = f['target_5bar_up_1atr'].copy()
        else:
            fwd_max_close = close.rolling(h).max().shift(-h)
            y = ((fwd_max_close - close) > atr).astype(int)
        # Drop NaNs (last h rows)
        mask = y.notna() & f[base_feat_cols].notna().all(axis=1) & atr.notna()
        Xs = f.loc[mask, base_feat_cols]
        ys = y[mask]
        embargo = h + 5  # purge + small buffer
        print(f"\n[h={h}] target balance={ys.mean():.3f}, rows={len(Xs):,}, embargo={embargo}")
        mean_auc, std_auc, null_auc = evaluate(Xs, ys, n_folds=5, embargo=embargo)
        edge = mean_auc - null_auc
        print(f"  AUC={mean_auc:.4f}±{std_auc:.4f}, null={null_auc:.4f}, edge={edge:+.4f}")
        results.append({
            'variant': 'baseline', 'horizon': h, 'embargo': embargo,
            'mean_auc': mean_auc, 'std_auc': std_auc, 'null_auc': null_auc, 'edge': edge,
            'n_rows': len(Xs), 'target_balance': float(ys.mean()),
        })

    # === DXY-residualized variant on h=5 ===
    if 'dxy_resid_ret_5' in feat_cols_resid:
        y = f['target_5bar_up_1atr']
        mask = y.notna() & f[feat_cols_resid].notna().all(axis=1)
        Xs = f.loc[mask, feat_cols_resid]
        ys = y[mask]
        print(f"\n[h=5 + dxy_resid] rows={len(Xs):,}")
        mean_auc, std_auc, null_auc = evaluate(Xs, ys, n_folds=5, embargo=10)
        edge = mean_auc - null_auc
        print(f"  AUC={mean_auc:.4f}±{std_auc:.4f}, null={null_auc:.4f}, edge={edge:+.4f}")
        results.append({
            'variant': 'dxy_residualized', 'horizon': 5, 'embargo': 10,
            'mean_auc': mean_auc, 'std_auc': std_auc, 'null_auc': null_auc, 'edge': edge,
            'n_rows': len(Xs), 'target_balance': float(ys.mean()),
        })

    print("\n=== SUMMARY ===")
    print(f"{'variant':<22} {'h':>3} {'AUC':>8} {'null':>8} {'edge':>7}")
    for r in sorted(results, key=lambda x: x['edge'], reverse=True):
        print(f"{r['variant']:<22} {r['horizon']:>3} {r['mean_auc']:>8.4f} {r['null_auc']:>8.4f} {r['edge']:>+7.4f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(results, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
