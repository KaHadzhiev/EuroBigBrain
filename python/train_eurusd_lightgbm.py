#!/usr/bin/env python3
"""LightGBM walk-forward 5-fold CV on EURUSD M5 features.

Per agent E spec: gate AUC ≥ 0.58 (NOT 0.65 — gold target won't transfer to FX).
Per HARD RULE feedback_null_test_before_mt5: also runs shuffled-label null test
to confirm signal vs noise. Real AUC must beat null AUC by ≥0.03.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/lightgbm_eurusd_results.json")

# LightGBM params per FX practice: shallower than gold, higher regularization
PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 200,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'num_threads': 8,
}

def train_one(X, y, n_folds=5, target_name="?", shuffle_y=False):
    if shuffle_y:
        rng = np.random.default_rng(42)
        y = pd.Series(rng.permutation(y.values), index=y.index)
    aucs = []
    fold_dates = []
    feat_imp_sum = pd.Series(0.0, index=X.columns)
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=len(X)//(n_folds+2))
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        model = lgb.train(PARAMS, d_tr, num_boost_round=300,
                          valid_sets=[d_va], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        pred = model.predict(Xva)
        a = roc_auc_score(yva, pred)
        aucs.append(a)
        fold_dates.append((str(Xtr.index.min())[:10], str(Xva.index.min())[:10], str(Xva.index.max())[:10], len(Xva)))
        gain = pd.Series(model.feature_importance(importance_type='gain'), index=X.columns)
        feat_imp_sum += gain / gain.sum()
    return {
        'target': target_name,
        'shuffled': shuffle_y,
        'fold_aucs': [float(a) for a in aucs],
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'fold_windows': fold_dates,
        'top_features': feat_imp_sum.sort_values(ascending=False).head(15).to_dict(),
    }

def main():
    t0 = time.time()
    print(f"Loading {FEATURES}...")
    f = pd.read_parquet(FEATURES)
    print(f"  rows={len(f):,}, cols={len(f.columns)}")
    feat_cols = [c for c in f.columns if not c.startswith('target_') and c != 'close']
    print(f"  feature count: {len(feat_cols)}")

    results = {'feature_count': len(feat_cols), 'feature_list': feat_cols, 'rows': len(f), 'experiments': []}

    for target in ['target_next_dir', 'target_5bar_up_1atr']:
        X = f[feat_cols]
        y = f[target]
        print(f"\n=== Real signal: {target} ===")
        real = train_one(X, y, target_name=target, shuffle_y=False)
        print(f"  Mean AUC: {real['mean_auc']:.4f} ± {real['std_auc']:.4f}")
        print(f"  Top 5 features: {list(real['top_features'].items())[:5]}")

        print(f"\n=== Null (shuffled label): {target} ===")
        null = train_one(X, y, target_name=target, shuffle_y=True)
        print(f"  Mean AUC: {null['mean_auc']:.4f} (gate: real - null ≥ 0.03)")

        results['experiments'].append({'real': real, 'null': null, 'edge': real['mean_auc'] - null['mean_auc']})

    # Verdict
    print("\n=== VERDICT ===")
    for exp in results['experiments']:
        edge = exp['edge']
        verdict = "PASS" if exp['real']['mean_auc'] >= 0.55 and edge >= 0.03 else "FAIL"
        print(f"  {exp['real']['target']}: real_auc={exp['real']['mean_auc']:.4f}, edge={edge:+.4f} → {verdict}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {OUT_JSON}")
    print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
