#!/usr/bin/env python3
"""Add WMR fix proximity + month-end + event-window features (Agent 3 priority #2).
Free, zero external data. Test AUC delta vs baseline.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/wmr_features_test.json")

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 2}

def add_wmr_features(f):
    idx = f.index
    h = pd.Series(idx.hour, index=idx)
    m = pd.Series(idx.minute, index=idx)
    dom = pd.Series(idx.day, index=idx)
    dow = pd.Series(idx.dayofweek, index=idx)
    mon = pd.Series(idx.month, index=idx)

    # WMR 4pm London fix = 16:00 GMT (winter). Distance in minutes (signed).
    wmr_min = (h - 16) * 60 + m
    f['wmr_dist_abs_min'] = wmr_min.abs().clip(upper=240)
    f['wmr_pre_60m'] = ((wmr_min >= -60) & (wmr_min <= 0)).astype(int)
    f['wmr_post_60m'] = ((wmr_min > 0) & (wmr_min <= 60)).astype(int)

    # Month/quarter/year-end flags (last 3 trading days)
    is_month_end_3d = (dom >= 28).astype(int)
    is_quarter_end_3d = (is_month_end_3d & ((mon == 3) | (mon == 6) | (mon == 9) | (mon == 12))).astype(int)
    is_year_end_3d = (is_month_end_3d & (mon == 12)).astype(int)
    f['month_end_3d'] = is_month_end_3d
    f['qtr_end_3d'] = is_quarter_end_3d
    f['year_end_3d'] = is_year_end_3d

    # Interactions: WMR pre-fix on month-end (the "anomaly window")
    f['wmr_pre_x_qtr_end'] = f['wmr_pre_60m'] * f['qtr_end_3d']
    f['wmr_pre_x_month_end'] = f['wmr_pre_60m'] * f['month_end_3d']

    # ECB Thursday window 12:30-14:30 GMT
    f['ecb_window'] = ((dow == 3) & (h >= 12) & (h < 15)).astype(int)
    # NFP first Friday 13:30 ±15 min
    f['nfp_window'] = ((dow == 4) & (dom <= 7) & (h == 13) & (m >= 15) & (m <= 45)).astype(int)
    # FOMC last Wed of month 18:00 ±30 min (approximation)
    f['fomc_window'] = ((dow == 2) & (dom >= 22) & (h == 17) & (m >= 30)).astype(int) | \
                       ((dow == 2) & (dom >= 22) & (h == 18) & (m <= 30)).astype(int)

    # 17:00 ET rollover (20:00-22:00 GMT)
    f['rollover_window'] = ((h >= 20) & (h < 22)).astype(int)

    return f

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES)
    print(f"baseline rows={len(f):,}, cols={len(f.columns)}")

    f = add_wmr_features(f)
    new_feats = ['wmr_dist_abs_min', 'wmr_pre_60m', 'wmr_post_60m',
                 'month_end_3d', 'qtr_end_3d', 'year_end_3d',
                 'wmr_pre_x_qtr_end', 'wmr_pre_x_month_end',
                 'ecb_window', 'nfp_window', 'fomc_window', 'rollover_window']
    print(f"added {len(new_feats)} WMR/event features → cols={len(f.columns)}")
    print(f"firing rates: " + ", ".join(f"{c}={f[c].mean():.3f}" for c in new_feats[:6]))

    base_feats = [c for c in f.columns if not c.startswith('target_') and c != 'close' and c not in new_feats]
    aug_feats = base_feats + new_feats
    y = f['target_5bar_up_1atr']
    mask = y.notna() & f[aug_feats].notna().all(axis=1)
    f, y = f[mask], y[mask]

    def cv_auc(X, y):
        n = len(X)
        tscv = TimeSeriesSplit(n_splits=5, test_size=n//7)
        aucs = []
        for tr, va in tscv.split(X):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            d_tr = lgb.Dataset(Xtr, label=ytr)
            d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
            m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            aucs.append(roc_auc_score(yva, m.predict(Xva)))
        return aucs

    print("\n[BASELINE: 23 features]")
    base_aucs = cv_auc(f[base_feats], y)
    print(f"  fold AUCs: {[round(a,4) for a in base_aucs]}, mean={np.mean(base_aucs):.4f}")

    print(f"\n[+WMR ({len(new_feats)} features added)]")
    aug_aucs = cv_auc(f[aug_feats], y)
    print(f"  fold AUCs: {[round(a,4) for a in aug_aucs]}, mean={np.mean(aug_aucs):.4f}")

    delta = np.mean(aug_aucs) - np.mean(base_aucs)
    print(f"\n[DELTA] +{delta:+.4f} AUC ({'UP' if delta > 0 else 'DOWN'})")

    out = {'baseline_aucs': [float(a) for a in base_aucs],
           'augmented_aucs': [float(a) for a in aug_aucs],
           'baseline_mean': float(np.mean(base_aucs)),
           'augmented_mean': float(np.mean(aug_aucs)),
           'delta': float(delta),
           'new_features': new_feats,
           'firing_rates': {c: float(f[c].mean()) for c in new_feats}}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(out, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
