#!/usr/bin/env python3
"""Triple-barrier target (de Prado): label = +1 if TP hit before SL, -1 if SL first, 0 if timeout.
Reformulated as binary (TP hit before SL OR not) to align with our LightGBM pipeline.
Predicts whether a long bracket trade will WIN within H bars.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/triple_barrier.json")

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 2}

def triple_barrier_label(close, atr, h_bars, sl_mult, tp_mult):
    """Vectorized approximation. For each bar, look forward h_bars and check
    whether close-only path hit tp_px before sl_px. Returns 1=TP first, 0=SL first or timeout."""
    n = len(close)
    label = np.zeros(n, dtype=np.int8)
    cl = close.values
    at = atr.values
    for i in range(n - h_bars):
        tp_px = cl[i] + tp_mult * at[i]
        sl_px = cl[i] - sl_mult * at[i]
        for j in range(1, h_bars + 1):
            if cl[i+j] >= tp_px:
                label[i] = 1
                break
            if cl[i+j] <= sl_px:
                label[i] = 0
                break
    return pd.Series(label, index=close.index)

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES)
    close = f['close']
    atr = f['atr14_norm'] * close
    feat_cols = [c for c in f.columns if not c.startswith('target_') and c != 'close']

    print("Building triple-barrier targets...")
    grid = []
    for h in [5, 10]:
        for sl, tp in [(0.5, 1.0), (0.5, 1.5), (0.7, 1.5), (1.0, 2.0)]:
            print(f"  h={h}, SL={sl}, TP={tp}...", end=" ", flush=True)
            t1 = time.time()
            y = triple_barrier_label(close, atr, h, sl, tp)
            print(f"balance={y.mean():.3f}, build={time.time()-t1:.1f}s")
            mask = y.notna() & f[feat_cols].notna().all(axis=1) & atr.notna()
            Xs = f.loc[mask, feat_cols]
            ys = y[mask]
            tscv = TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
            aucs = []
            for tr, va in tscv.split(Xs):
                Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
                ytr, yva = ys.iloc[tr], ys.iloc[va]
                d_tr = lgb.Dataset(Xtr, label=ytr)
                d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
                m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                              callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                aucs.append(roc_auc_score(yva, m.predict(Xva)))
            grid.append({'h': h, 'sl_mult': sl, 'tp_mult': tp,
                         'target_balance': float(ys.mean()),
                         'mean_auc': float(np.mean(aucs)),
                         'std_auc': float(np.std(aucs)),
                         'fold_aucs': [float(a) for a in aucs]})
            print(f"    AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}")

    grid.sort(key=lambda r: r['mean_auc'], reverse=True)
    print("\n=== SUMMARY ===")
    print(f"{'h':>3} {'sl':>4} {'tp':>4} {'bal':>6} {'AUC':>8}")
    for r in grid:
        print(f"{r['h']:>3} {r['sl_mult']:>4} {r['tp_mult']:>4} {r['target_balance']:>6.3f} {r['mean_auc']:>8.4f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(grid, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
