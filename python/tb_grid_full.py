#!/usr/bin/env python3
"""Wide TB hyperparameter grid: 8 SL x 8 TP x 3 h = 192 configs.
Each config: full LightGBM 5-fold CV + threshold sweep + best PF.
Sustained Mac saturation while we wait on 6yr data pull.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_grid_full.json")
EUR_PIP = 0.0001
SPREAD = 0.7 * EUR_PIP

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 2}

def tb_label(close, atr, h, sl_m, tp_m):
    n = len(close); label = np.zeros(n, dtype=np.int8)
    cl = close.values; at = atr.values
    for i in range(n - h):
        tp_px = cl[i] + tp_m * at[i]; sl_px = cl[i] - sl_m * at[i]
        for j in range(1, h+1):
            if cl[i+j] >= tp_px: label[i] = 1; break
            if cl[i+j] <= sl_px: label[i] = 0; break
    return pd.Series(label, index=close.index)

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES)
    close = f['close']; atr = f['atr14_norm'] * close
    feat_cols = [c for c in f.columns if not c.startswith('target_') and c != 'close']

    h_grid  = [3, 5, 10]
    sl_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    tp_grid = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]
    total = len(h_grid) * len(sl_grid) * len(tp_grid)
    print(f"Grid: {total} configs ({len(h_grid)}h x {len(sl_grid)}sl x {len(tp_grid)}tp)")

    results = []
    cnt = 0
    for h in h_grid:
        hi_proxy = close.rolling(h).max().shift(-h)
        lo_proxy = close.rolling(h).min().shift(-h)
        for sl_m in sl_grid:
            for tp_m in tp_grid:
                cnt += 1
                t1 = time.time()
                y = tb_label(close, atr, h, sl_m, tp_m)
                mask = y.notna() & f[feat_cols].notna().all(axis=1) & atr.notna() & hi_proxy.notna() & lo_proxy.notna()
                Xs = f.loc[mask, feat_cols]; ys = y[mask]
                cl_s = close[mask]; at_s = atr[mask]; hi_s = hi_proxy[mask]; lo_s = lo_proxy[mask]
                bal = float(ys.mean())
                if bal < 0.02 or bal > 0.95:
                    continue
                tscv = TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
                oof = pd.Series(np.nan, index=ys.index)
                fold_aucs = []
                for tr, va in tscv.split(Xs):
                    if len(tr) > h+5: tr = tr[:-(h+5)]
                    Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
                    ytr, yva = ys.iloc[tr], ys.iloc[va]
                    d_tr = lgb.Dataset(Xtr, label=ytr)
                    d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
                    m = lgb.train(PARAMS, d_tr, num_boost_round=200, valid_sets=[d_va],
                                  callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                    p = m.predict(Xva); oof.iloc[va] = p
                    fold_aucs.append(roc_auc_score(yva, p))
                auc = float(np.mean(fold_aucs))

                valid = oof.notna()
                cl_v=cl_s[valid].values; at_v=at_s[valid].values; hi_v=hi_s[valid].values; lo_v=lo_s[valid].values; oof_v=oof[valid].values
                idx_v = oof[valid].index
                months = (idx_v[-1] - idx_v[0]).days / 30.44

                best = None
                for thr in np.arange(0.20, 0.55, 0.02):
                    sig = oof_v > thr
                    n = int(sig.sum())
                    if n < 60: continue
                    entry=cl_v[sig]; ae=at_v[sig]; he=hi_v[sig]; le=lo_v[sig]
                    tp_px=entry+tp_m*ae; sl_px=entry-sl_m*ae
                    tp_hit=(he>=tp_px); sl_hit=(le<=sl_px)
                    wins=tp_hit&~sl_hit; losses=sl_hit&~tp_hit; ambig=tp_hit&sl_hit; timeout=~tp_hit&~sl_hit
                    pnl_w=(tp_m*ae[wins]).sum()
                    pnl_l=-(sl_m*ae[losses|ambig]).sum()
                    pnl_t=((he[timeout]+le[timeout])/2 - entry[timeout]).sum()
                    net = pnl_w + pnl_l + pnl_t - SPREAD*n
                    gw=pnl_w; gl=abs(pnl_l)+SPREAD*n
                    pf = gw/gl if gl>0 else float('inf')
                    if best is None or pf > best['pf']:
                        best = {'thr': round(float(thr),2), 'n': n, 'tpm': round(n/months,1),
                                'wr': round(int(wins.sum())/n,3), 'pf': round(pf,3),
                                'pips': round(net/EUR_PIP,1)}
                results.append({
                    'h': h, 'sl': sl_m, 'tp': tp_m, 'auc': round(auc,4), 'balance': round(bal,3),
                    'best': best, 'wall_s': round(time.time()-t1,1),
                })
                if cnt % 8 == 0 or cnt == total:
                    elapsed = time.time()-t0; eta = elapsed/cnt*(total-cnt)
                    print(f"[{cnt:3d}/{total}] h={h} sl={sl_m} tp={tp_m} AUC={auc:.4f} bal={bal:.3f} "
                          f"PF={best['pf'] if best else 'na'} "
                          f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    # Sort by best PF
    results_with_best = [r for r in results if r['best']]
    results_with_best.sort(key=lambda r: r['best']['pf'], reverse=True)
    print(f"\n=== TOP 20 (by best PF, n>=60) ===")
    print(f"{'h':>3} {'sl':>4} {'tp':>4} {'AUC':>6} {'thr':>5} {'n':>5} {'/mo':>5} {'WR':>5} {'PF':>6} {'pips':>7}")
    for r in results_with_best[:20]:
        b = r['best']
        print(f"{r['h']:>3} {r['sl']:>4} {r['tp']:>4} {r['auc']:>6} {b['thr']:>5} {b['n']:>5} {b['tpm']:>5} {b['wr']:>5} {b['pf']:>6} {b['pips']:>7}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(results, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
