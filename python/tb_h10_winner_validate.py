#!/usr/bin/env python3
"""TB grid winner h=10/SL=0.7/TP=2.0 (PF=1.94 @ 14.6tr/mo) full validation:
1. Null permutation test (HARD RULE: real >= 5x null edge gate)
2. Bootstrap p-value (1000 resamples, deploy gate p < 0.05)
3. Per-year breakdown
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_winner_validate.json")

H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
EUR_PIP = 0.0001
SPREAD = 0.7 * EUR_PIP

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 4}

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
    hi_proxy = close.rolling(H_BARS).max().shift(-H_BARS)
    lo_proxy = close.rolling(H_BARS).min().shift(-H_BARS)

    print(f"Building TB target h={H_BARS}, SL={SL_MULT}, TP={TP_MULT}...")
    y = tb_label(close, atr, H_BARS, SL_MULT, TP_MULT)
    mask = y.notna() & f[feat_cols].notna().all(axis=1) & atr.notna() & hi_proxy.notna() & lo_proxy.notna()
    Xs = f.loc[mask, feat_cols]; ys = y[mask]
    cl_s = close[mask]; at_s = atr[mask]; hi_s = hi_proxy[mask]; lo_s = lo_proxy[mask]
    print(f"rows={len(Xs):,}, balance={ys.mean():.3f}")

    tscv = TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
    rng = np.random.default_rng(42)
    y_null = pd.Series(rng.permutation(ys.values), index=ys.index)

    oof = pd.Series(np.nan, index=ys.index)
    fold_aucs = []; null_aucs = []
    embargo = H_BARS + 5
    for fold, (tr, va) in enumerate(tscv.split(Xs)):
        if embargo > 0 and len(tr) > embargo: tr = tr[:-embargo]
        Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
        ytr, yva = ys.iloc[tr], ys.iloc[va]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        p = m.predict(Xva); oof.iloc[va] = p
        a = roc_auc_score(yva, p); fold_aucs.append(a)

        ytrn, yvan = y_null.iloc[tr], y_null.iloc[va]
        d_trn = lgb.Dataset(Xtr, label=ytrn)
        d_van = lgb.Dataset(Xva, label=yvan, reference=d_trn)
        mn = lgb.train(PARAMS, d_trn, num_boost_round=300, valid_sets=[d_van],
                       callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        null_aucs.append(roc_auc_score(yvan, mn.predict(Xva)))
        print(f"  Fold {fold}: AUC={a:.4f}  null={null_aucs[-1]:.4f}")

    real_auc = float(np.mean(fold_aucs)); null_auc = float(np.mean(null_aucs))
    edge = real_auc - 0.5; null_edge = null_auc - 0.5
    ratio = edge / null_edge if null_edge > 0 else float('inf')
    print(f"\n[NULL] real_auc={real_auc:.4f} null_auc={null_auc:.4f} ratio={ratio:.2f}x  ({'PASS' if ratio>=5 else 'FAIL'})")

    valid = oof.notna()
    cl_v = cl_s[valid].values; at_v = at_s[valid].values; hi_v = hi_s[valid].values; lo_v = lo_s[valid].values
    oof_v = oof[valid].values
    idx_v = oof[valid].index

    sig = oof_v > THR
    n = int(sig.sum())
    entry = cl_v[sig]; ae = at_v[sig]; he = hi_v[sig]; le = lo_v[sig]
    idx_sig = idx_v[sig]
    tp_px = entry + TP_MULT * ae; sl_px = entry - SL_MULT * ae
    tp_hit = (he >= tp_px); sl_hit = (le <= sl_px)
    wins = tp_hit & ~sl_hit
    losses = sl_hit & ~tp_hit
    ambig = tp_hit & sl_hit
    timeout = ~tp_hit & ~sl_hit

    pnl_per = np.zeros(n)
    pnl_per[wins] = TP_MULT * ae[wins]
    pnl_per[losses | ambig] = -SL_MULT * ae[losses | ambig]
    pnl_per[timeout] = (he[timeout] + le[timeout])/2 - entry[timeout]
    pnl_per -= SPREAD

    real_pf = pnl_per[pnl_per>0].sum() / abs(pnl_per[pnl_per<0].sum()) if (pnl_per<0).any() else float('inf')
    real_total = pnl_per.sum() / EUR_PIP
    print(f"\nReal: n={n}, PF={real_pf:.3f}, total_pips={real_total:.1f}")

    rng2 = np.random.default_rng(42)
    boot_pfs = []; boot_totals = []
    for _ in range(1000):
        idx = rng2.integers(0, n, size=n); pp = pnl_per[idx]
        pf = pp[pp>0].sum() / abs(pp[pp<0].sum()) if (pp<0).any() else float('inf')
        boot_pfs.append(pf); boot_totals.append(pp.sum())
    boot_pfs = np.array(boot_pfs); boot_totals = np.array(boot_totals)
    p_pf_le_1 = float((boot_pfs <= 1.0).mean())
    pf_p5, pf_p50, pf_p95 = np.percentile(boot_pfs, [5, 50, 95])
    print(f"\nBootstrap (1000):")
    print(f"  P(PF<=1) = {p_pf_le_1:.4f}  ({'PASS' if p_pf_le_1<0.05 else 'FAIL'})")
    print(f"  PF p5/p50/p95 = {pf_p5:.3f} / {pf_p50:.3f} / {pf_p95:.3f}")

    per_year = {}
    for yr in pd.Series(idx_sig.year).unique():
        ym = (idx_sig.year == yr)
        if ym.sum() < 10: continue
        py = pnl_per[ym]
        py_pf = py[py>0].sum() / abs(py[py<0].sum()) if (py<0).any() else float('inf')
        per_year[int(yr)] = {'n': int(ym.sum()), 'pf': round(py_pf, 3),
                              'pips': round(py.sum()/EUR_PIP, 1)}
    print(f"\nPer-year:")
    for yr, m in sorted(per_year.items()):
        print(f"  {yr}: n={m['n']} PF={m['pf']} pips={m['pips']}")

    out = {
        'config': {'h': H_BARS, 'sl': SL_MULT, 'tp': TP_MULT, 'thr': THR},
        'real_auc': real_auc, 'null_auc': null_auc, 'edge_ratio': ratio, 'null_pass': ratio>=5,
        'fold_aucs': [float(a) for a in fold_aucs],
        'null_fold_aucs': [float(a) for a in null_aucs],
        'real_pf': float(real_pf), 'n_trades': n,
        'real_total_pips': float(real_total),
        'p_pf_le_1': p_pf_le_1, 'bootstrap_pass': p_pf_le_1<0.05,
        'pf_p5': float(pf_p5), 'pf_p50': float(pf_p50), 'pf_p95': float(pf_p95),
        'per_year': per_year,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(out, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
