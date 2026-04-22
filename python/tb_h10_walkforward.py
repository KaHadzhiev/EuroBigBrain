#!/usr/bin/env python3
"""Walk-forward on h=10/sl=0.7/tp=2.0: rolling 6mo train -> 1mo test.
Tests degradation across consecutive months. Each window separate train.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_walkforward.json")
H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
EUR_PIP=0.0001; SPREAD=0.7*EUR_PIP

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 3}

def tb_label(close, atr, h, sl_m, tp_m):
    n = len(close); label = np.zeros(n, dtype=np.int8)
    cl = close.values; at = atr.values
    for i in range(n - h):
        tp_px = cl[i] + tp_m*at[i]; sl_px = cl[i] - sl_m*at[i]
        for j in range(1, h+1):
            if cl[i+j] >= tp_px: label[i]=1; break
            if cl[i+j] <= sl_px: label[i]=0; break
    return pd.Series(label, index=close.index)

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES); close=f['close']; atr=f['atr14_norm']*close
    fc = [c for c in f.columns if not c.startswith('target_') and c != 'close']
    hi = close.rolling(H_BARS).max().shift(-H_BARS); lo = close.rolling(H_BARS).min().shift(-H_BARS)
    y = tb_label(close, atr, H_BARS, SL_MULT, TP_MULT)
    mask = y.notna() & f[fc].notna().all(axis=1) & atr.notna() & hi.notna() & lo.notna()
    Xs = f.loc[mask, fc]; ys = y[mask]
    cl_s = close[mask]; at_s = atr[mask]; hi_s = hi[mask]; lo_s = lo[mask]
    idx = ys.index
    print(f"Data: {idx[0]} -> {idx[-1]} ({len(ys):,} rows, {(idx[-1]-idx[0]).days/30.44:.1f}mo)")

    months = pd.date_range(idx[0].normalize(), idx[-1], freq='MS')
    results = []
    for i in range(6, len(months)):  # need 6mo train history
        tr_start = months[i-6]; tr_end = months[i]
        te_start = months[i]; te_end = months[i+1] if i+1 < len(months) else idx[-1]
        tr_mask = (idx >= tr_start) & (idx < tr_end)
        te_mask = (idx >= te_start) & (idx < te_end)
        if tr_mask.sum() < 1000 or te_mask.sum() < 100: continue
        Xtr = Xs.loc[tr_mask]; ytr = ys.loc[tr_mask]
        Xte = Xs.loc[te_mask]; yte = ys.loc[te_mask]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
        p = m.predict(Xte)
        try: auc = roc_auc_score(yte, p)
        except: auc = float('nan')
        sig = p > THR; n = int(sig.sum())
        if n == 0:
            results.append({'test_month': str(te_start.date()), 'tr_n': int(tr_mask.sum()),
                            'te_n': int(te_mask.sum()), 'auc': float(auc), 'trades': 0, 'pf': None, 'pips': 0})
            continue
        e = cl_s[te_mask].values[sig]; ae = at_s[te_mask].values[sig]
        he = hi_s[te_mask].values[sig]; le = lo_s[te_mask].values[sig]
        tp_px = e + TP_MULT*ae; sl_px = e - SL_MULT*ae
        tph = (he >= tp_px); slh = (le <= sl_px)
        wins = tph & ~slh; losses = slh & ~tph; ambig = tph & slh; timeout = ~tph & ~slh
        pnl = np.zeros(n)
        pnl[wins] = TP_MULT*ae[wins]; pnl[losses|ambig] = -SL_MULT*ae[losses|ambig]
        pnl[timeout] = (he[timeout]+le[timeout])/2 - e[timeout]; pnl -= SPREAD
        pf = pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
        wr = int(wins.sum())/n
        results.append({'test_month': str(te_start.date()), 'tr_n': int(tr_mask.sum()),
                        'te_n': int(te_mask.sum()), 'auc': float(auc),
                        'trades': n, 'wr': round(wr,3), 'pf': round(pf,3),
                        'pips': round(pnl.sum()/EUR_PIP,1)})
        print(f"  {te_start.date()}: AUC={auc:.4f} n={n} WR={wr:.3f} PF={pf:.3f} pips={pnl.sum()/EUR_PIP:.1f}")

    pf_vals = [r['pf'] for r in results if r['pf'] is not None]
    pos_months = sum(1 for r in results if r['pf'] is not None and r['pf']>=1.0)
    print(f"\nWalk-forward summary: {len(results)} months, {pos_months} profitable (PF>=1)")
    print(f"PF range: {min(pf_vals):.3f} -> {max(pf_vals):.3f}, median {np.median(pf_vals):.3f}")
    out = {'config':{'h':H_BARS,'sl':SL_MULT,'tp':TP_MULT,'thr':THR},
           'n_months': len(results), 'profitable_months': pos_months,
           'pf_min': float(min(pf_vals)), 'pf_max': float(max(pf_vals)), 'pf_median': float(np.median(pf_vals)),
           'monthly': results}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out, fh, indent=2)
    print(f"Saved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
