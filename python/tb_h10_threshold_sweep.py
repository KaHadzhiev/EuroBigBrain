#!/usr/bin/env python3
"""h=10/sl=0.7/tp=2.0 threshold fine-grain (0.30..0.55 step 0.01).
Find the (PF, trades/mo) frontier. Goal: find a thr that hits >=20 tr/mo at PF>=1.3.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_thr_sweep.json")
H_BARS, SL_MULT, TP_MULT = 10, 0.7, 2.0
EUR_PIP = 0.0001; SPREAD = 0.7 * EUR_PIP

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 3}

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
    f = pd.read_parquet(FEATURES); close = f['close']; atr = f['atr14_norm'] * close
    fc = [c for c in f.columns if not c.startswith('target_') and c != 'close']
    hi = close.rolling(H_BARS).max().shift(-H_BARS); lo = close.rolling(H_BARS).min().shift(-H_BARS)
    y = tb_label(close, atr, H_BARS, SL_MULT, TP_MULT)
    mask = y.notna() & f[fc].notna().all(axis=1) & atr.notna() & hi.notna() & lo.notna()
    Xs = f.loc[mask, fc]; ys = y[mask]
    cl_s = close[mask].values; at_s = atr[mask].values; hi_s = hi[mask].values; lo_s = lo[mask].values

    tscv = TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
    oof = pd.Series(np.nan, index=ys.index)
    for tr, va in tscv.split(Xs):
        if len(tr) > H_BARS+5: tr = tr[:-(H_BARS+5)]
        d_tr = lgb.Dataset(Xs.iloc[tr], label=ys.iloc[tr])
        d_va = lgb.Dataset(Xs.iloc[va], label=ys.iloc[va], reference=d_tr)
        m = lgb.train(PARAMS, d_tr, 300, valid_sets=[d_va], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        oof.iloc[va] = m.predict(Xs.iloc[va])

    valid = oof.notna()
    cl_v = cl_s[valid.values]; at_v = at_s[valid.values]; hi_v = hi_s[valid.values]; lo_v = lo_s[valid.values]; oof_v = oof[valid].values
    idx_v = oof[valid].index
    months = (idx_v[-1] - idx_v[0]).days / 30.44

    rows = []
    for thr in np.arange(0.20, 0.60, 0.01):
        sig = oof_v > thr; n = int(sig.sum())
        if n < 30: continue
        e = cl_v[sig]; ae = at_v[sig]; he = hi_v[sig]; le = lo_v[sig]
        tp_px = e + TP_MULT*ae; sl_px = e - SL_MULT*ae
        tph = (he >= tp_px); slh = (le <= sl_px)
        wins = tph & ~slh; losses = slh & ~tph; ambig = tph & slh; timeout = ~tph & ~slh
        pnl = np.zeros(n)
        pnl[wins] = TP_MULT*ae[wins]; pnl[losses|ambig] = -SL_MULT*ae[losses|ambig]
        pnl[timeout] = (he[timeout]+le[timeout])/2 - e[timeout]
        pnl -= SPREAD
        pf = pnl[pnl>0].sum() / abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
        rows.append({
            'thr': round(float(thr),2), 'n': n, 'tpm': round(n/months,1),
            'wr': round(int(wins.sum())/n,3), 'pf': round(pf,3),
            'pips': round(pnl.sum()/EUR_PIP,1),
            'pips_per_trade': round(pnl.sum()/n/EUR_PIP,2),
        })
    rows.sort(key=lambda r: r['pf'], reverse=True)
    print(f"{'thr':>5} {'n':>4} {'tpm':>5} {'WR':>5} {'PF':>6} {'pips':>7} {'p/t':>5}")
    for r in rows:
        print(f"{r['thr']:>5} {r['n']:>4} {r['tpm']:>5} {r['wr']:>5} {r['pf']:>6} {r['pips']:>7} {r['pips_per_trade']:>5}")
    # Find frontier: PF>=1.3, max trades
    deploy = [r for r in rows if r['pf']>=1.3 and r['tpm']>=20]
    if deploy:
        b = max(deploy, key=lambda r: r['tpm'])
        print(f"\nDEPLOY-FRONTIER (PF>=1.3 + tpm>=20): thr={b['thr']} PF={b['pf']} tpm={b['tpm']}")
    else:
        b = max([r for r in rows if r['pf']>=1.3], key=lambda r: r['tpm'], default=None)
        print(f"\nNo (PF>=1.3, tpm>=20) — best PF>=1.3 by tpm: thr={b['thr'] if b else 'NA'} tpm={b['tpm'] if b else 'NA'} PF={b['pf'] if b else 'NA'}")
    out = {'config':{'h':H_BARS,'sl':SL_MULT,'tp':TP_MULT}, 'months':months, 'sweep':rows, 'deploy_frontier': b}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out, fh, indent=2)
    print(f"Saved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__ == "__main__": main()
