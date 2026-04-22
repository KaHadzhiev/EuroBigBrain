#!/usr/bin/env python3
"""Bootstrap p-value for TB sweet-spot winner sl=0.7/tp=1.5.
H0: PF >= 1.0 by chance. We resample trades 1000x, count how often PF <= 1.0.
Deploy gate: p < 0.05.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_bootstrap.json")

H_BARS, SL_MULT, TP_MULT, THR = 5, 0.7, 1.5, 0.38
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
    high_proxy = close.rolling(H_BARS).max().shift(-H_BARS)
    low_proxy  = close.rolling(H_BARS).min().shift(-H_BARS)

    print(f"Building TB target h={H_BARS}, SL={SL_MULT}, TP={TP_MULT}...")
    y = tb_label(close, atr, H_BARS, SL_MULT, TP_MULT)
    mask = y.notna() & f[feat_cols].notna().all(axis=1) & atr.notna() & high_proxy.notna() & low_proxy.notna()
    Xs = f.loc[mask, feat_cols]; ys = y[mask]
    cl_s = close[mask]; at_s = atr[mask]; hi_s = high_proxy[mask]; lo_s = low_proxy[mask]

    tscv = TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
    oof = pd.Series(np.nan, index=ys.index)
    for tr, va in tscv.split(Xs):
        if len(tr) > H_BARS+5: tr = tr[:-(H_BARS+5)]
        Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
        ytr, yva = ys.iloc[tr], ys.iloc[va]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        oof.iloc[va] = m.predict(Xva)

    valid = oof.notna()
    cl_v = cl_s[valid]; at_v = at_s[valid]; hi_v = hi_s[valid]; lo_v = lo_s[valid]; oof_v = oof[valid]
    sig = oof_v > THR
    n = int(sig.sum())
    entry = cl_v[sig].values; ae = at_v[sig].values; he = hi_v[sig].values; le = lo_v[sig].values
    tp_px = entry + TP_MULT * ae; sl_px = entry - SL_MULT * ae
    tp_hit = (he >= tp_px); sl_hit = (le <= sl_px)
    wins = tp_hit & ~sl_hit
    losses = sl_hit & ~tp_hit
    ambig = tp_hit & sl_hit
    timeout = ~tp_hit & ~sl_hit

    # Per-trade PnL array (in price units)
    pnl_per = np.zeros(n)
    pnl_per[wins] = TP_MULT * ae[wins]
    pnl_per[losses | ambig] = -SL_MULT * ae[losses | ambig]
    pnl_per[timeout] = (he[timeout] + le[timeout])/2 - entry[timeout]
    pnl_per -= SPREAD  # spread per trade

    real_pf = pnl_per[pnl_per>0].sum() / abs(pnl_per[pnl_per<0].sum()) if (pnl_per<0).any() else float('inf')
    real_total = pnl_per.sum() / EUR_PIP
    print(f"Real: n={n}, PF={real_pf:.3f}, total_pips={real_total:.1f}")

    # Bootstrap: resample n trades with replacement 1000x
    rng = np.random.default_rng(42)
    boot_pfs = []
    boot_totals = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        pp = pnl_per[idx]
        if (pp<0).any():
            pf = pp[pp>0].sum() / abs(pp[pp<0].sum())
        else:
            pf = float('inf')
        boot_pfs.append(pf)
        boot_totals.append(pp.sum())

    boot_pfs = np.array(boot_pfs)
    boot_totals = np.array(boot_totals)
    p_pf_le_1 = float((boot_pfs <= 1.0).mean())
    p_total_le_0 = float((boot_totals <= 0).mean())
    pf_p5, pf_p50, pf_p95 = np.percentile(boot_pfs, [5, 50, 95])
    total_pips_p5 = np.percentile(boot_totals, 5) / EUR_PIP

    print(f"\nBootstrap (1000 resamples):")
    print(f"  P(PF <= 1.0) = {p_pf_le_1:.4f}  (deploy gate p < 0.05 = {'PASS' if p_pf_le_1 < 0.05 else 'FAIL'})")
    print(f"  P(total <= 0) = {p_total_le_0:.4f}")
    print(f"  PF p5/p50/p95 = {pf_p5:.3f} / {pf_p50:.3f} / {pf_p95:.3f}")
    print(f"  Worst-5% total pips: {total_pips_p5:.1f}")

    # Per-year breakdown
    idx_sig = oof_v.index[sig.values]
    per_year = {}
    for yr in pd.Series(idx_sig.year).unique():
        ymask = (idx_sig.year == yr)
        if ymask.sum() < 10: continue
        py = pnl_per[ymask]
        py_pf = py[py>0].sum() / abs(py[py<0].sum()) if (py<0).any() else float('inf')
        per_year[int(yr)] = {'n': int(ymask.sum()), 'pf': round(py_pf, 3),
                              'pips': round(py.sum()/EUR_PIP, 1)}
    print(f"\nPer-year:")
    for yr, m in sorted(per_year.items()):
        print(f"  {yr}: n={m['n']} PF={m['pf']} pips={m['pips']}")

    out = {
        'config': {'h': H_BARS, 'sl': SL_MULT, 'tp': TP_MULT, 'thr': THR},
        'n_trades': n, 'real_pf': float(real_pf), 'real_total_pips': float(real_total),
        'bootstrap_n': 1000,
        'p_pf_le_1': p_pf_le_1, 'p_total_le_0': p_total_le_0,
        'pf_p5': float(pf_p5), 'pf_p50': float(pf_p50), 'pf_p95': float(pf_p95),
        'worst5pct_pips': float(total_pips_p5),
        'deploy_gate_pass': p_pf_le_1 < 0.05,
        'per_year': per_year,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(out, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
