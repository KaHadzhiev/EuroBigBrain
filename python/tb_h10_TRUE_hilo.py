#!/usr/bin/env python3
"""AUDITOR FALSIFICATION TEST — h=10/sl=0.7/tp=2.0 with TRUE bar high/low.
Replaces close.rolling().max/min proxy with actual EURUSD M5 bar high/low.
Auditor bet: if PF >= 1.8 = real candidate; if < 1.4 = artifact. 1.4-1.8 = ambiguous.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_CSV  = os.path.expanduser("~/GoldBigBrain/data/EURUSD_M5_full.csv")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_TRUE_hilo.json")
H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
EUR_PIP=0.0001; SPREAD=0.7*EUR_PIP

PARAMS = {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.03,
          'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
          'min_child_samples':200,'lambda_l1':0.1,'lambda_l2':0.1,'verbose':-1,'num_threads':4}

def tb_label_TRUE(close, atr, true_hi, true_lo, h, sl_m, tp_m):
    """TRUE bar-high/low version. fwd window: real bar high/low, not close-rolling."""
    n = len(close); label = np.zeros(n, dtype=np.int8)
    cl = close.values; at = atr.values
    th = true_hi.values; tl = true_lo.values
    for i in range(n - h):
        tp_px = cl[i] + tp_m * at[i]
        sl_px = cl[i] - sl_m * at[i]
        for j in range(1, h+1):
            # TRUE bar high/low at i+j
            if tl[i+j] <= sl_px:
                # SL hit first if low penetrates AND high doesn't go above tp before low
                # Conservative: if both hit in same bar, assume SL first (intra-bar order unknown)
                if th[i+j] >= tp_px:
                    label[i] = 0  # ambiguous, conservative loss
                else:
                    label[i] = 0
                break
            if th[i+j] >= tp_px:
                label[i] = 1
                break
    return pd.Series(label, index=close.index)

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES)
    raw = pd.read_csv(RAW_CSV)
    raw['time'] = pd.to_datetime(raw['time'])
    raw = raw.set_index('time').sort_index()
    print(f"features: {len(f):,} rows, raw: {len(raw):,} rows")

    # Inner join on index
    common = f.index.intersection(raw.index)
    f = f.loc[common]
    raw = raw.loc[common]
    print(f"common: {len(f):,} rows ({f.index[0]} -> {f.index[-1]})")

    close = f['close']
    atr = f['atr14_norm'] * close
    true_hi = raw['high']  # TRUE bar high
    true_lo = raw['low']   # TRUE bar low

    fc = [c for c in f.columns if not c.startswith('target_') and c != 'close']

    print(f"Building TRUE-hilo TB target h={H_BARS}, SL={SL_MULT}, TP={TP_MULT}...")
    y = tb_label_TRUE(close, atr, true_hi, true_lo, H_BARS, SL_MULT, TP_MULT)
    mask = y.notna() & f[fc].notna().all(axis=1) & atr.notna() & true_hi.notna() & true_lo.notna()
    Xs = f.loc[mask, fc]; ys = y[mask]
    cl_s = close[mask]; at_s = atr[mask]
    th_s = true_hi[mask]; tl_s = true_lo[mask]
    print(f"rows={len(Xs):,}, balance={ys.mean():.3f} (vs proxy ~0.13 for h=10)")

    # Build TRUE forward hi/lo (max of bar highs / min of bar lows in window)
    fwd_hi = true_hi.rolling(H_BARS).max().shift(-H_BARS)  # max BAR HIGH in next h bars
    fwd_lo = true_lo.rolling(H_BARS).min().shift(-H_BARS)  # min BAR LOW in next h bars
    fwd_hi_s = fwd_hi[mask]; fwd_lo_s = fwd_lo[mask]
    mask2 = fwd_hi_s.notna() & fwd_lo_s.notna()
    Xs = Xs[mask2]; ys = ys[mask2]
    cl_s = cl_s[mask2]; at_s = at_s[mask2]; fwd_hi_s = fwd_hi_s[mask2]; fwd_lo_s = fwd_lo_s[mask2]
    print(f"after fwd-mask: {len(Xs):,}")

    tscv = TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
    rng = np.random.default_rng(42)
    y_null = pd.Series(rng.permutation(ys.values), index=ys.index)
    oof = pd.Series(np.nan, index=ys.index); fold_aucs=[]; null_aucs=[]
    embargo = H_BARS + 5

    for fold, (tr, va) in enumerate(tscv.split(Xs)):
        if embargo > 0 and len(tr) > embargo: tr = tr[:-embargo]
        Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]; ytr, yva = ys.iloc[tr], ys.iloc[va]
        d_tr = lgb.Dataset(Xtr, label=ytr); d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, 300, valid_sets=[d_va], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        p = m.predict(Xva); oof.iloc[va] = p
        fold_aucs.append(roc_auc_score(yva, p))

        ytrn, yvan = y_null.iloc[tr], y_null.iloc[va]
        d_trn = lgb.Dataset(Xtr, label=ytrn); d_van = lgb.Dataset(Xva, label=yvan, reference=d_trn)
        mn = lgb.train(PARAMS, d_trn, 300, valid_sets=[d_van], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        null_aucs.append(roc_auc_score(yvan, mn.predict(Xva)))
        print(f"  Fold {fold}: AUC={fold_aucs[-1]:.4f} null={null_aucs[-1]:.4f}")

    real_auc=float(np.mean(fold_aucs)); null_auc=float(np.mean(null_aucs))
    edge=real_auc-0.5; null_edge=null_auc-0.5
    ratio = edge/null_edge if null_edge>0 else float('inf')
    print(f"\n[NULL] real={real_auc:.4f} null={null_auc:.4f} ratio={ratio:.2f}x  ({'PASS' if ratio>=5 else 'FAIL'})")

    # PnL with TRUE bar high/low
    valid = oof.notna()
    cl_v=cl_s[valid].values; at_v=at_s[valid].values
    hi_v=fwd_hi_s[valid].values; lo_v=fwd_lo_s[valid].values
    oof_v = oof[valid].values
    sig = oof_v > THR; n=int(sig.sum())
    if n == 0:
        print("ZERO trades — falsified at threshold level")
        return
    e=cl_v[sig]; ae=at_v[sig]; he=hi_v[sig]; le=lo_v[sig]
    tp_px=e+TP_MULT*ae; sl_px=e-SL_MULT*ae
    tph=(he>=tp_px); slh=(le<=sl_px)
    wins=tph&~slh; losses=slh&~tph; ambig=tph&slh; timeout=~tph&~slh
    pnl=np.zeros(n)
    pnl[wins] = TP_MULT*ae[wins]
    pnl[losses|ambig] = -SL_MULT*ae[losses|ambig]
    pnl[timeout] = (he[timeout]+le[timeout])/2 - e[timeout]
    pnl -= SPREAD
    pf_real = pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
    print(f"\nTRUE-hilo Real: n={n}, PF={pf_real:.3f}, total_pips={pnl.sum()/EUR_PIP:.1f}")
    print(f"Win/loss/ambig/timeout breakdown: {int(wins.sum())}/{int(losses.sum())}/{int(ambig.sum())}/{int(timeout.sum())}")

    # Verdict per auditor's gate
    if pf_real >= 1.8: verdict = "REAL_CANDIDATE — PF held under real-tick simulation"
    elif pf_real < 1.4: verdict = "ARTIFACT — PF collapsed, was close-only proxy bias"
    else: verdict = "AMBIGUOUS — PF degraded but not killed, needs MT5 every-tick"
    print(f"\n*** {verdict} ***")
    print(f"(was PF=2.76 with close-only proxy; auditor predicted drop)")

    # Bootstrap
    rng2 = np.random.default_rng(42)
    boot_pfs=[]
    for _ in range(1000):
        idx = rng2.integers(0,n,size=n); pp=pnl[idx]
        pf = pp[pp>0].sum()/abs(pp[pp<0].sum()) if (pp<0).any() else float('inf')
        boot_pfs.append(pf)
    boot_pfs = np.array(boot_pfs)
    p_pf_le_1 = float((boot_pfs<=1.0).mean())
    p5,p50,p95 = np.percentile(boot_pfs,[5,50,95])
    print(f"\nBootstrap: P(PF<=1)={p_pf_le_1:.4f}  p5/p50/p95 = {p5:.3f}/{p50:.3f}/{p95:.3f}")

    out = {
        'config':{'h':H_BARS,'sl':SL_MULT,'tp':TP_MULT,'thr':THR},
        'real_auc':real_auc,'null_auc':null_auc,'edge_ratio':ratio,
        'fold_aucs':[float(a) for a in fold_aucs], 'null_fold_aucs':[float(a) for a in null_aucs],
        'pf_real_TRUE_hilo':float(pf_real), 'n_trades':n,
        'wins':int(wins.sum()),'losses':int(losses.sum()),'ambig':int(ambig.sum()),'timeout':int(timeout.sum()),
        'pf_proxy_was':2.759,
        'pf_drop':float(2.759-pf_real),
        'verdict': verdict,
        'p_pf_le_1':p_pf_le_1,'pf_p5':float(p5),'pf_p50':float(p50),'pf_p95':float(p95),
    }
    os.makedirs(os.path.dirname(OUT_JSON),exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out,fh,indent=2)
    print(f"\nSaved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
