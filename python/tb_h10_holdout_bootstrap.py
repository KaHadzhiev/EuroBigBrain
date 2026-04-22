#!/usr/bin/env python3
"""Bootstrap CI on held-out PF=1.325 to verify it's significantly above 1.0.
Per-trade resampling, 1000 iterations, on the 296-trade held-out slice only.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_CSV  = os.path.expanduser("~/GoldBigBrain/data/EURUSD_M5_full.csv")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_holdout_boot.json")
H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
EUR_PIP=0.0001; SPREAD=0.7*EUR_PIP; SPLIT=0.70

PARAMS = {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.03,
          'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
          'min_child_samples':200,'lambda_l1':0.1,'lambda_l2':0.1,'verbose':-1,'num_threads':2}

def tb_label_TRUE(close,atr,th,tl,h,sl_m,tp_m):
    n=len(close); l=np.zeros(n,dtype=np.int8); cl=close.values; at=atr.values
    th=th.values; tl=tl.values
    for i in range(n-h):
        tp_px=cl[i]+tp_m*at[i]; sl_px=cl[i]-sl_m*at[i]
        for j in range(1,h+1):
            if tl[i+j]<=sl_px: l[i]=0; break
            if th[i+j]>=tp_px: l[i]=1; break
    return pd.Series(l,index=close.index)

def main():
    t0=time.time()
    f=pd.read_parquet(FEATURES); raw=pd.read_csv(RAW_CSV)
    raw['time']=pd.to_datetime(raw['time']); raw=raw.set_index('time').sort_index()
    common=f.index.intersection(raw.index); f=f.loc[common]; raw=raw.loc[common]
    close=f['close']; atr=f['atr14_norm']*close; true_hi=raw['high']; true_lo=raw['low']
    fc=[c for c in f.columns if not c.startswith('target_') and c!='close']
    y=tb_label_TRUE(close,atr,true_hi,true_lo,H_BARS,SL_MULT,TP_MULT)
    fwd_hi=true_hi.rolling(H_BARS).max().shift(-H_BARS)
    fwd_lo=true_lo.rolling(H_BARS).min().shift(-H_BARS)
    mask=y.notna()&f[fc].notna().all(axis=1)&atr.notna()&fwd_hi.notna()&fwd_lo.notna()
    Xs=f.loc[mask,fc]; ys=y[mask]
    cl_s=close[mask]; at_s=atr[mask]; hi_s=fwd_hi[mask]; lo_s=fwd_lo[mask]
    n=len(Xs); split=int(n*SPLIT)
    Xtr=Xs.iloc[:split-(H_BARS+5)]; ytr=ys.iloc[:split-(H_BARS+5)]
    Xte=Xs.iloc[split:]; yte=ys.iloc[split:]
    cl_te=cl_s.iloc[split:].values; at_te=at_s.iloc[split:].values
    hi_te=hi_s.iloc[split:].values; lo_te=lo_s.iloc[split:].values
    d=lgb.Dataset(Xtr,label=ytr)
    m=lgb.train(PARAMS,d,300,callbacks=[lgb.log_evaluation(0)])
    p=m.predict(Xte)
    auc=roc_auc_score(yte,p)
    sig=p>THR; nt=int(sig.sum())
    e=cl_te[sig]; ae=at_te[sig]; he=hi_te[sig]; le=lo_te[sig]
    tp_px=e+TP_MULT*ae; sl_px=e-SL_MULT*ae
    tph=(he>=tp_px); slh=(le<=sl_px)
    wins=tph&~slh; losses=slh&~tph; ambig=tph&slh; timeout=~tph&~slh
    pnl=np.zeros(nt)
    pnl[wins]=TP_MULT*ae[wins]; pnl[losses|ambig]=-SL_MULT*ae[losses|ambig]
    pnl[timeout]=(he[timeout]+le[timeout])/2 - e[timeout]; pnl-=SPREAD
    real_pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
    print(f"Held-out: AUC={auc:.4f}, n={nt}, real PF={real_pf:.3f}")
    rng=np.random.default_rng(42)
    boot_pfs=[]
    for _ in range(1000):
        idx=rng.integers(0,nt,size=nt); pp=pnl[idx]
        pf=pp[pp>0].sum()/abs(pp[pp<0].sum()) if (pp<0).any() else float('inf')
        boot_pfs.append(pf)
    boot_pfs=np.array(boot_pfs)
    p_pf_le_1=float((boot_pfs<=1.0).mean())
    p_pf_le_13=float((boot_pfs<=1.3).mean())
    p5,p50,p95=np.percentile(boot_pfs,[5,50,95])
    print(f"Bootstrap on HELD-OUT 296 trades:")
    print(f"  P(PF<=1.0) = {p_pf_le_1:.4f}  ({'PASS' if p_pf_le_1<0.05 else 'FAIL'})")
    print(f"  P(PF<=1.3) = {p_pf_le_13:.4f}  (deploy floor confidence)")
    print(f"  PF p5/p50/p95 = {p5:.3f}/{p50:.3f}/{p95:.3f}")
    out={'auc':float(auc),'n_trades':nt,'real_pf':float(real_pf),
         'boot_p_pf_le_1':p_pf_le_1,'boot_p_pf_le_13':p_pf_le_13,
         'pf_p5':float(p5),'pf_p50':float(p50),'pf_p95':float(p95)}
    os.makedirs(os.path.dirname(OUT_JSON),exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out,fh,indent=2)
    print(f"Saved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
