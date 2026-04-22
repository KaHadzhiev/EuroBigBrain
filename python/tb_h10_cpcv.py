#!/usr/bin/env python3
"""Combinatorial Purged CV (de Prado) on h=10/sl=0.7/tp=2.0 with TRUE bar high/low.
N=8 splits, k=2 test groups. Total 28 paths. Reports PF distribution.
Auditor #1 recommendation — addresses selection bias more rigorously than 5-fold TS.
"""
import os, json, time
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_CSV  = os.path.expanduser("~/GoldBigBrain/data/EURUSD_M5_full.csv")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_cpcv.json")
H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
N_SPLITS, K_TEST = 8, 2
EUR_PIP=0.0001; SPREAD=0.7*EUR_PIP

PARAMS = {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.03,
          'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
          'min_child_samples':200,'lambda_l1':0.1,'lambda_l2':0.1,'verbose':-1,'num_threads':3}

def tb_TRUE(close,atr,th,tl,h,sl_m,tp_m):
    n=len(close); l=np.zeros(n,dtype=np.int8); cl=close.values; at=atr.values; tha=th.values; tla=tl.values
    for i in range(n-h):
        tp_px=cl[i]+tp_m*at[i]; sl_px=cl[i]-sl_m*at[i]
        for j in range(1,h+1):
            if tla[i+j]<=sl_px: l[i]=0; break
            if tha[i+j]>=tp_px: l[i]=1; break
    return pd.Series(l,index=close.index)

def main():
    t0=time.time()
    f=pd.read_parquet(FEATURES); raw=pd.read_csv(RAW_CSV)
    raw['time']=pd.to_datetime(raw['time']); raw=raw.set_index('time').sort_index()
    common=f.index.intersection(raw.index); f=f.loc[common]; raw=raw.loc[common]
    close=f['close']; atr=f['atr14_norm']*close
    fc=[c for c in f.columns if not c.startswith('target_') and c!='close']
    y=tb_TRUE(close,atr,raw['high'],raw['low'],H_BARS,SL_MULT,TP_MULT)
    fwd_hi=raw['high'].rolling(H_BARS).max().shift(-H_BARS)
    fwd_lo=raw['low'].rolling(H_BARS).min().shift(-H_BARS)
    mask=y.notna()&f[fc].notna().all(axis=1)&atr.notna()&fwd_hi.notna()&fwd_lo.notna()
    Xs=f.loc[mask,fc]; ys=y[mask]
    cl_s=close[mask].values; at_s=atr[mask].values; hi_s=fwd_hi[mask].values; lo_s=fwd_lo[mask].values
    n=len(Xs)
    print(f"rows={n:,}, balance={ys.mean():.3f}, N_SPLITS={N_SPLITS}, K_TEST={K_TEST}")

    embargo=H_BARS+5
    splits=np.array_split(np.arange(n), N_SPLITS)
    paths_pf=[]; paths_auc=[]; paths_n=[]
    cnt=0
    for test_combo in combinations(range(N_SPLITS), K_TEST):
        cnt+=1
        test_idx=np.concatenate([splits[i] for i in test_combo])
        train_idx=np.array([i for i in range(n) if not any(i in splits[j] for j in test_combo)])
        # purge embargo around train/test boundaries
        if embargo>0:
            for j in test_combo:
                start,end=splits[j][0],splits[j][-1]
                train_idx=train_idx[(train_idx<start-embargo)|(train_idx>end+embargo)]
        if len(train_idx)<1000 or len(test_idx)<100: continue
        Xtr=Xs.iloc[train_idx]; ytr=ys.iloc[train_idx]
        Xte=Xs.iloc[test_idx]; yte=ys.iloc[test_idx]
        d=lgb.Dataset(Xtr,label=ytr)
        m=lgb.train(PARAMS,d,200,callbacks=[lgb.log_evaluation(0)])
        p=m.predict(Xte)
        try: auc=roc_auc_score(yte,p)
        except: auc=float('nan')
        sig=p>THR; nt=int(sig.sum())
        if nt<10:
            paths_pf.append(None); paths_auc.append(auc); paths_n.append(nt); continue
        e=cl_s[test_idx][sig]; ae=at_s[test_idx][sig]
        he=hi_s[test_idx][sig]; le=lo_s[test_idx][sig]
        tp_px=e+TP_MULT*ae; sl_px=e-SL_MULT*ae
        tph=(he>=tp_px); slh=(le<=sl_px)
        wins=tph&~slh; losses=slh&~tph; ambig=tph&slh; timeout=~tph&~slh
        pnl=np.zeros(nt)
        pnl[wins]=TP_MULT*ae[wins]; pnl[losses|ambig]=-SL_MULT*ae[losses|ambig]
        pnl[timeout]=(he[timeout]+le[timeout])/2-e[timeout]; pnl-=SPREAD
        pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
        paths_pf.append(float(pf)); paths_auc.append(float(auc)); paths_n.append(nt)
        if cnt%5==0: print(f"  path {cnt}: AUC={auc:.4f} n={nt} PF={pf:.3f}")
    pfs=[p for p in paths_pf if p is not None and not np.isinf(p)]
    aucs=[a for a in paths_auc if not np.isnan(a)]
    print(f"\nCPCV {len(paths_pf)} paths:")
    print(f"  AUC: mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}")
    print(f"  PF:  median={np.median(pfs):.3f} p5={np.percentile(pfs,5):.3f} p95={np.percentile(pfs,95):.3f}")
    print(f"  PF>=1.3: {sum(1 for p in pfs if p>=1.3)}/{len(pfs)} ({100*sum(1 for p in pfs if p>=1.3)/len(pfs):.0f}%)")
    out={'n_paths':len(paths_pf),'paths_pf':paths_pf,'paths_auc':paths_auc,'paths_n':paths_n,
         'pf_median':float(np.median(pfs)),'pf_p5':float(np.percentile(pfs,5)),'pf_p95':float(np.percentile(pfs,95)),
         'auc_mean':float(np.mean(aucs)),'auc_std':float(np.std(aucs)),
         'pct_pf_ge_13':float(sum(1 for p in pfs if p>=1.3)/len(pfs))}
    os.makedirs(os.path.dirname(OUT_JSON),exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out,fh,indent=2)
    print(f"Saved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
