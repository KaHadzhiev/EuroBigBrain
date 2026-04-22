#!/usr/bin/env python3
"""Selection-bias test: train h=10/sl=0.7/tp=2.0 on first 70% (months never seen by 192-grid).
Test on last 30% as TRUE held-out. The 192-grid + threshold sweep saw OOF on full data,
so selection bias is real. This split does FIT on a slice + test on a slice the selection
process never saw — addresses auditor concern #2.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_CSV  = os.path.expanduser("~/GoldBigBrain/data/EURUSD_M5_full.csv")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_holdout.json")
H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
EUR_PIP=0.0001; SPREAD=0.7*EUR_PIP
SPLIT = 0.70  # 70/30

PARAMS = {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.03,
          'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
          'min_child_samples':200,'lambda_l1':0.1,'lambda_l2':0.1,'verbose':-1,'num_threads':3}

def tb_label_TRUE(close, atr, th, tl, h, sl_m, tp_m):
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

    n=len(Xs); split_idx=int(n*SPLIT)
    Xtr=Xs.iloc[:split_idx-(H_BARS+5)]; ytr=ys.iloc[:split_idx-(H_BARS+5)]  # embargo
    Xte=Xs.iloc[split_idx:]; yte=ys.iloc[split_idx:]
    cl_te=cl_s.iloc[split_idx:].values; at_te=at_s.iloc[split_idx:].values
    hi_te=hi_s.iloc[split_idx:].values; lo_te=lo_s.iloc[split_idx:].values
    print(f"Train: {ytr.index[0]} -> {ytr.index[-1]} ({len(ytr):,})")
    print(f"Test:  {yte.index[0]} -> {yte.index[-1]} ({len(yte):,})")
    months_test = (yte.index[-1] - yte.index[0]).days / 30.44

    d=lgb.Dataset(Xtr,label=ytr)
    m=lgb.train(PARAMS,d,300,callbacks=[lgb.log_evaluation(0)])
    p=m.predict(Xte)
    auc=roc_auc_score(yte,p)
    print(f"Held-out AUC: {auc:.4f}")

    sig=p>THR; nt=int(sig.sum())
    print(f"Trades at thr={THR}: {nt} ({nt/months_test:.1f}/mo)")
    if nt==0:
        print("0 trades — selection-bias killer"); return
    e=cl_te[sig]; ae=at_te[sig]; he=hi_te[sig]; le=lo_te[sig]
    tp_px=e+TP_MULT*ae; sl_px=e-SL_MULT*ae
    tph=(he>=tp_px); slh=(le<=sl_px)
    wins=tph&~slh; losses=slh&~tph; ambig=tph&slh; timeout=~tph&~slh
    pnl=np.zeros(nt)
    pnl[wins]=TP_MULT*ae[wins]; pnl[losses|ambig]=-SL_MULT*ae[losses|ambig]
    pnl[timeout]=(he[timeout]+le[timeout])/2 - e[timeout]; pnl-=SPREAD
    pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
    wr=int(wins.sum())/nt
    pips=pnl.sum()/EUR_PIP
    print(f"\nHELD-OUT (last 30% never seen by 192-grid selection):")
    print(f"  PF={pf:.3f}  WR={wr:.3f}  pips={pips:.1f}  trades/mo={nt/months_test:.1f}")
    print(f"  W/L/A/T: {int(wins.sum())}/{int(losses.sum())}/{int(ambig.sum())}/{int(timeout.sum())}")
    if pf>=1.5: verdict="STRONG — selection bias not the dominant issue"
    elif pf>=1.0: verdict="WEAK PROFITABLE — some bias but not catastrophic"
    else: verdict="FAIL — selection bias killed it on truly held-out data"
    print(f"\n*** {verdict} ***")
    out={'auc':float(auc),'pf':float(pf),'wr':float(wr),'pips':float(pips),
         'n_trades':nt,'trades_per_month':float(nt/months_test),
         'wins':int(wins.sum()),'losses':int(losses.sum()),
         'ambig':int(ambig.sum()),'timeout':int(timeout.sum()),
         'verdict':verdict,
         'train_start':str(ytr.index[0]),'train_end':str(ytr.index[-1]),
         'test_start':str(yte.index[0]),'test_end':str(yte.index[-1])}
    os.makedirs(os.path.dirname(OUT_JSON),exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out,fh,indent=2)
    print(f"Saved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
