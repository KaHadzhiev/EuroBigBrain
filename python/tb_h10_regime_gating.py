#!/usr/bin/env python3
"""Regime gating exploration: test if filtering by VOLATILITY regime improves PF.
Test 4 regime filters on h=10/sl=0.7/tp=2.0 OOF predictions:
- low_vol: atr14_norm < median (calm)
- high_vol: atr14_norm > median (active)
- london+ny: hour 7-21 GMT
- avoid_news: exclude Wed 13-15 GMT (FOMC/ECB common)
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_CSV  = os.path.expanduser("~/GoldBigBrain/data/EURUSD_M5_full.csv")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_regime.json")
H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
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
    close=f['close']; atr_n=f['atr14_norm']; atr=atr_n*close
    fc=[c for c in f.columns if not c.startswith('target_') and c!='close']
    y=tb_TRUE(close,atr,raw['high'],raw['low'],H_BARS,SL_MULT,TP_MULT)
    fwd_hi=raw['high'].rolling(H_BARS).max().shift(-H_BARS)
    fwd_lo=raw['low'].rolling(H_BARS).min().shift(-H_BARS)
    mask=y.notna()&f[fc].notna().all(axis=1)&atr.notna()&fwd_hi.notna()&fwd_lo.notna()
    Xs=f.loc[mask,fc]; ys=y[mask]
    cl_s=close[mask].values; at_s=atr[mask].values; hi_s=fwd_hi[mask].values; lo_s=fwd_lo[mask].values
    atr_n_s=atr_n[mask]; idx=ys.index
    tscv=TimeSeriesSplit(n_splits=5,test_size=len(Xs)//7)
    oof=pd.Series(np.nan,index=ys.index)
    for tr,va in tscv.split(Xs):
        if len(tr)>H_BARS+5: tr=tr[:-(H_BARS+5)]
        d=lgb.Dataset(Xs.iloc[tr],label=ys.iloc[tr])
        dv=lgb.Dataset(Xs.iloc[va],label=ys.iloc[va],reference=d)
        m=lgb.train(PARAMS,d,300,valid_sets=[dv],callbacks=[lgb.early_stopping(20),lgb.log_evaluation(0)])
        oof.iloc[va]=m.predict(Xs.iloc[va])
    valid=oof.notna()
    cl_v=cl_s[valid.values]; at_v=at_s[valid.values]; hi_v=hi_s[valid.values]; lo_v=lo_s[valid.values]
    oof_v=oof[valid].values; idx_v=idx[valid]; atr_n_v=atr_n_s[valid].values
    months=(idx_v[-1]-idx_v[0]).days/30.44
    median_atr=np.median(atr_n_v)
    hours=idx_v.hour; dows=idx_v.dayofweek
    base_sig=oof_v>THR
    regimes={
        'baseline_no_filter': np.ones(len(oof_v),dtype=bool),
        'low_vol_only': atr_n_v<median_atr,
        'high_vol_only': atr_n_v>=median_atr,
        'london_ny_only': (hours>=7)&(hours<=21),
        'avoid_wed_news': ~((dows==2)&(hours>=13)&(hours<=15)),
        'low_vol+london': (atr_n_v<median_atr)&(hours>=7)&(hours<=21),
        'high_vol+london': (atr_n_v>=median_atr)&(hours>=7)&(hours<=21),
    }
    results={}
    for name,gate in regimes.items():
        sig=base_sig&gate; n=int(sig.sum())
        if n<30: continue
        e=cl_v[sig]; ae=at_v[sig]; he=hi_v[sig]; le=lo_v[sig]
        tp_px=e+TP_MULT*ae; sl_px=e-SL_MULT*ae
        tph=(he>=tp_px); slh=(le<=sl_px)
        wins=tph&~slh; losses=slh&~tph; ambig=tph&slh; timeout=~tph&~slh
        pnl=np.zeros(n)
        pnl[wins]=TP_MULT*ae[wins]; pnl[losses|ambig]=-SL_MULT*ae[losses|ambig]
        pnl[timeout]=(he[timeout]+le[timeout])/2-e[timeout]; pnl-=SPREAD
        pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
        wr=int(wins.sum())/n
        results[name]={'n':n,'tpm':round(n/months,1),'wr':round(wr,3),'pf':round(float(pf),3),
                        'pips':round(pnl.sum()/EUR_PIP,1)}
        print(f"{name:28s} n={n:4d} tpm={n/months:5.1f} WR={wr:.3f} PF={pf:.3f} pips={pnl.sum()/EUR_PIP:7.1f}")
    os.makedirs(os.path.dirname(OUT_JSON),exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(results,fh,indent=2)
    print(f"\nSaved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
