#!/usr/bin/env python3
"""TRUE bar high/low threshold sweep on h=10/sl=0.7/tp=2.0.
Find the (PF, trades/mo) frontier with REAL bar high/low (not close-proxy).
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_CSV  = os.path.expanduser("~/GoldBigBrain/data/EURUSD_M5_full.csv")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_TRUE_thr.json")
H_BARS, SL_MULT, TP_MULT = 10, 0.7, 2.0
EUR_PIP=0.0001; SPREAD=0.7*EUR_PIP

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
    cl_s=close[mask].values; at_s=atr[mask].values
    hi_s=fwd_hi[mask].values; lo_s=fwd_lo[mask].values
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
    oof_v=oof[valid].values; idx_v=oof[valid].index
    months=(idx_v[-1]-idx_v[0]).days/30.44
    rows=[]
    for thr in np.arange(0.20,0.60,0.01):
        sig=oof_v>thr; n=int(sig.sum())
        if n<30: continue
        e=cl_v[sig]; ae=at_v[sig]; he=hi_v[sig]; le=lo_v[sig]
        tp_px=e+TP_MULT*ae; sl_px=e-SL_MULT*ae
        tph=(he>=tp_px); slh=(le<=sl_px)
        wins=tph&~slh; losses=slh&~tph; ambig=tph&slh; timeout=~tph&~slh
        pnl=np.zeros(n)
        pnl[wins]=TP_MULT*ae[wins]; pnl[losses|ambig]=-SL_MULT*ae[losses|ambig]
        pnl[timeout]=(he[timeout]+le[timeout])/2 - e[timeout]; pnl-=SPREAD
        pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
        rows.append({'thr':round(float(thr),2),'n':n,'tpm':round(n/months,1),
                     'wr':round(int(wins.sum())/n,3),'pf':round(pf,3),
                     'pips':round(pnl.sum()/EUR_PIP,1),'p_t':round(pnl.sum()/n/EUR_PIP,2)})
    rows.sort(key=lambda r:-r['pf'])
    print('top10 by PF (TRUE hi/lo):')
    for r in rows[:10]: print(f"  thr={r['thr']} n={r['n']} tpm={r['tpm']} WR={r['wr']} PF={r['pf']} pips={r['pips']} p_t={r['p_t']}")
    deploy=[r for r in rows if r['pf']>=1.3 and r['tpm']>=20]
    if deploy:
        b=max(deploy,key=lambda r:r['tpm'])
        print(f"\nDEPLOY-FRONTIER (PF>=1.3 + tpm>=20): thr={b['thr']} PF={b['pf']} tpm={b['tpm']} pips_per_trade={b['p_t']}")
    out={'config':{'h':H_BARS,'sl':SL_MULT,'tp':TP_MULT},'months':months,'sweep':rows}
    os.makedirs(os.path.dirname(OUT_JSON),exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out,fh,indent=2)
    print(f"Saved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
