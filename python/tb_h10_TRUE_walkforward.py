#!/usr/bin/env python3
"""Walk-forward on h=10/sl=0.7/tp=2.0 with TRUE bar high/low.
Re-runs walk-forward but uses real bar hi/lo for both label generation AND PnL eval.
Should give a more honest per-month picture vs the close-proxy version (which had median PF only 1.054).
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_CSV  = os.path.expanduser("~/GoldBigBrain/data/EURUSD_M5_full.csv")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_TRUE_walkforward.json")
H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
EUR_PIP=0.0001; SPREAD=0.7*EUR_PIP

PARAMS = {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.03,
          'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
          'min_child_samples':200,'lambda_l1':0.1,'lambda_l2':0.1,'verbose':-1,'num_threads':3}

def tb_label_TRUE(close, atr, true_hi, true_lo, h, sl_m, tp_m):
    n=len(close); l=np.zeros(n,dtype=np.int8); cl=close.values; at=atr.values
    th=true_hi.values; tl=true_lo.values
    for i in range(n-h):
        tp_px=cl[i]+tp_m*at[i]; sl_px=cl[i]-sl_m*at[i]
        for j in range(1,h+1):
            if tl[i+j]<=sl_px:
                l[i]=0; break
            if th[i+j]>=tp_px:
                l[i]=1; break
    return pd.Series(l, index=close.index)

def main():
    t0=time.time()
    f=pd.read_parquet(FEATURES); raw=pd.read_csv(RAW_CSV)
    raw['time']=pd.to_datetime(raw['time']); raw=raw.set_index('time').sort_index()
    common = f.index.intersection(raw.index)
    f=f.loc[common]; raw=raw.loc[common]
    close=f['close']; atr=f['atr14_norm']*close
    true_hi=raw['high']; true_lo=raw['low']
    fc=[c for c in f.columns if not c.startswith('target_') and c!='close']
    y = tb_label_TRUE(close, atr, true_hi, true_lo, H_BARS, SL_MULT, TP_MULT)
    fwd_hi = true_hi.rolling(H_BARS).max().shift(-H_BARS)
    fwd_lo = true_lo.rolling(H_BARS).min().shift(-H_BARS)
    mask = y.notna() & f[fc].notna().all(axis=1) & atr.notna() & fwd_hi.notna() & fwd_lo.notna()
    Xs=f.loc[mask,fc]; ys=y[mask]
    cl_s=close[mask]; at_s=atr[mask]
    fwd_hi_s=fwd_hi[mask]; fwd_lo_s=fwd_lo[mask]
    idx=ys.index
    print(f"Data {idx[0]} -> {idx[-1]} ({len(ys):,} rows, {(idx[-1]-idx[0]).days/30.44:.1f}mo)")

    months = pd.date_range(idx[0].normalize(), idx[-1], freq='MS')
    results=[]
    for i in range(6, len(months)):
        tr_start=months[i-6]; tr_end=months[i]
        te_start=months[i]; te_end=months[i+1] if i+1<len(months) else idx[-1]
        trm=(idx>=tr_start)&(idx<tr_end); tem=(idx>=te_start)&(idx<te_end)
        if trm.sum()<1000 or tem.sum()<100: continue
        Xtr=Xs.loc[trm]; ytr=ys.loc[trm]
        Xte=Xs.loc[tem]; yte=ys.loc[tem]
        d=lgb.Dataset(Xtr,label=ytr)
        m=lgb.train(PARAMS,d,200,callbacks=[lgb.log_evaluation(0)])
        p=m.predict(Xte)
        try: auc=roc_auc_score(yte,p)
        except: auc=float('nan')
        sig=p>THR; n=int(sig.sum())
        if n==0:
            results.append({'month':str(te_start.date()),'auc':float(auc),'n':0,'pf':None,'pips':0}); continue
        e=cl_s[tem].values[sig]; ae=at_s[tem].values[sig]
        he=fwd_hi_s[tem].values[sig]; le=fwd_lo_s[tem].values[sig]
        tp_px=e+TP_MULT*ae; sl_px=e-SL_MULT*ae
        tph=(he>=tp_px); slh=(le<=sl_px)
        wins=tph&~slh; losses=slh&~tph; ambig=tph&slh; timeout=~tph&~slh
        pnl=np.zeros(n)
        pnl[wins]=TP_MULT*ae[wins]; pnl[losses|ambig]=-SL_MULT*ae[losses|ambig]
        pnl[timeout]=(he[timeout]+le[timeout])/2 - e[timeout]; pnl-=SPREAD
        pf=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum()) if (pnl<0).any() else float('inf')
        wr=int(wins.sum())/n
        results.append({'month':str(te_start.date()),'auc':float(auc),'n':n,
                        'wr':round(wr,3),'pf':round(pf,3),'pips':round(pnl.sum()/EUR_PIP,1)})
        print(f"  {te_start.date()}: AUC={auc:.4f} n={n} WR={wr:.3f} PF={pf:.3f} pips={pnl.sum()/EUR_PIP:.1f}")

    pf_vals=[r['pf'] for r in results if r['pf'] is not None]
    pos=sum(1 for r in results if r['pf'] is not None and r['pf']>=1.0)
    print(f"\nTRUE-hilo walk-forward: {len(results)} mo, {pos} profitable")
    print(f"PF range {min(pf_vals):.3f} -> {max(pf_vals):.3f}, median {np.median(pf_vals):.3f}")
    out={'config':{'h':H_BARS,'sl':SL_MULT,'tp':TP_MULT,'thr':THR},
         'n_months':len(results),'profitable_months':pos,
         'pf_min':float(min(pf_vals)),'pf_max':float(max(pf_vals)),'pf_median':float(np.median(pf_vals)),
         'monthly':results}
    os.makedirs(os.path.dirname(OUT_JSON),exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out,fh,indent=2)
    print(f"Saved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
