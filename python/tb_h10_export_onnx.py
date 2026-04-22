#!/usr/bin/env python3
"""Train final h=10/sl=0.7/tp=2.0 LightGBM on FULL 16mo with TRUE labels.
Export as ONNX for MQL5 EA. This is the model that will run live.
"""
import os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_CSV  = os.path.expanduser("~/GoldBigBrain/data/EURUSD_M5_full.csv")
OUT_ONNX = os.path.expanduser("~/EuroBigBrain/models/eur_tb_h10.onnx")
OUT_TXT  = os.path.expanduser("~/EuroBigBrain/models/eur_tb_h10_features.txt")
H_BARS, SL_MULT, TP_MULT = 10, 0.7, 2.0

PARAMS = {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.03,
          'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
          'min_child_samples':200,'lambda_l1':0.1,'lambda_l2':0.1,'verbose':-1,'num_threads':4}

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
    close=f['close']; atr=f['atr14_norm']*close
    fc=[c for c in f.columns if not c.startswith('target_') and c!='close']
    y=tb_label_TRUE(close,atr,raw['high'],raw['low'],H_BARS,SL_MULT,TP_MULT)
    fwd_hi=raw['high'].rolling(H_BARS).max().shift(-H_BARS)
    fwd_lo=raw['low'].rolling(H_BARS).min().shift(-H_BARS)
    mask=y.notna()&f[fc].notna().all(axis=1)&atr.notna()&fwd_hi.notna()&fwd_lo.notna()
    Xs=f.loc[mask,fc]; ys=y[mask]
    print(f"Train final on {len(Xs):,} rows, {len(fc)} features, balance {ys.mean():.3f}")
    d=lgb.Dataset(Xs,label=ys)
    m=lgb.train(PARAMS,d,num_boost_round=300,callbacks=[lgb.log_evaluation(0)])
    print(f"Trained {m.num_trees()} trees")
    # Save feature list
    os.makedirs(os.path.dirname(OUT_ONNX),exist_ok=True)
    with open(OUT_TXT,'w') as fh: fh.write('\n'.join(fc))
    print(f"Saved feature list: {OUT_TXT}")
    # Convert to ONNX
    initial_types=[('input', FloatTensorType([None, len(fc)]))]
    onnx_model = convert_lightgbm(m, initial_types=initial_types, target_opset=12)
    with open(OUT_ONNX,'wb') as fh: fh.write(onnx_model.SerializeToString())
    sz=os.path.getsize(OUT_ONNX)
    print(f"Saved ONNX: {OUT_ONNX} ({sz} bytes)")
    print(f"Total {time.time()-t0:.1f}s")

if __name__=="__main__": main()
