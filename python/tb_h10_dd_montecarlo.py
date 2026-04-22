#!/usr/bin/env python3
"""Drawdown Monte Carlo on h=10/sl=0.7/tp=2.0 trade sequence.
1000 random reorderings of OOF trade PnLs -> measure max DD distribution.
Goal: confirm typical/p95/p99 DD < 25% sub-friendly threshold.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_h10_dd_mc.json")
H_BARS, SL_MULT, TP_MULT, THR = 10, 0.7, 2.0, 0.44
EUR_PIP=0.0001; SPREAD=0.7*EUR_PIP
RISK_PCT = 0.006  # 0.6% per trade
INIT_EQUITY = 1000  # $1k account (Vantage live target)

PARAMS = {'objective':'binary','metric':'auc','num_leaves':31,'learning_rate':0.03,
          'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
          'min_child_samples':200,'lambda_l1':0.1,'lambda_l2':0.1,'verbose':-1,'num_threads':3}

def tb_label(close, atr, h, sl_m, tp_m):
    n=len(close); l=np.zeros(n,dtype=np.int8); cl=close.values; at=atr.values
    for i in range(n-h):
        tp_px=cl[i]+tp_m*at[i]; sl_px=cl[i]-sl_m*at[i]
        for j in range(1,h+1):
            if cl[i+j]>=tp_px: l[i]=1; break
            if cl[i+j]<=sl_px: l[i]=0; break
    return pd.Series(l, index=close.index)

def main():
    t0=time.time()
    f=pd.read_parquet(FEATURES); close=f['close']; atr=f['atr14_norm']*close
    fc=[c for c in f.columns if not c.startswith('target_') and c!='close']
    hi=close.rolling(H_BARS).max().shift(-H_BARS); lo=close.rolling(H_BARS).min().shift(-H_BARS)
    y=tb_label(close,atr,H_BARS,SL_MULT,TP_MULT)
    mask=y.notna()&f[fc].notna().all(axis=1)&atr.notna()&hi.notna()&lo.notna()
    Xs=f.loc[mask,fc]; ys=y[mask]
    cl_s=close[mask].values; at_s=atr[mask].values; hi_s=hi[mask].values; lo_s=lo[mask].values

    tscv=TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
    oof=pd.Series(np.nan, index=ys.index)
    for tr,va in tscv.split(Xs):
        if len(tr)>H_BARS+5: tr=tr[:-(H_BARS+5)]
        d=lgb.Dataset(Xs.iloc[tr], label=ys.iloc[tr])
        dv=lgb.Dataset(Xs.iloc[va], label=ys.iloc[va], reference=d)
        m=lgb.train(PARAMS, d, 300, valid_sets=[dv], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        oof.iloc[va]=m.predict(Xs.iloc[va])

    valid=oof.notna()
    cl_v=cl_s[valid.values]; at_v=at_s[valid.values]; hi_v=hi_s[valid.values]; lo_v=lo_s[valid.values]
    oof_v=oof[valid].values
    sig=oof_v>THR; n=int(sig.sum())
    e=cl_v[sig]; ae=at_v[sig]; he=hi_v[sig]; le=lo_v[sig]
    tp_px=e+TP_MULT*ae; sl_px=e-SL_MULT*ae
    tph=(he>=tp_px); slh=(le<=sl_px)
    wins=tph&~slh; losses=slh&~tph; ambig=tph&slh; timeout=~tph&~slh
    # Convert each trade PnL to RR (in units of risk)
    rr = np.zeros(n)
    rr[wins] = TP_MULT/SL_MULT  # ~2.857 R per win
    rr[losses|ambig] = -1.0  # -1 R per loss
    rr[timeout] = ((he[timeout]+le[timeout])/2 - e[timeout]) / (SL_MULT*ae[timeout])  # fractional R
    # subtract spread in R units (per-trade, only for selected signals)
    rr -= SPREAD / (SL_MULT * ae)

    # Simulate equity curve with compounding 0.6% risk
    rng = np.random.default_rng(42)
    n_sim = 1000
    max_dds = []
    final_equities = []
    for sim in range(n_sim):
        order = rng.permutation(n) if sim > 0 else np.arange(n)  # sim 0 = real order
        eq = INIT_EQUITY
        peak = eq
        max_dd = 0
        for k in order:
            risk_dollars = eq * RISK_PCT
            pnl = rr[k] * risk_dollars
            eq += pnl
            if eq > peak: peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd: max_dd = dd
        max_dds.append(max_dd); final_equities.append(eq)

    max_dds = np.array(max_dds); final_equities = np.array(final_equities)
    real_dd = max_dds[0]
    real_eq = final_equities[0]
    typical_dd = np.percentile(max_dds, 50)
    p95_dd = np.percentile(max_dds, 95)
    p99_dd = np.percentile(max_dds, 99)
    print(f"\nMonte Carlo DD ({n_sim} reorderings, {n} trades, 0.6% risk, $1k init):")
    print(f"  Real-order DD: {real_dd*100:.2f}%, final equity ${real_eq:.0f}")
    print(f"  Typical (p50) DD: {typical_dd*100:.2f}%")
    print(f"  P95 DD: {p95_dd*100:.2f}%")
    print(f"  P99 DD: {p99_dd*100:.2f}%")
    print(f"  Final equity p5/p50/p95: ${np.percentile(final_equities,5):.0f} / ${np.percentile(final_equities,50):.0f} / ${np.percentile(final_equities,95):.0f}")
    print(f"  Sub-friendly gate (p95 < 25%): {'PASS' if p95_dd<0.25 else 'FAIL'}")
    print(f"  Sub-friendly gate (p99 < 25%): {'PASS' if p99_dd<0.25 else 'FAIL'}")

    out={'n_trades':n, 'init_equity':INIT_EQUITY, 'risk_pct':RISK_PCT,
         'real_dd_pct':float(real_dd*100), 'real_final':float(real_eq),
         'typical_dd_pct':float(typical_dd*100), 'p95_dd_pct':float(p95_dd*100),
         'p99_dd_pct':float(p99_dd*100),
         'final_equity_p5':float(np.percentile(final_equities,5)),
         'final_equity_p50':float(np.percentile(final_equities,50)),
         'final_equity_p95':float(np.percentile(final_equities,95)),
         'pass_p95_25pct':bool(p95_dd<0.25), 'pass_p99_25pct':bool(p99_dd<0.25)}
    os.makedirs(os.path.dirname(OUT_JSON),exist_ok=True)
    with open(OUT_JSON,'w') as fh: json.dump(out,fh,indent=2)
    print(f"Saved {OUT_JSON}, {time.time()-t0:.1f}s")

if __name__=="__main__": main()
