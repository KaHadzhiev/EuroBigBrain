#!/usr/bin/env python3
"""TB winner config (h=5, SL=0.7, TP=1.5) on 4 other FX/metals on M5.
Tests if EUR edge generalizes or is symbol-specific.
Uses GoldBigBrain raw OHLC CSVs (~16mo each).
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

DATA_DIR = os.path.expanduser("~/GoldBigBrain/data")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_multi_instrument.json")
EUR_PIP_BY_SYM = {'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01, 'USDCHF': 0.0001, 'XAGUSD': 0.001, 'BTCUSD': 1.0, 'CL_OIL': 0.01, 'SP500': 0.1}
SPREAD_BY_SYM = {'EURUSD': 0.7e-4, 'GBPUSD': 1.2e-4, 'USDJPY': 1.2e-2, 'USDCHF': 1.5e-4, 'XAGUSD': 0.025, 'BTCUSD': 30, 'CL_OIL': 0.03, 'SP500': 0.5}

H_BARS, SL_MULT, TP_MULT = 5, 0.7, 1.5

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 2}

def build_features(df, sym):
    df = df.copy()
    df.index = pd.to_datetime(df['time']) if 'time' in df.columns else pd.to_datetime(df.index)
    df = df.sort_index()
    c = df['close']
    df['ret1'] = c.pct_change()
    df['ret5'] = c.pct_change(5)
    df['ret20'] = c.pct_change(20)
    df['atr14'] = (df['high']-df['low']).rolling(14).mean()
    df['atr14_norm'] = df['atr14'] / c
    df['vol_z'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std().replace(0, np.nan)
    df['rsi14'] = 100 - 100/(1 + (c.diff().clip(lower=0).rolling(14).mean() / -c.diff().clip(upper=0).rolling(14).mean().replace(0, np.nan)))
    df['ema20_z'] = (c - c.ewm(span=20).mean()) / c.ewm(span=20).std()
    df['ema50_z'] = (c - c.ewm(span=50).mean()) / c.ewm(span=50).std()
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['range_norm'] = (df['high']-df['low']) / df['atr14'].replace(0, np.nan)
    return df

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
    out = {}
    syms = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAGUSD']

    for sym in syms:
        path = f"{DATA_DIR}/{sym}_M5_full.csv"
        if not os.path.exists(path):
            print(f"SKIP {sym}: no file")
            continue
        df = pd.read_csv(path)
        df = build_features(df, sym)
        feat_cols = ['ret1','ret5','ret20','atr14_norm','vol_z','rsi14','ema20_z','ema50_z','hour','dow','range_norm']
        close = df['close']
        atr = df['atr14_norm'] * close
        hi_proxy = close.rolling(H_BARS).max().shift(-H_BARS)
        lo_proxy = close.rolling(H_BARS).min().shift(-H_BARS)
        y = tb_label(close, atr, H_BARS, SL_MULT, TP_MULT)
        mask = y.notna() & df[feat_cols].notna().all(axis=1) & atr.notna() & hi_proxy.notna() & lo_proxy.notna()
        Xs = df.loc[mask, feat_cols]; ys = y[mask]
        cl_s = close[mask].values; at_s = atr[mask].values; hi_s = hi_proxy[mask].values; lo_s = lo_proxy[mask].values
        bal = float(ys.mean())
        print(f"\n=== {sym} (rows={len(Xs):,}, bal={bal:.3f}) ===")

        tscv = TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
        oof = pd.Series(np.nan, index=ys.index); fold_aucs = []
        for tr, va in tscv.split(Xs):
            if len(tr) > H_BARS+5: tr = tr[:-(H_BARS+5)]
            Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
            ytr, yva = ys.iloc[tr], ys.iloc[va]
            d_tr = lgb.Dataset(Xtr, label=ytr)
            d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
            m = lgb.train(PARAMS, d_tr, num_boost_round=200, valid_sets=[d_va],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            p = m.predict(Xva); oof.iloc[va] = p
            fold_aucs.append(roc_auc_score(yva, p))
        auc = float(np.mean(fold_aucs))
        print(f"  AUC = {auc:.4f}")

        # PnL sweep
        valid = oof.notna()
        cl_v=cl_s[valid.values]; at_v=at_s[valid.values]; hi_v=hi_s[valid.values]; lo_v=lo_s[valid.values]; oof_v=oof[valid].values
        SP = SPREAD_BY_SYM.get(sym, 0.0001); PIP = EUR_PIP_BY_SYM.get(sym, 0.0001)
        best = None
        for thr in np.arange(0.20, 0.55, 0.02):
            sig = oof_v > thr; n = int(sig.sum())
            if n < 60: continue
            entry=cl_v[sig]; ae=at_v[sig]; he=hi_v[sig]; le=lo_v[sig]
            tp_px=entry+TP_MULT*ae; sl_px=entry-SL_MULT*ae
            tp_hit=(he>=tp_px); sl_hit=(le<=sl_px)
            wins=tp_hit&~sl_hit; losses=sl_hit&~tp_hit; ambig=tp_hit&sl_hit; timeout=~tp_hit&~sl_hit
            pnl_w=(TP_MULT*ae[wins]).sum()
            pnl_l=-(SL_MULT*ae[losses|ambig]).sum()
            pnl_t=((he[timeout]+le[timeout])/2 - entry[timeout]).sum()
            net = pnl_w + pnl_l + pnl_t - SP*n
            gw=pnl_w; gl=abs(pnl_l)+SP*n
            pf = gw/gl if gl>0 else float('inf')
            if best is None or pf > best['pf']:
                best = {'thr': round(float(thr),2), 'n': n,
                        'wr': round(int(wins.sum())/n,3), 'pf': round(pf,3),
                        'pips': round(net/PIP,1)}
        out[sym] = {'auc': auc, 'balance': bal, 'best': best, 'rows': int(len(Xs))}
        if best:
            print(f"  Best: thr={best['thr']} n={best['n']} WR={best['wr']} PF={best['pf']} pips={best['pips']}")

    print("\n=== SUMMARY ===")
    print(f"{'sym':<8} {'AUC':>6} {'bal':>5} {'PF':>6} {'n':>5} {'pips':>8}")
    for sym, r in out.items():
        b = r['best']
        if b:
            print(f"{sym:<8} {r['auc']:>6.4f} {r['balance']:>5.3f} {b['pf']:>6} {b['n']:>5} {b['pips']:>8}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(out, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
