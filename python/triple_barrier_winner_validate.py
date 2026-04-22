#!/usr/bin/env python3
"""TB winner (h=5, SL=1.0, TP=2.0, AUC=0.6362) full validation:
1. Null permutation test (HARD RULE: real >= 5x null edge gate)
2. OOF-pred + threshold sweep -> PF estimate
3. Per-fold AUC + per-fold null AUC table
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_winner_validate.json")

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 2}

H_BARS, SL_MULT, TP_MULT = 5, 1.0, 2.0
EUR_PIP = 0.0001
SPREAD = 0.7 * EUR_PIP

def triple_barrier_label(close, atr, h_bars, sl_mult, tp_mult):
    n = len(close)
    label = np.zeros(n, dtype=np.int8)
    cl = close.values; at = atr.values
    for i in range(n - h_bars):
        tp_px = cl[i] + tp_mult * at[i]
        sl_px = cl[i] - sl_mult * at[i]
        for j in range(1, h_bars + 1):
            if cl[i+j] >= tp_px: label[i] = 1; break
            if cl[i+j] <= sl_px: label[i] = 0; break
    return pd.Series(label, index=close.index)

def main():
    t0 = time.time()
    f = pd.read_parquet(FEATURES)
    close = f['close']
    high = close.rolling(H_BARS).max().shift(-H_BARS)  # close-only proxy fwd extreme
    low  = close.rolling(H_BARS).min().shift(-H_BARS)
    atr = f['atr14_norm'] * close
    feat_cols = [c for c in f.columns if not c.startswith('target_') and c != 'close']

    print(f"Building TB target h={H_BARS}, SL={SL_MULT}, TP={TP_MULT}...")
    y = triple_barrier_label(close, atr, H_BARS, SL_MULT, TP_MULT)
    mask = y.notna() & f[feat_cols].notna().all(axis=1) & atr.notna() & high.notna() & low.notna()
    Xs = f.loc[mask, feat_cols]; ys = y[mask]
    cl_s = close[mask]; at_s = atr[mask]; hi_s = high[mask]; lo_s = low[mask]
    print(f"rows={len(Xs):,}, balance={ys.mean():.3f}")

    n_folds = 5
    embargo = H_BARS + 5
    test_size = len(Xs)//(n_folds+2)
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=test_size)

    rng = np.random.default_rng(42)
    y_null = pd.Series(rng.permutation(ys.values), index=ys.index)

    oof_pred = pd.Series(np.nan, index=ys.index)
    fold_aucs = []
    null_aucs = []
    for fold, (tr, va) in enumerate(tscv.split(Xs)):
        if embargo > 0 and len(tr) > embargo:
            tr = tr[:-embargo]
        Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
        ytr, yva = ys.iloc[tr], ys.iloc[va]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        p = m.predict(Xva)
        oof_pred.iloc[va] = p
        a = roc_auc_score(yva, p)
        fold_aucs.append(a)

        ytrn, yvan = y_null.iloc[tr], y_null.iloc[va]
        d_trn = lgb.Dataset(Xtr, label=ytrn)
        d_van = lgb.Dataset(Xva, label=yvan, reference=d_trn)
        mn = lgb.train(PARAMS, d_trn, num_boost_round=300, valid_sets=[d_van],
                       callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        null_aucs.append(roc_auc_score(yvan, mn.predict(Xva)))
        print(f"  Fold {fold}: AUC={a:.4f}  null={null_aucs[-1]:.4f}")

    real_auc = float(np.mean(fold_aucs))
    null_auc = float(np.mean(null_aucs))
    edge = real_auc - 0.5
    null_edge = null_auc - 0.5
    ratio = edge / null_edge if null_edge > 0 else float('inf')
    print(f"\n[REAL] AUC = {real_auc:.4f} +- {np.std(fold_aucs):.4f}")
    print(f"[NULL] AUC = {null_auc:.4f} +- {np.std(null_aucs):.4f}")
    print(f"[EDGE_RATIO] real_edge / null_edge = {ratio:.2f}x  (HARD RULE gate >= 5x)")
    null_pass = ratio >= 5.0
    print(f"[NULL_GATE] {'PASS' if null_pass else 'FAIL'}")

    valid = oof_pred.notna()
    cl_v = cl_s[valid]; at_v = at_s[valid]; hi_v = hi_s[valid]; lo_v = lo_s[valid]
    oof_v = oof_pred[valid]

    sweep = []
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        sig = oof_v > thr
        n = int(sig.sum())
        if n < 50: continue
        entry = cl_v[sig]; atr_e = at_v[sig]; hi_e = hi_v[sig]; lo_e = lo_v[sig]
        tp_px = entry + TP_MULT * atr_e
        sl_px = entry - SL_MULT * atr_e
        tp_hit = (hi_e >= tp_px); sl_hit = (lo_e <= sl_px)
        wins = tp_hit & ~sl_hit
        losses = sl_hit & ~tp_hit
        ambiguous = tp_hit & sl_hit  # both in same window -> assume SL first (conservative)
        timeout = ~tp_hit & ~sl_hit
        pnl_w = (TP_MULT * atr_e[wins]).sum()
        pnl_l = -(SL_MULT * atr_e[losses | ambiguous]).sum()
        pnl_t = ((hi_e[timeout] + lo_e[timeout])/2 - entry[timeout]).sum()
        gross = pnl_w + pnl_l + pnl_t
        spread_cost = SPREAD * n
        net = gross - spread_cost
        gw = pnl_w
        gl = abs(pnl_l) + spread_cost
        pf = gw/gl if gl > 0 else float('inf')
        wr = int(wins.sum())/n
        sweep.append({
            'threshold': thr, 'n_trades': n,
            'wins': int(wins.sum()), 'losses': int(losses.sum()),
            'ambiguous': int(ambiguous.sum()), 'timeout': int(timeout.sum()),
            'win_rate': round(wr, 4), 'pf_net': round(pf, 3),
            'avg_per_trade_pips': round(net/n/EUR_PIP, 2),
            'total_pips': round(net/EUR_PIP, 1),
        })
    sweep.sort(key=lambda r: r['pf_net'], reverse=True)
    print(f"\nThreshold sweep (sorted by net PF):")
    print(f"{'thr':>5} {'n':>5} {'WR':>6} {'PF':>6} {'pips':>7} {'tot_pips':>9}")
    for r in sweep:
        print(f"{r['threshold']:>5} {r['n_trades']:>5} {r['win_rate']:>6.3f} {r['pf_net']:>6.3f} "
              f"{r['avg_per_trade_pips']:>7.2f} {r['total_pips']:>9.1f}")

    out = {
        'config': {'h_bars': H_BARS, 'sl_mult': SL_MULT, 'tp_mult': TP_MULT},
        'real_auc': real_auc, 'null_auc': null_auc, 'edge_ratio_to_null': ratio,
        'null_gate_pass': null_pass,
        'fold_aucs': [float(a) for a in fold_aucs],
        'null_fold_aucs': [float(a) for a in null_aucs],
        'sweep': sweep,
        'target_balance': float(ys.mean()), 'n_rows': int(len(Xs)),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(out, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
