#!/usr/bin/env python3
"""TB winner sweet-spot exploration:
1. Finer threshold grid (0.27..0.40 step 0.01) on h=5/SL=1/TP=2
2. Test alt TPs (2.5, 3.0) and alt SLs (0.7, 0.8) at h=5
3. Per-fold + per-year breakdown
Target: PF >= 1.5 at >= 20 trades/month (MQL5 floor)
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
OUT_JSON = os.path.expanduser("~/EuroBigBrain/runs/tb_sweet_spot.json")
EUR_PIP = 0.0001
SPREAD = 0.7 * EUR_PIP
H_BARS = 5

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 2}

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
    f = pd.read_parquet(FEATURES)
    close = f['close']
    atr = f['atr14_norm'] * close
    feat_cols = [c for c in f.columns if not c.startswith('target_') and c != 'close']
    high_proxy = close.rolling(H_BARS).max().shift(-H_BARS)
    low_proxy  = close.rolling(H_BARS).min().shift(-H_BARS)

    results = {}
    for sl_m, tp_m in [(1.0, 2.0), (1.0, 2.5), (1.0, 3.0), (0.8, 2.0), (0.7, 1.5)]:
        cfg = f'sl{sl_m}_tp{tp_m}'
        print(f"\n=== {cfg} ===")
        y = tb_label(close, atr, H_BARS, sl_m, tp_m)
        mask = y.notna() & f[feat_cols].notna().all(axis=1) & atr.notna() & high_proxy.notna() & low_proxy.notna()
        Xs = f.loc[mask, feat_cols]; ys = y[mask]
        cl_s = close[mask]; at_s = atr[mask]; hi_s = high_proxy[mask]; lo_s = low_proxy[mask]
        n = len(Xs); embargo = H_BARS + 5
        test_size = n // 7
        tscv = TimeSeriesSplit(n_splits=5, test_size=test_size)
        oof = pd.Series(np.nan, index=ys.index)
        fold_aucs = []
        for tr, va in tscv.split(Xs):
            if embargo > 0 and len(tr) > embargo: tr = tr[:-embargo]
            Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
            ytr, yva = ys.iloc[tr], ys.iloc[va]
            d_tr = lgb.Dataset(Xtr, label=ytr)
            d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
            m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            p = m.predict(Xva)
            oof.iloc[va] = p
            fold_aucs.append(roc_auc_score(yva, p))
        print(f"  AUC={np.mean(fold_aucs):.4f}, balance={ys.mean():.3f}")

        # Sweep
        valid = oof.notna()
        cl_v = cl_s[valid]; at_v = at_s[valid]; hi_v = hi_s[valid]; lo_v = lo_s[valid]; oof_v = oof[valid]
        oof_idx = oof_v.index
        # Months span for trades/mo calculation
        months = (oof_idx[-1] - oof_idx[0]).days / 30.44
        sweep = []
        for thr in np.arange(0.20, 0.50, 0.01):
            sig = oof_v > thr
            ntr = int(sig.sum())
            if ntr < 30: continue
            entry = cl_v[sig]; ae = at_v[sig]; he = hi_v[sig]; le = lo_v[sig]
            tp_px = entry + tp_m * ae; sl_px = entry - sl_m * ae
            tp_hit = (he >= tp_px); sl_hit = (le <= sl_px)
            wins = tp_hit & ~sl_hit
            losses = sl_hit & ~tp_hit
            ambig = tp_hit & sl_hit
            timeout = ~tp_hit & ~sl_hit
            pnl_w = (tp_m * ae[wins]).sum()
            pnl_l = -(sl_m * ae[losses | ambig]).sum()
            pnl_t = ((he[timeout] + le[timeout])/2 - entry[timeout]).sum()
            net = pnl_w + pnl_l + pnl_t - SPREAD * ntr
            gw = pnl_w; gl = abs(pnl_l) + SPREAD * ntr
            pf = gw/gl if gl > 0 else float('inf')
            sweep.append({
                'thr': round(float(thr), 2), 'n_trades': ntr,
                'trades_per_month': round(ntr/months, 1),
                'win_rate': round(int(wins.sum())/ntr, 3),
                'pf_net': round(pf, 3),
                'pips_per_trade': round(net/ntr/EUR_PIP, 2),
                'total_pips': round(net/EUR_PIP, 1),
            })

        # Yearly breakdown for the best PF row with >= 60 trades
        best = max([r for r in sweep if r['n_trades'] >= 60], key=lambda r: r['pf_net'], default=None)
        per_year = {}
        if best:
            sig = oof_v > best['thr']
            entry = cl_v[sig]; ae = at_v[sig]; he = hi_v[sig]; le = lo_v[sig]
            idx_sig = oof_idx[sig]
            tp_px = entry + tp_m * ae; sl_px = entry - sl_m * ae
            tp_hit = (he >= tp_px); sl_hit = (le <= sl_px)
            wins_arr = (tp_hit & ~sl_hit).values
            losses_arr = (sl_hit & ~tp_hit).values
            ambig_arr = (tp_hit & sl_hit).values
            timeout_arr = (~tp_hit & ~sl_hit).values
            for yr in idx_sig.year.unique():
                ymask = idx_sig.year == yr
                if ymask.sum() < 10: continue
                pnl_w_y = (tp_m * ae.values[ymask & wins_arr]).sum()
                pnl_l_y = -(sl_m * ae.values[ymask & (losses_arr | ambig_arr)]).sum()
                pnl_t_y = ((he.values[ymask & timeout_arr] + le.values[ymask & timeout_arr])/2 - entry.values[ymask & timeout_arr]).sum()
                ntr_y = int(ymask.sum())
                net_y = pnl_w_y + pnl_l_y + pnl_t_y - SPREAD * ntr_y
                gw_y = pnl_w_y; gl_y = abs(pnl_l_y) + SPREAD * ntr_y
                pf_y = gw_y/gl_y if gl_y > 0 else float('inf')
                per_year[int(yr)] = {'n_trades': ntr_y, 'pf': round(pf_y, 3),
                                     'pips': round(net_y/EUR_PIP, 1)}

        results[cfg] = {
            'sl_mult': sl_m, 'tp_mult': tp_m, 'mean_auc': float(np.mean(fold_aucs)),
            'target_balance': float(ys.mean()), 'sweep': sweep,
            'best_60plus': best, 'per_year_at_best': per_year,
        }
        if best:
            print(f"  Best (n>=60): thr={best['thr']} n={best['n_trades']} ({best['trades_per_month']}/mo) "
                  f"WR={best['win_rate']} PF={best['pf_net']} pips={best['total_pips']}")
            for yr, m in sorted(per_year.items()):
                print(f"    {yr}: n={m['n_trades']} PF={m['pf']} pips={m['pips']}")

    print("\n=== SUMMARY (best n>=60 each cfg) ===")
    print(f"{'cfg':<14} {'AUC':>6} {'thr':>5} {'n':>5} {'/mo':>5} {'WR':>5} {'PF':>6} {'pips':>7}")
    for cfg, r in results.items():
        b = r['best_60plus']
        if b:
            print(f"{cfg:<14} {r['mean_auc']:>6.4f} {b['thr']:>5} {b['n_trades']:>5} "
                  f"{b['trades_per_month']:>5} {b['win_rate']:>5.3f} {b['pf_net']:>6.3f} {b['total_pips']:>7.1f}")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh: json.dump(results, fh, indent=2)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
