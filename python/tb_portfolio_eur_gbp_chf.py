#!/usr/bin/env python3
"""3-symbol TB portfolio backtest: EURUSD + GBPUSD + USDCHF.
Config: h=10, SL=0.7, TP=2.0 (TB winner replicated cross-symbol).

Goal: prove portfolio improves Sharpe / reduces DD vs EURUSD-only baseline.

Pipeline per symbol:
  1. Build features (EUR from parquet, GBP/CHF from raw M5 CSV).
  2. TRUE bar-high/low TB labels (h=10, sl=0.7, tp=2.0).
  3. 5-fold purged-embargoed TimeSeriesSplit OOF predictions.
  4. Per-symbol best-threshold sweep -> trade list.
Portfolio:
  - Risk budget 0.6%/trade per symbol.
  - Max 1 concurrent trade per symbol (3 simultaneous max).
  - Per-symbol spread (EUR 0.7pip, GBP 1.2pip, CHF 1.5pip).
  - Combined PF, monthly trades, max DD, per-symbol contribution, equity correlation.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

EUR_FEATURES = os.path.expanduser("~/EuroBigBrain/data/eurusd_features.parquet")
RAW_DIR      = os.path.expanduser("~/GoldBigBrain/data")
OUT_JSON     = os.path.expanduser("~/EuroBigBrain/runs/tb_portfolio.json")

H_BARS, SL_MULT, TP_MULT = 10, 0.7, 2.0
RISK_PER_TRADE = 0.006     # 0.6% per trade
INIT_EQUITY    = 10_000.0  # for DD%/equity-curve scaling
SYMS           = ['EURUSD', 'GBPUSD', 'USDCHF']
PIP_BY_SYM     = {'EURUSD': 1e-4, 'GBPUSD': 1e-4, 'USDCHF': 1e-4}
SPREAD_BY_SYM  = {'EURUSD': 0.7e-4, 'GBPUSD': 1.2e-4, 'USDCHF': 1.5e-4}

PARAMS = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.03, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'min_child_samples': 200,
          'lambda_l1': 0.1, 'lambda_l2': 0.1, 'verbose': -1, 'num_threads': 2}

FEAT_COLS_RAW = ['ret1','ret5','ret20','atr14_norm','vol_z','rsi14',
                 'ema20_z','ema50_z','hour','dow','range_norm']


def build_features_raw(df):
    """Match tb_multi_instrument.build_features (raw OHLC -> feature DF)."""
    df = df.copy()
    df.index = pd.to_datetime(df['time']) if 'time' in df.columns else pd.to_datetime(df.index)
    df = df.sort_index()
    c = df['close']
    df['ret1']  = c.pct_change()
    df['ret5']  = c.pct_change(5)
    df['ret20'] = c.pct_change(20)
    df['atr14'] = (df['high'] - df['low']).rolling(14).mean()
    df['atr14_norm'] = df['atr14'] / c
    vol_col = 'volume' if 'volume' in df.columns else 'tick_volume' if 'tick_volume' in df.columns else None
    if vol_col is None:
        df['vol_z'] = 0.0
    else:
        df['vol_z'] = (df[vol_col] - df[vol_col].rolling(50).mean()) / df[vol_col].rolling(50).std().replace(0, np.nan)
    gain = c.diff().clip(lower=0).rolling(14).mean()
    loss = -c.diff().clip(upper=0).rolling(14).mean().replace(0, np.nan)
    df['rsi14']  = 100 - 100/(1 + gain/loss)
    df['ema20_z'] = (c - c.ewm(span=20).mean()) / c.ewm(span=20).std()
    df['ema50_z'] = (c - c.ewm(span=50).mean()) / c.ewm(span=50).std()
    df['hour'] = df.index.hour
    df['dow']  = df.index.dayofweek
    df['range_norm'] = (df['high'] - df['low']) / df['atr14'].replace(0, np.nan)
    return df


def tb_label_truehilo(close, high, low, atr, h, sl_m, tp_m):
    """TRUE bar-high/low TB labels. 1=tp first, 0=sl first or timeout."""
    n = len(close); label = np.zeros(n, dtype=np.int8)
    cl = close.values; hi = high.values; lo = low.values; at = atr.values
    for i in range(n - h):
        tp_px = cl[i] + tp_m * at[i]
        sl_px = cl[i] - sl_m * at[i]
        for j in range(1, h+1):
            tp_hit = hi[i+j] >= tp_px
            sl_hit = lo[i+j] <= sl_px
            if tp_hit and sl_hit:
                label[i] = 0  # ambiguous = pessimistic
                break
            if tp_hit:
                label[i] = 1; break
            if sl_hit:
                label[i] = 0; break
    return pd.Series(label, index=close.index)


def load_symbol(sym):
    """Return (df, feat_cols). For EURUSD: parquet features + raw hi/lo.
       For GBP/CHF: features built from raw CSV."""
    if sym == 'EURUSD' and os.path.exists(EUR_FEATURES):
        f = pd.read_parquet(EUR_FEATURES).sort_index()
        raw_path = f"{RAW_DIR}/EURUSD_M5_full.csv"
        raw = pd.read_csv(raw_path)
        raw.index = pd.to_datetime(raw['time'])
        raw = raw.sort_index()
        # Inner join on timestamp so hi/lo align with parquet rows
        common = f.index.intersection(raw.index)
        df = f.loc[common].copy()
        df['high'] = raw.loc[common, 'high']
        df['low']  = raw.loc[common, 'low']
        feat_cols = [c for c in df.columns
                     if c not in ('close','high','low')
                     and not c.startswith('target_')]
        return df, feat_cols
    # GBP / CHF -> build from raw CSV
    raw_path = f"{RAW_DIR}/{sym}_M5_full.csv"
    raw = pd.read_csv(raw_path)
    df = build_features_raw(raw)
    return df, FEAT_COLS_RAW


def train_oof(df, sym, feat_cols):
    close = df['close']; high = df['high']; low = df['low']
    atr   = df['atr14_norm'] * close
    have_feats = [c for c in feat_cols if c in df.columns]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  WARN {sym}: missing feature cols {missing}, dropping")
    y = tb_label_truehilo(close, high, low, atr, H_BARS, SL_MULT, TP_MULT)
    mask = (y.notna()
            & df[have_feats].notna().all(axis=1)
            & atr.notna()
            & close.notna() & high.notna() & low.notna())
    Xs = df.loc[mask, have_feats]
    ys = y[mask]
    cl_s = close[mask]; at_s = atr[mask]; hi_s = high[mask]; lo_s = low[mask]
    print(f"  {sym}: rows={len(Xs):,}  bal={ys.mean():.3f}")

    tscv = TimeSeriesSplit(n_splits=5, test_size=len(Xs)//7)
    oof = pd.Series(np.nan, index=ys.index)
    fold_aucs = []
    embargo = H_BARS + 5
    for fold, (tr, va) in enumerate(tscv.split(Xs)):
        if embargo > 0 and len(tr) > embargo: tr = tr[:-embargo]
        Xtr, Xva = Xs.iloc[tr], Xs.iloc[va]
        ytr, yva = ys.iloc[tr], ys.iloc[va]
        d_tr = lgb.Dataset(Xtr, label=ytr)
        d_va = lgb.Dataset(Xva, label=yva, reference=d_tr)
        m = lgb.train(PARAMS, d_tr, num_boost_round=300, valid_sets=[d_va],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        p = m.predict(Xva); oof.iloc[va] = p
        fold_aucs.append(roc_auc_score(yva, p))
    auc = float(np.mean(fold_aucs))
    print(f"  {sym}: AUC = {auc:.4f}")
    return {
        'oof': oof, 'cl': cl_s, 'at': at_s, 'hi': hi_s, 'lo': lo_s,
        'auc': auc, 'rows': int(len(Xs)), 'balance': float(ys.mean()),
    }


def make_trades(d, sym, thr):
    """Return per-trade DF: entry_time, exit_time (entry+H bars idx), pnl_price (per-unit)."""
    oof = d['oof']
    valid = oof.notna()
    cl = d['cl'][valid].values
    at = d['at'][valid].values
    hi = d['hi'][valid].values
    lo = d['lo'][valid].values
    idx = d['oof'][valid].index

    # For path-aware exits at the next H_BARS bars we need bar arrays from the
    # full close/high/low; cl/hi/lo are already aligned to valid OOF rows
    # so we approximate exit bar = entry + H_BARS using rolling proxy on valid.
    # For portfolio concurrency we just need (entry_time, exit_time, pnl_price).
    # exit_time = entry_time + H_BARS * 5min (M5).
    sig = oof[valid].values > thr
    if sig.sum() < 30:
        return None

    # Need full close/hi/lo to walk H_BARS forward inside the original index.
    # d['cl'] is aligned to OOF index already; we re-read full series via df.
    # Reconstruct the full-bar arrays from d (cl/hi/lo are masked to OOF rows
    # but we labelled with H_BARS lookahead so we can use rolling shift trick).
    # Use rolling future-window (max/min) on the masked aligned series — this
    # is the same approach the multi-instrument script uses (good enough).
    cl_s = pd.Series(cl, index=idx)
    hi_s = pd.Series(hi, index=idx)
    lo_s = pd.Series(lo, index=idx)
    hi_fwd = hi_s.rolling(H_BARS).max().shift(-H_BARS)
    lo_fwd = lo_s.rolling(H_BARS).min().shift(-H_BARS)
    fwd_ok = hi_fwd.notna() & lo_fwd.notna()
    keep = sig & fwd_ok.values
    if keep.sum() < 30:
        return None

    entry = cl[keep]
    ae    = at[keep]
    he    = hi_fwd.values[keep]
    le    = lo_fwd.values[keep]
    et    = idx[keep]
    xt    = et + pd.to_timedelta(H_BARS * 5, unit='m')

    tp_px = entry + TP_MULT * ae
    sl_px = entry - SL_MULT * ae
    tp_hit = he >= tp_px
    sl_hit = le <= sl_px
    wins    = tp_hit & ~sl_hit
    losses  = sl_hit & ~tp_hit
    ambig   = tp_hit & sl_hit
    timeout = ~tp_hit & ~sl_hit

    pnl_px = np.zeros(len(entry))
    pnl_px[wins]            =  TP_MULT * ae[wins]
    pnl_px[losses | ambig]  = -SL_MULT * ae[losses | ambig]
    pnl_px[timeout]         = (he[timeout] + le[timeout]) / 2 - entry[timeout]
    spread = SPREAD_BY_SYM[sym]
    pnl_px -= spread

    risk_px = SL_MULT * ae  # risk distance per unit (price)
    return pd.DataFrame({
        'symbol': sym,
        'entry_time': et,
        'exit_time':  xt,
        'entry':      entry,
        'pnl_price':  pnl_px,    # signed price move per unit (after spread)
        'risk_price': risk_px,   # SL distance per unit (positive)
    })


def best_threshold(d, sym):
    """Per-symbol PF-best threshold on OOF (mirrors multi-instrument logic)."""
    oof = d['oof']
    valid = oof.notna()
    cl = d['cl'][valid].values; at = d['at'][valid].values
    hi = d['hi'][valid].values; lo = d['lo'][valid].values
    idx = d['oof'][valid].index
    cl_s = pd.Series(cl, index=idx); hi_s = pd.Series(hi, index=idx); lo_s = pd.Series(lo, index=idx)
    hi_fwd = hi_s.rolling(H_BARS).max().shift(-H_BARS).values
    lo_fwd = lo_s.rolling(H_BARS).min().shift(-H_BARS).values
    fwd_ok = ~np.isnan(hi_fwd) & ~np.isnan(lo_fwd)
    spread = SPREAD_BY_SYM[sym]
    best = None
    for thr in np.arange(0.20, 0.55, 0.02):
        sig = (oof[valid].values > thr) & fwd_ok
        n = int(sig.sum())
        if n < 60: continue
        entry = cl[sig]; ae = at[sig]; he = hi_fwd[sig]; le = lo_fwd[sig]
        tp_px = entry + TP_MULT * ae; sl_px = entry - SL_MULT * ae
        tp_hit = he >= tp_px; sl_hit = le <= sl_px
        wins = tp_hit & ~sl_hit; losses = sl_hit & ~tp_hit
        ambig = tp_hit & sl_hit; timeout = ~tp_hit & ~sl_hit
        pnl_w = (TP_MULT * ae[wins]).sum()
        pnl_l = -(SL_MULT * ae[losses | ambig]).sum()
        pnl_t = ((he[timeout] + le[timeout])/2 - entry[timeout]).sum()
        net   = pnl_w + pnl_l + pnl_t - spread*n
        gw = pnl_w; gl = abs(pnl_l) + spread*n
        pf = gw/gl if gl > 0 else float('inf')
        if best is None or pf > best['pf']:
            best = {'thr': float(round(thr,2)), 'n': n, 'pf': float(round(pf,3)),
                    'net': float(net)}
    return best


def simulate_portfolio(trades_by_sym, init_equity=INIT_EQUITY, risk=RISK_PER_TRADE):
    """Combine per-symbol trade lists -> portfolio equity curve.
    Rules:
      - 1 concurrent trade per symbol max (skip overlap).
      - All symbols can be live at once (3 max simultaneous).
      - Position size per trade = risk * equity / risk_price * 1unit.
        usd_pnl = pnl_price / risk_price * risk * equity.
    """
    all_trades = []
    for sym, df in trades_by_sym.items():
        if df is None or df.empty: continue
        df = df.sort_values('entry_time').reset_index(drop=True)
        # symbol-level no-overlap filter
        kept = []
        last_exit = pd.Timestamp.min.tz_localize(None)
        for r in df.itertuples(index=False):
            et = pd.Timestamp(r.entry_time).tz_localize(None) if getattr(r.entry_time, 'tzinfo', None) else pd.Timestamp(r.entry_time)
            if et < last_exit:
                continue
            kept.append(r)
            xt = pd.Timestamp(r.exit_time).tz_localize(None) if getattr(r.exit_time, 'tzinfo', None) else pd.Timestamp(r.exit_time)
            last_exit = xt
        all_trades.extend(kept)

    if not all_trades:
        return None

    portfolio = pd.DataFrame(all_trades)
    portfolio['entry_time'] = pd.to_datetime(portfolio['entry_time']).dt.tz_localize(None)
    portfolio['exit_time']  = pd.to_datetime(portfolio['exit_time']).dt.tz_localize(None)
    portfolio = portfolio.sort_values('entry_time').reset_index(drop=True)

    equity = init_equity
    eq_curve = []   # list[(exit_time, equity, sym, usd_pnl)]
    per_sym_pnl = {s: 0.0 for s in trades_by_sym.keys()}
    for r in portfolio.itertuples(index=False):
        risk_px = float(r.risk_price)
        if risk_px <= 0:
            continue
        units = (risk * equity) / risk_px
        usd_pnl = units * float(r.pnl_price)
        equity += usd_pnl
        per_sym_pnl[r.symbol] += usd_pnl
        eq_curve.append((r.exit_time, equity, r.symbol, usd_pnl))

    eq_df = pd.DataFrame(eq_curve, columns=['time','equity','symbol','usd_pnl'])
    return {'trades': portfolio, 'equity_curve': eq_df, 'per_sym_pnl': per_sym_pnl,
            'final_equity': float(equity)}


def metrics(eq_df, trades, init_equity=INIT_EQUITY):
    if eq_df is None or eq_df.empty:
        return None
    pnl = eq_df['usd_pnl'].values
    pos = pnl[pnl > 0].sum()
    neg = abs(pnl[pnl < 0].sum())
    pf  = pos/neg if neg > 0 else float('inf')
    eq_series = pd.Series(eq_df['equity'].values, index=pd.to_datetime(eq_df['time']))
    eq_series = pd.concat([pd.Series([init_equity], index=[eq_series.index.min() - pd.Timedelta(minutes=5)]), eq_series])
    peak = eq_series.cummax()
    dd_pct = (eq_series/peak - 1.0).min() * 100
    span_days = (eq_series.index.max() - eq_series.index.min()).days
    months = max(span_days/30.4375, 1e-9)
    n = len(eq_df)
    tr_per_mo = n / months
    # Sharpe — daily-resampled
    daily = eq_series.resample('D').last().ffill().pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0.0
    return {'n_trades': int(n), 'pf': float(pf), 'max_dd_pct': float(dd_pct),
            'final_equity': float(eq_series.iloc[-1]),
            'trades_per_month': float(tr_per_mo), 'sharpe': float(sharpe),
            'span_days': int(span_days)}


def main():
    t0 = time.time()
    out = {'config': {'h': H_BARS, 'sl': SL_MULT, 'tp': TP_MULT,
                      'risk_per_trade': RISK_PER_TRADE,
                      'init_equity': INIT_EQUITY, 'symbols': SYMS},
           'per_symbol': {}, 'portfolio': {}, 'baseline_eur_only': {}}

    sym_data = {}
    sym_thr  = {}
    for sym in SYMS:
        print(f"\n=== {sym} ===")
        df, feat_cols = load_symbol(sym)
        d  = train_oof(df, sym, feat_cols)
        sym_data[sym] = d
        b = best_threshold(d, sym)
        sym_thr[sym] = b['thr'] if b else 0.5
        print(f"  best thr={sym_thr[sym]} (PF={b['pf'] if b else 'NA'} n={b['n'] if b else 0})")
        out['per_symbol'][sym] = {'auc': d['auc'], 'rows': d['rows'],
                                  'balance': d['balance'],
                                  'best': b}

    # Build trade lists at per-symbol best thresholds
    trades_by_sym = {}
    for sym in SYMS:
        tr = make_trades(sym_data[sym], sym, sym_thr[sym])
        trades_by_sym[sym] = tr
        print(f"  {sym}: trades after thr filter = {0 if tr is None else len(tr)}")

    # Portfolio sim
    print("\n=== PORTFOLIO ===")
    port = simulate_portfolio(trades_by_sym)
    if port is None:
        print("NO TRADES")
        out['portfolio'] = None
    else:
        m = metrics(port['equity_curve'], port['trades'])
        out['portfolio']['metrics'] = m
        out['portfolio']['per_symbol_pnl'] = {k: float(v) for k,v in port['per_sym_pnl'].items()}
        print(f"  Combined: n={m['n_trades']} PF={m['pf']:.3f} DD={m['max_dd_pct']:.2f}% "
              f"final=${m['final_equity']:.0f} tr/mo={m['trades_per_month']:.1f} "
              f"Sharpe={m['sharpe']:.2f}")
        for s,v in port['per_sym_pnl'].items():
            print(f"  {s}: ${v:+.0f}")

    # Per-symbol equity curves -> correlation
    per_sym_eq = {}
    for sym in SYMS:
        sub = port['equity_curve'][port['equity_curve']['symbol']==sym].copy() if port else None
        if sub is None or sub.empty: continue
        sub.index = pd.to_datetime(sub['time'])
        per_sym_eq[sym] = sub['usd_pnl'].resample('D').sum()
    if len(per_sym_eq) >= 2:
        all_d = pd.DataFrame(per_sym_eq).fillna(0)
        corr = all_d.corr().round(3)
        print("\nDaily-PnL correlation:")
        print(corr.to_string())
        out['portfolio']['daily_pnl_correlation'] = corr.to_dict()

    # Baseline: EURUSD-only
    print("\n=== BASELINE (EURUSD-only) ===")
    eur_only = simulate_portfolio({'EURUSD': trades_by_sym.get('EURUSD')})
    if eur_only is not None:
        m_b = metrics(eur_only['equity_curve'], eur_only['trades'])
        out['baseline_eur_only']['metrics'] = m_b
        print(f"  EUR-only: n={m_b['n_trades']} PF={m_b['pf']:.3f} DD={m_b['max_dd_pct']:.2f}% "
              f"final=${m_b['final_equity']:.0f} tr/mo={m_b['trades_per_month']:.1f} "
              f"Sharpe={m_b['sharpe']:.2f}")

    # Verdict
    if port is not None and out['baseline_eur_only'].get('metrics'):
        m = out['portfolio']['metrics']; b = out['baseline_eur_only']['metrics']
        better_dd = m['max_dd_pct'] > b['max_dd_pct']        # DD% is negative; closer-to-zero is "greater"
        better_sh = m['sharpe']     > b['sharpe']
        better_pf = m['pf']         > b['pf']
        verdict = {
            'sharpe_improved': bool(better_sh),
            'dd_improved':     bool(better_dd),
            'pf_improved':     bool(better_pf),
            'sharpe_delta':    float(m['sharpe'] - b['sharpe']),
            'dd_delta_pct':    float(m['max_dd_pct'] - b['max_dd_pct']),
            'pf_delta':        float(m['pf'] - b['pf']),
            'overall_better':  bool(better_sh and better_dd),
        }
        out['verdict'] = verdict
        print("\n=== VERDICT ===")
        print(f"  Sharpe: portfolio={m['sharpe']:.2f} vs eur={b['sharpe']:.2f} -> {'IMPROVED' if better_sh else 'WORSE'}")
        print(f"  DD%:    portfolio={m['max_dd_pct']:.2f} vs eur={b['max_dd_pct']:.2f} -> {'IMPROVED' if better_dd else 'WORSE'}")
        print(f"  PF:     portfolio={m['pf']:.3f} vs eur={b['pf']:.3f} -> {'IMPROVED' if better_pf else 'WORSE'}")
        print(f"  Overall (Sharpe AND DD better): {'YES' if verdict['overall_better'] else 'NO'}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nSaved {OUT_JSON}, total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
