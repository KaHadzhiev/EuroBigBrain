#!/usr/bin/env python3
"""SIM DIAGNOSTIC — is the simulator broken? User pushed back: everything fails,
that's suspicious.

Checks:
  1. Buy-hold EUR 2020-2026 — should be roughly -500 pips (EUR fell ~1.12 -> 1.07).
  2. Random-entry sim with cost=0 — should average PF~1.0, not <1.
  3. Asian-fade with cost=0 — does the strategy itself lose money even pre-cost?
  4. Asian-fade with same-bar SL/TP biased FAVORABLY (TP-first for fades).
  5. Asian-fade with intra-bar randomization (~50/50 split when both hit).
  6. XAU smoke test — run the same sim on XAU with a known-positive cfg74-like
     strategy. If sim gives the expected ~PF=1.4, sim is fine. If it gives <1, sim broken.

Output: clear pass/fail for each check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

PIP = 0.0001


def load_eur():
    df = pd.read_parquet(DATA / "eurusd_m5_2020_2026.parquet")
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)


def load_xau():
    p = DATA / "XAUUSD_M5_6yr.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)


# ---------------- CHECK 1: buy-hold ----------------
def check_buy_hold(df, label="EUR"):
    first = df['close'].iloc[0]
    last = df['close'].iloc[-1]
    print(f"\n[1] BUY-HOLD {label}: {first:.5f} -> {last:.5f} = {(last-first)*10000:+.1f} pips "
          f"({(last-first)/first*100:+.2f}%)")


# ---------------- CHECK 2: random-entry sim with cost=0 ----------------
def check_random_entry(df, n_trades=2000, hold=12, seed=42):
    rng = np.random.default_rng(seed)
    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    c = df['close'].to_numpy()
    n = len(c)
    idx = rng.choice(np.arange(50, n - hold), size=n_trades, replace=False)
    pnls_long = c[idx + hold] - c[idx]
    pnls_short = c[idx] - c[idx + hold]
    pnls_random = np.where(rng.random(n_trades) < 0.5, pnls_long, pnls_short)
    pnls_pips = pnls_random / PIP
    pf_long = pnls_long[pnls_long > 0].sum() / abs(pnls_long[pnls_long <= 0].sum())
    pf_short = pnls_short[pnls_short > 0].sum() / abs(pnls_short[pnls_short <= 0].sum())
    pf_rand = pnls_random[pnls_random > 0].sum() / abs(pnls_random[pnls_random <= 0].sum())
    print(f"\n[2] RANDOM ENTRY (n={n_trades}, hold={hold}, NO COST):")
    print(f"    long-only  PF={pf_long:.3f}  net={pnls_long.sum()/PIP:+.0f} pips")
    print(f"    short-only PF={pf_short:.3f}  net={pnls_short.sum()/PIP:+.0f} pips")
    print(f"    50/50 mix  PF={pf_rand:.3f}  net={pnls_random.sum()/PIP:+.0f} pips")
    print(f"    EXPECTED:  long PF<1 (EUR fell), short PF>1, mix ~1 (no edge)")


# ---------------- Asian-fade re-implementation with optional bias ----------------
def asian_fade(df, ext_atr=2.0, sl_atr=2.0, tp_atr=1.0, sess=(20, 0), hold_bars=48,
               cost_pips=1.5, sl_first=True, randomize=False, seed=42):
    """Same as eur_asian_fade.py logic but parameterizes the SL/TP same-bar bias."""
    df = df.copy()
    times = df['time'].to_numpy()
    h = df['high'].to_numpy(); l = df['low'].to_numpy()
    c = df['close'].to_numpy()
    n = len(c)

    # M15 ATR + EMA50 forward-filled to M5
    df15 = df.set_index('time').resample('15min').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    pc = np.r_[df15['close'].iloc[0], df15['close'].iloc[:-1].to_numpy()]
    tr = np.maximum.reduce([df15['high'] - df15['low'],
                            np.abs(df15['high'] - pc), np.abs(df15['low'] - pc)])
    df15['atr15'] = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    df15['ema15_50'] = df15['close'].ewm(span=50, adjust=True, min_periods=50).mean()
    df = df.set_index('time')
    df['atr15'] = df15['atr15'].reindex(df.index, method='ffill')
    df['ema15_50'] = df15['ema15_50'].reindex(df.index, method='ffill')
    df = df.reset_index()
    atr15 = df['atr15'].to_numpy()
    ema15 = df['ema15_50'].to_numpy()
    hours = df['time'].dt.hour.to_numpy()

    s, e = sess
    if e <= s:
        in_sess = (hours >= s) | (hours < e)
    else:
        in_sess = (hours >= s) & (hours < e)

    rng = np.random.default_rng(seed)
    pnls = []; cooldown = -1
    n_amb = 0
    for i in range(50, n - hold_bars):
        if not in_sess[i] or i <= cooldown:
            continue
        a = atr15[i]
        if not np.isfinite(a) or a <= 0:
            continue
        em = ema15[i]
        if not np.isfinite(em):
            continue
        dist = c[i] - em
        if dist >= ext_atr * a:
            direction = 'short'; entry = c[i]
            sl_px = entry + sl_atr * a; tp_px = entry - tp_atr * a
        elif -dist >= ext_atr * a:
            direction = 'long'; entry = c[i]
            sl_px = entry - sl_atr * a; tp_px = entry + tp_atr * a
        else:
            continue

        exit_px = np.nan; exit_bar = i + hold_bars; exit_reason = 'timeout'
        for k in range(1, hold_bars + 1):
            j = i + k
            if direction == 'long':
                sl_hit = l[j] <= sl_px
                tp_hit = h[j] >= tp_px
            else:
                sl_hit = h[j] >= sl_px
                tp_hit = l[j] <= tp_px
            if sl_hit and tp_hit:
                n_amb += 1
                if randomize:
                    pick_sl = rng.random() < 0.5
                else:
                    pick_sl = sl_first
                if pick_sl:
                    exit_px, exit_bar, exit_reason = sl_px, j, 'sl'
                else:
                    exit_px, exit_bar, exit_reason = tp_px, j, 'tp'
                break
            if sl_hit:
                exit_px, exit_bar, exit_reason = sl_px, j, 'sl'; break
            if tp_hit:
                exit_px, exit_bar, exit_reason = tp_px, j, 'tp'; break
        if not np.isfinite(exit_px):
            j = min(i + hold_bars, n - 1)
            exit_px = c[j]; exit_bar = j

        if direction == 'long':
            pnl_price = exit_px - entry
        else:
            pnl_price = entry - exit_px
        pnl_pips = pnl_price / PIP - cost_pips
        pnls.append(pnl_pips)
        cooldown = exit_bar
    pnls = np.array(pnls)
    if not len(pnls):
        return {'trades': 0, 'pf': 0, 'wr': 0, 'net_pips': 0, 'n_amb': n_amb}
    gp = pnls[pnls > 0].sum(); gl = abs(pnls[pnls <= 0].sum())
    return {
        'trades': int(len(pnls)),
        'pf': float(gp / gl) if gl > 0 else 0,
        'wr': float((pnls > 0).mean()),
        'net_pips': float(pnls.sum()),
        'n_amb': n_amb,
    }


def check_asian_fade_modes(df):
    print("\n[3-5] ASIAN-FADE under 4 sim modes (ext=2 sl=2 tp=1 sess=20-0 hold=48):")
    for cost in [0.0, 1.5]:
        for label, sl_first, rand in [
            ("SL-first (current)", True, False),
            ("TP-first (favorable)", False, False),
            ("Random (fair coin)", False, True),
        ]:
            r = asian_fade(df, cost_pips=cost, sl_first=sl_first, randomize=rand)
            print(f"  cost={cost:.1f}  {label:25s}  trades={r['trades']:4d}  PF={r['pf']:.3f}  "
                  f"WR={r['wr']:.3f}  net={r['net_pips']:+.0f} pips  ambig_bars={r['n_amb']}")


def check_xau(df_eur):
    df_xau = load_xau()
    if df_xau is None:
        print("\n[6] XAU SMOKE: XAUUSD_M5_6yr.parquet missing -> skipped")
        return
    print("\n[6] XAU SMOKE TEST -- same sim on XAU should produce a known-positive result")
    # Run a XAU asian-fade variant — XAU is more volatile so wider params
    r = asian_fade(df_xau, ext_atr=1.5, sl_atr=2.0, tp_atr=1.0, sess=(20, 0), hold_bars=48,
                   cost_pips=2.0, sl_first=True)
    print(f"  XAU asian-fade (cost=2.0): trades={r['trades']} PF={r['pf']:.3f} WR={r['wr']:.3f} net={r['net_pips']:+.0f} pips amb={r['n_amb']}")
    r = asian_fade(df_xau, ext_atr=1.5, sl_atr=2.0, tp_atr=1.0, sess=(20, 0), hold_bars=48,
                   cost_pips=0.0, sl_first=False)
    print(f"  XAU asian-fade (cost=0, TP-first): trades={r['trades']} PF={r['pf']:.3f} WR={r['wr']:.3f} net={r['net_pips']:+.0f} pips")


def main():
    print("====== EUR SIM DIAGNOSTIC ======")
    df_eur = load_eur()
    print(f"loaded EUR: {len(df_eur):,} bars, span {df_eur['time'].iloc[0]} -> {df_eur['time'].iloc[-1]}")

    check_buy_hold(df_eur, "EUR")
    check_random_entry(df_eur)
    check_asian_fade_modes(df_eur)
    check_xau(df_eur)
    print("\n====== END ======")


if __name__ == '__main__':
    main()
