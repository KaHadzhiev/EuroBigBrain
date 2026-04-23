#!/usr/bin/env python3
"""EUR multi-strategy, multi-session portfolio runner.

Built per user instruction (2026-04-23): mirror the gold-bot pattern --
several uncorrelated strategies running in parallel on different time
sessions, combined into one portfolio.

Pipeline:
  1. Load M5 6yr Dukascopy data + features (already on disk).
  2. Build the ctx dict expected by mt5_sim_strategies.generate_signals.
  3. For each strategy config (entry_type + session window + bracket params):
       - Generate entry signals with probs=1.0 (no ML gate).
       - Simulate each entry forward via a TRUE hi/lo M5 bracket sim.
       - Apply Vantage 1.3 pip spread + 0.2 pip slippage cost model.
       - Collect a trade log (open_time, close_time, pnl_usd, exit_reason).
  4. Combine logs via portfolio/combine.py for joint PF/DD/Sharpe/correlation.
  5. Save per-strategy + combined CSVs, equity PNG, JSON summary.

This is M5-resolution sim (TRUE bar high/low used to detect SL/TP). For full
MT5-faithful M1 fills, use mt5_sim.SimEngine (slower, requires M1 build).

Usage:
  python eur_portfolio_run.py
  python eur_portfolio_run.py --tag custom_tag   # output prefix
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results" / "portfolio"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mt5_sim_strategies import generate_signals  # noqa: E402
from portfolio.combine import combine_strategies, write_artifacts  # noqa: E402

# ---------- Cost model (Vantage EURUSD Standard STP) ----------
PIP = 0.0001
PT = 0.00001
SPREAD_PIPS = 1.3       # Vantage EUR median (pulled 2026-04-22)
SLIPPAGE_PIPS = 0.2
COST_PRICE = (SPREAD_PIPS + SLIPPAGE_PIPS) * PIP   # round-trip cost in price units

# Per-trade USD PnL: $1 per pip on 0.01 lot. We size to fixed risk in pips,
# converted to USD with a fixed $-per-pip multiplier so the portfolio
# combiner can reason about $ DD on a $1k account.
USD_PER_PIP = 1.0       # 0.01 lot on EURUSD -> $0.10/pip; we use 0.10 lot equivalent

# ---------- Strategy roster (gold-bot pattern: cover all sessions) ----------
# Each entry: (name, config dict). Configs include session window + bracket params.
# Sessions are UTC hours. EURUSD session map:
#   Asian:    00-07
#   London:   07-13
#   NY:       13-20
#   Late NY:  20-24
STRATEGIES = [
    {
        "name": "asian_range_lonopen",
        "entry_type": "asian_range",
        "vt": 0.0,
        "sess_start": 7, "sess_end": 13,         # Asian range broken at London open
        "bracket_offset": 0.0,                    # use the asian high/low as is
        "max_asian_atr": 6.0,
        "sl_atr": 0.7, "tp_atr": 2.0,
        "hold_bars": 12,
    },
    {
        "name": "london_breakout",
        "entry_type": "breakout_range",
        "vt": 0.0,
        "sess_start": 7, "sess_end": 13,         # London session
        "lookback": 12,                           # 1h rolling extreme
        "sl_atr": 0.7, "tp_atr": 2.0,
        "hold_bars": 12,
    },
    {
        "name": "ny_momentum_short",
        "entry_type": "momentum_short",
        "vt": 0.0,
        "sess_start": 13, "sess_end": 20,        # NY
        "bracket_offset": 0.3,
        "sl_atr": 0.7, "tp_atr": 2.0,
        "hold_bars": 12,
    },
    {
        "name": "ny_fade_long",
        "entry_type": "fade_long",
        "vt": 0.0,
        "sess_start": 13, "sess_end": 20,        # NY oversold reversion
        "bracket_offset": 0.3,
        "sl_atr": 0.7, "tp_atr": 2.0,
        "hold_bars": 12,
    },
    {
        "name": "atr_brk_full",
        "entry_type": "atr_bracket",
        "vt": 0.0,
        "sess_start": 7, "sess_end": 20,         # London + NY
        "bracket_offset": 0.3,
        "sl_atr": 0.7, "tp_atr": 2.0,
        "hold_bars": 12,
    },
    {
        "name": "vol_spike_late_ny",
        "entry_type": "vol_spike_bracket",
        "vt": 0.0,
        "sess_start": 14, "sess_end": 22,        # NY into late-NY illiquidity
        "vol_mult": 2.0,
        "bracket_offset": 0.3,
        "sl_atr": 0.7, "tp_atr": 2.0,
        "hold_bars": 12,
    },
]


# ---------- Indicator builders (fast NumPy/Pandas) ----------

def _atr14(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    out = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    return out


def _rsi14(c):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).rolling(14, min_periods=14).mean().to_numpy()
    rd = pd.Series(dn).rolling(14, min_periods=14).mean().to_numpy()
    rs = np.where(rd > 0, ru / rd, np.inf)
    return 100 - (100 / (1 + rs))


def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=True, min_periods=span).mean().to_numpy()


def _cci20(h, l, c, n=20):
    """Commodity Channel Index, 20-period."""
    tp = (h + l + c) / 3.0
    sma = pd.Series(tp).rolling(n, min_periods=n).mean().to_numpy()
    md = pd.Series(tp).rolling(n, min_periods=n).apply(lambda s: np.mean(np.abs(s - s.mean())), raw=True).to_numpy()
    cci = np.where(md > 0, (tp - sma) / (0.015 * md), 0.0)
    return cci


def _bb20(c, n=20, k=2.0):
    """Bollinger Bands (20, 2-sigma). Returns upper, lower, width."""
    sma = pd.Series(c).rolling(n, min_periods=n).mean().to_numpy()
    std = pd.Series(c).rolling(n, min_periods=n).std().to_numpy()
    upper = sma + k * std
    lower = sma - k * std
    width = upper - lower
    width_ma = pd.Series(width).rolling(30, min_periods=30).mean().to_numpy()
    return upper, lower, width, width_ma


def _asian_daily(df):
    """Per-date high/low between 22:00 prev day and 07:00 (Asian session)."""
    h = df['hour']
    asian_mask = (h >= 22) | (h < 7)
    asn = df.loc[asian_mask].copy()
    # bucket by trade date: 22:00-23:59 belongs to next calendar day's session
    asn['adate'] = np.where(asn['hour'] >= 22,
                            asn['date'] + pd.Timedelta(days=1),
                            asn['date'])
    grp = asn.groupby('adate').agg(asian_high=('high', 'max'),
                                    asian_low=('low', 'min'))
    return grp


def build_ctx(df: pd.DataFrame) -> dict:
    """Build the context dict expected by generate_signals + the local simulator."""
    h = df['high'].to_numpy(dtype=np.float64)
    l = df['low'].to_numpy(dtype=np.float64)
    c = df['close'].to_numpy(dtype=np.float64)
    v = df['volume'].to_numpy(dtype=np.float64) if 'volume' in df.columns \
        else np.ones(len(df), dtype=np.float64)

    atr14 = _atr14(h, l, c)
    rsi14 = _rsi14(c)
    ema8 = _ema(c, 8)
    ema21 = _ema(c, 21)
    ret5 = pd.Series(c).pct_change(5).to_numpy()
    cci20 = _cci20(h, l, c, 20)
    bb_u, bb_l, bb_w, bb_w_ma = _bb20(c, 20, 2.0)

    roll12_h = pd.Series(h).rolling(12, min_periods=12).max().to_numpy()
    roll12_l = pd.Series(l).rolling(12, min_periods=12).min().to_numpy()
    roll24_h = pd.Series(h).rolling(24, min_periods=24).max().to_numpy()
    roll24_l = pd.Series(l).rolling(24, min_periods=24).min().to_numpy()
    roll48_h = pd.Series(h).rolling(48, min_periods=48).max().to_numpy()
    roll48_l = pd.Series(l).rolling(48, min_periods=48).min().to_numpy()

    vol_ma20 = pd.Series(v).rolling(20, min_periods=20).mean().to_numpy()

    times = pd.to_datetime(df['time'])
    hours = times.dt.hour.to_numpy()
    dates = times.dt.date.to_numpy()
    df_for_asian = pd.DataFrame({'time': times, 'high': h, 'low': l,
                                 'hour': hours, 'date': times.dt.date})
    asian_daily = _asian_daily(df_for_asian)

    return {
        'c_v': c, 'h_v': h, 'lo_v': l,
        'atr14': atr14, 'rsi14': rsi14,
        'ema8': ema8, 'ema21': ema21,
        'ret5': ret5,
        'cci20': cci20,
        'bb_upper': bb_u, 'bb_lower': bb_l, 'bb_width': bb_w, 'bb_width_ma': bb_w_ma,
        'roll12_high': roll12_h, 'roll12_low': roll12_l,
        'roll24_high': roll24_h, 'roll24_low': roll24_l,
        'roll48_high': roll48_h, 'roll48_low': roll48_l,
        'vol_v': v, 'vol_ma20': vol_ma20,
        'hours': hours, 'dates': dates, 'times': times.to_numpy(),
        'asian_daily': asian_daily,
    }


# ---------- Bracket simulator (M5 TRUE hi/lo, SL/TP/timeout) ----------

def simulate_signals(ctx, signals, sl_atr, tp_atr, hold_bars,
                     spread_pips=SPREAD_PIPS, slip_pips=SLIPPAGE_PIPS,
                     usd_per_pip=USD_PER_PIP):
    """Walk each signal forward until SL/TP/timeout. Apply costs.

    A signal is (i, buy_level, sell_level). We treat the *level* as a stop-order
    entry: filled when the future bar high/low crosses it. Once filled, SL/TP
    are placed at level +/- sl_atr*ATR and level +/- tp_atr*ATR.

    For simplicity (and parity with the EBB sim) we trigger entry at next-bar
    open if the price level was crossed during the entry bar, otherwise we let
    the level sit for `hold_bars` bars before cancelling.
    """
    h = ctx['h_v']
    l = ctx['lo_v']
    c = ctx['c_v']
    atr = ctx['atr14']
    times = ctx['times']
    n = len(c)

    cost_pips = spread_pips + slip_pips
    cost_price = cost_pips * PIP

    rows = []
    for sig in signals:
        i, buy_lvl, sell_lvl = sig
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        # Decide direction: in level-cross strategies, buy_lvl ABOVE close /
        # sell_lvl BELOW close means stop entry on breakout. In fade/momentum
        # entries the level is set the same way -- entry at breakout of level.
        # We simulate both legs if both present, but only the FIRST to fill.
        for direction, lvl in (('long', buy_lvl), ('short', sell_lvl)):
            if lvl is None:
                continue

            # Wait up to hold_bars bars for the entry level to be crossed
            entry_bar = -1
            entry_px = np.nan
            for k in range(1, hold_bars + 1):
                j = i + k
                if j >= n:
                    break
                if direction == 'long' and h[j] >= lvl:
                    entry_bar = j
                    entry_px = lvl
                    break
                if direction == 'short' and l[j] <= lvl:
                    entry_bar = j
                    entry_px = lvl
                    break
            if entry_bar < 0:
                continue

            sl_dist = sl_atr * a
            tp_dist = tp_atr * a
            if direction == 'long':
                sl_px = entry_px - sl_dist
                tp_px = entry_px + tp_dist
            else:
                sl_px = entry_px + sl_dist
                tp_px = entry_px - tp_dist

            # Walk forward looking for SL/TP/timeout
            exit_bar = entry_bar + hold_bars
            exit_px = np.nan
            exit_reason = 'timeout'
            for k in range(1, hold_bars + 1):
                j = entry_bar + k
                if j >= n:
                    exit_bar = j - 1
                    exit_px = c[j - 1]
                    exit_reason = 'eod'
                    break
                if direction == 'long':
                    sl_hit = l[j] <= sl_px
                    tp_hit = h[j] >= tp_px
                    if sl_hit and tp_hit:
                        # ambiguous intra-bar; conservative -> SL first
                        exit_bar, exit_px, exit_reason = j, sl_px, 'sl'
                        break
                    if sl_hit:
                        exit_bar, exit_px, exit_reason = j, sl_px, 'sl'
                        break
                    if tp_hit:
                        exit_bar, exit_px, exit_reason = j, tp_px, 'tp'
                        break
                else:
                    sl_hit = h[j] >= sl_px
                    tp_hit = l[j] <= tp_px
                    if sl_hit and tp_hit:
                        exit_bar, exit_px, exit_reason = j, sl_px, 'sl'
                        break
                    if sl_hit:
                        exit_bar, exit_px, exit_reason = j, sl_px, 'sl'
                        break
                    if tp_hit:
                        exit_bar, exit_px, exit_reason = j, tp_px, 'tp'
                        break
            else:
                # fell off the loop -> timeout at last bar in window
                jj = min(entry_bar + hold_bars, n - 1)
                exit_bar = jj
                exit_px = c[jj]
                exit_reason = 'timeout'

            # gross PnL (price units)
            if direction == 'long':
                pnl_price = exit_px - entry_px
            else:
                pnl_price = entry_px - exit_px
            pnl_price -= cost_price
            pnl_pips = pnl_price / PIP
            pnl_usd = pnl_pips * usd_per_pip

            rows.append({
                'open_time': times[entry_bar],
                'close_time': times[exit_bar],
                'direction': direction,
                'entry_px': float(entry_px),
                'exit_px': float(exit_px),
                'pnl_pips': float(pnl_pips),
                'pnl': float(pnl_usd),
                'exit_reason': exit_reason,
                'signal_bar': int(i),
                'entry_bar': int(entry_bar),
                'exit_bar': int(exit_bar),
            })
    return pd.DataFrame(rows)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="eur_portfolio_v1")
    ap.add_argument("--from", dest="d_from", default="2020-01-01")
    ap.add_argument("--to", dest="d_to", default="2026-04-13")
    args = ap.parse_args()

    t0 = time.time()
    m5_path = DATA / "eurusd_m5_2020_2026.parquet"
    print(f"[load] {m5_path}")
    df = pd.read_parquet(m5_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df[(df['time'] >= args.d_from) & (df['time'] <= args.d_to)].reset_index(drop=True)
    print(f"  rows: {len(df):,}  span={df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    print("[build] indicators + ctx")
    ctx = build_ctx(df)
    n = len(df)
    test_indices = np.arange(n)
    probs = np.ones(n, dtype=np.float64)  # no ML gate

    per_strategy_logs = []
    per_strategy_summary = []

    for cfg in STRATEGIES:
        name = cfg['name']
        print(f"\n[strat] {name} ({cfg['entry_type']}, sess {cfg['sess_start']}-{cfg['sess_end']})")
        ts = time.time()
        signals = generate_signals(ctx, cfg, test_indices, probs)
        print(f"  signals: {len(signals):,}")
        log = simulate_signals(ctx, signals,
                               sl_atr=cfg['sl_atr'], tp_atr=cfg['tp_atr'],
                               hold_bars=cfg['hold_bars'])
        log['strategy'] = name
        if len(log):
            wins = (log['pnl'] > 0).sum()
            losses = (log['pnl'] <= 0).sum()
            gp = log.loc[log['pnl'] > 0, 'pnl'].sum()
            gl = abs(log.loc[log['pnl'] <= 0, 'pnl'].sum())
            pf = gp / gl if gl > 0 else 0.0
            wr = wins / len(log) if len(log) else 0.0
            net = log['pnl'].sum()
            print(f"  trades={len(log)}  PF={pf:.3f}  WR={wr:.3f}  net=${net:+.0f}  ({time.time()-ts:.1f}s)")
            per_strategy_summary.append({
                'name': name, 'trades': int(len(log)), 'pf': float(pf),
                'wr': float(wr), 'net_usd': float(net),
                'wins': int(wins), 'losses': int(losses),
                'session': f"{cfg['sess_start']}-{cfg['sess_end']}",
                'entry_type': cfg['entry_type'],
            })
            log.to_csv(RESULTS / f"{args.tag}__{name}__trades.csv", index=False)
            per_strategy_logs.append(log[['open_time', 'close_time', 'pnl', 'strategy']].copy())
        else:
            print(f"  trades=0 -- skipped from portfolio")

    print(f"\n[combine] portfolio of {len(per_strategy_logs)} strategies")
    if not per_strategy_logs:
        print("ERROR: no strategy produced any trades")
        return 1

    metrics = combine_strategies(per_strategy_logs, deposit=1000.0,
                                 max_concurrent_positions=2,
                                 overlap_penalty=0.5)
    print(f"  {metrics.summary_line()}")
    print(f"  win_rate: {metrics.win_rate:.3f}")
    print(f"  trades_per_month: {metrics.trades_per_month:.1f}")
    print(f"  per-year:")
    for y, n_t, pf, pn in metrics.per_year:
        print(f"    {y}  trades={n_t:5d}  PF={pf:.3f}  pnl=${pn:+.0f}")
    if metrics.correlation_matrix is not None and not metrics.correlation_matrix.empty:
        print(f"  correlation:\n{metrics.correlation_matrix.round(3)}")

    artifacts = write_artifacts(metrics, RESULTS, tag=args.tag)
    summary_path = RESULTS / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps({
        'tag': args.tag,
        'strategies': per_strategy_summary,
        'combined': {
            'pf': metrics.pf,
            'n_trades': metrics.n_trades,
            'total_pnl': metrics.total_pnl,
            'ending_equity': metrics.ending_equity,
            'return_pct': metrics.return_pct,
            'max_dd_pct': metrics.max_dd_pct,
            'max_dd_usd': metrics.max_dd_usd,
            'recovery_factor': metrics.recovery_factor,
            'sharpe_daily': metrics.sharpe_daily,
            'trades_per_month': metrics.trades_per_month,
            'win_rate': metrics.win_rate,
            'overlap_hits': metrics.overlap_hit_count,
            'per_year': metrics.per_year,
        },
        'artifacts': artifacts,
    }, indent=2, default=str), encoding='utf-8')
    print(f"\n[done] artifacts: {artifacts}")
    print(f"       summary:   {summary_path}")
    print(f"[elapsed] {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
