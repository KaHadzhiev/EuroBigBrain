#!/usr/bin/env python3
"""Smoke test for the EBB EURUSD SimEngine port.

Loads any available EUR M5 OHLCV bar file (first match), runs SimEngine with
the `atr_breakout` strategy at default params, and asserts:
  - no exceptions during run
  - >=1 trade generated  (smoke threshold; real validation is a separate job)
  - output schema matches the GBB SimEngine result dict

If no EUR data is found locally the test fabricates 1000 synthetic M1 bars
(centered at 1.05) and runs against those. The synthetic-data run still must
return a valid schema dict, but the >=1-trade assertion is downgraded to a
warning (synthetic random walk may not generate any fills under default
brackets).
"""

import os
import sys
import glob
import datetime
import numpy as np
import pandas as pd

# Make the EBB python/ package importable when run as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from mt5_sim import SimEngine, build_m5_to_m1_map_uniform   # noqa: E402
from mt5_sim_strategies import generate_signals             # noqa: E402


# Search paths (Win + relative). Mac drive isn't mounted on Win, but a copy
# typically exists locally in IdeaProjects.
SEARCH_PATTERNS = [
    r"C:\Users\kahad\IdeaProjects\EuroBigBrain\data\EURUSD*M5*.csv",
    r"C:\Users\kahad\IdeaProjects\EuroBigBrain\data\EURUSD*M5*.parquet",
    r"C:\Users\kahad\IdeaProjects\EuroBigBrain\data\eur*M5*.csv",
    r"C:\Users\kahad\IdeaProjects\GoldBigBrain\data\EURUSD*.csv",
    r"C:\Users\kahad\IdeaProjects\GoldBigBrain\data\eur*.csv",
]

EXPECTED_KEYS = {
    'trades', 'pf', 'pnl', 'win_rate', 'gross_win', 'gross_loss',
    'wins', 'losses', 'max_dd', 'exit_reasons', 'trade_list', 'per_year',
    'pf_raw', 'pf_correction', 'ea_family',
    'rejected_for_stops_level', 'rejected_for_freeze_level',
}


def _find_eur_data() -> str | None:
    for pat in SEARCH_PATTERNS:
        matches = glob.glob(pat)
        if matches:
            # Prefer the largest file (most data)
            matches.sort(key=lambda p: os.path.getsize(p), reverse=True)
            return matches[0]
    return None


def _load_eur_m5(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    elif 'date' in df.columns:
        df['time'] = pd.to_datetime(df['date'])
    elif 'datetime' in df.columns:
        df['time'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError(f"No time column found in {path} (columns={list(df.columns)})")
    df = df.sort_values('time').reset_index(drop=True)
    return df


def _build_synthetic_m5(n_m5: int = 200) -> pd.DataFrame:
    """Fabricate a synthetic EURUSD M5 frame (random walk centered at 1.05)."""
    np.random.seed(7)
    closes = 1.05 + np.cumsum(np.random.randn(n_m5) * 0.0001)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_m5) * 0.00015)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_m5) * 0.00015)
    base_t = datetime.datetime(2024, 1, 2, 8, 0, 0)
    times = [base_t + datetime.timedelta(minutes=5 * i) for i in range(n_m5)]
    return pd.DataFrame({
        'time': pd.to_datetime(times),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.randint(50, 500, size=n_m5),
    })


def _atr14(df: pd.DataFrame) -> np.ndarray:
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr = np.zeros(len(df))
    tr[0] = high[0] - low[0]
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    # Simple rolling mean ATR-14
    atr = np.full(len(df), np.nan)
    if len(df) >= 14:
        for i in range(13, len(df)):
            atr[i] = tr[i - 13:i + 1].mean()
    return atr


def main() -> int:
    path = _find_eur_data()
    used_synthetic = path is None
    if path is not None:
        print(f"[smoke] EUR data: {path}")
        df = _load_eur_m5(path)
    else:
        print("[smoke] No EUR data found — fabricating 1000 synthetic M1-equivalent bars.")
        df = _build_synthetic_m5(n_m5=1000)

    # Cap to a reasonable smoke-test slice
    if len(df) > 5000:
        df = df.iloc[:5000].copy()
    print(f"[smoke] M5 bars: {len(df)}  (range: {df['time'].iloc[0]} to {df['time'].iloc[-1]})")

    # Synthesize M1 from M5 by repeating each M5 bar 5x. Crude but sufficient
    # for state-machine smoke check (engine resolution is M1, not strategy).
    n_m5 = len(df)
    n_m1 = n_m5 * 5
    m1_open = np.repeat(df['open'].values, 5)
    m1_high = np.repeat(df['high'].values, 5)
    m1_low = np.repeat(df['low'].values, 5)
    m1_close = np.repeat(df['close'].values, 5)

    m1_data = {
        'open': m1_open,
        'high': m1_high,
        'low': m1_low,
        'close': m1_close,
    }

    atr = _atr14(df)
    m5_data = {
        'open': df['open'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'close': df['close'].values,
        'hours': df['time'].dt.hour.values,
        'dates': df['time'].dt.date.values,
        'years': df['time'].dt.year.values,
        'atr14': atr,
    }

    # Strategy ctx — atr_breakout uses bracket-offset of ATR around close.
    # rsi/ema/vol arrays are required by the dispatch but unused for atr_breakout.
    closes = df['close'].values
    ctx = {
        'c_v': closes,
        'h_v': df['high'].values,
        'lo_v': df['low'].values,
        'atr14': atr,
        'hours': m5_data['hours'],
        'dates': m5_data['dates'],
        'rsi14': np.full(n_m5, 50.0),
        'ema8': closes,
        'ema21': closes,
        'vol_v': df['volume'].values if 'volume' in df.columns else np.ones(n_m5),
        'vol_ma20': np.ones(n_m5),
        'ret5': np.zeros(n_m5),
        'asian_daily': pd.DataFrame(),
        'roll12_high': np.full(n_m5, np.nan),
        'roll12_low': np.full(n_m5, np.nan),
        'roll24_high': np.full(n_m5, np.nan),
        'roll24_low': np.full(n_m5, np.nan),
        'roll48_high': np.full(n_m5, np.nan),
        'roll48_low': np.full(n_m5, np.nan),
    }

    cfg_signals = {
        'entry_type': 'atr_breakout',
        'vt': 0.0,             # no ML gate for smoke test
        'sess_start': 0,       # 24/5 EURUSD — use full session
        'sess_end': 24,
        'bracket_offset': 0.3,
    }
    valid_idx = np.where(~np.isnan(atr) & (atr > 0))[0]
    probs = np.full(len(valid_idx), 1.0)
    signals_raw = generate_signals(ctx, cfg_signals, valid_idx.tolist(), probs)
    print(f"[smoke] strategy signals generated: {len(signals_raw)}")

    # Pad signal tuples with sl/tp/be/trail distances (in price units).
    # Use ~10 pips SL, 20 pips TP, 5 pips BE, 5 pips trail — modest defaults
    # so the smoke fills are not all wiped by the stops_level guard.
    sl_dist = 0.0010   # 10 pips
    tp_dist = 0.0020   # 20 pips
    be_dist = 0.0005
    trail_dist = 0.0005
    sigs = [(m5_i, b, s, sl_dist, tp_dist, be_dist, trail_dist)
            for (m5_i, b, s) in signals_raw]

    config = {
        'bracket_bars': 3,
        'max_hold_bars': 24,
        'max_trades_per_day': 20,
        'daily_loss_cap_pct': 5.0,
        'starting_balance': 10000.0,
    }
    costs = {
        'spread': 0.00007,
        'slippage': 0.00002,
        'commission': 0.0,
        'use_variable_spread': False,
    }

    engine = SimEngine(m1_data, m5_data, config, costs, ea_family='ar')
    m5_map = build_m5_to_m1_map_uniform(n_m1, n_m5, bars_per_m5=5)
    print(f"[smoke] running SimEngine ({len(sigs)} signals, {n_m1} M1 bars)...")
    result = engine.run(sigs, m5_map)
    print(f"[smoke] DONE — trades={result['trades']}  pf={result['pf']}  "
          f"pnl={result['pnl']}  wr={result['win_rate']}  max_dd={result['max_dd']}")
    if result['exit_reasons']:
        print(f"[smoke] exit reasons: {result['exit_reasons']}")

    # Schema check
    missing = EXPECTED_KEYS - set(result.keys())
    if missing:
        print(f"[smoke] FAIL — schema missing keys: {missing}")
        return 1

    # Trades check (downgrade for synthetic)
    if result['trades'] < 1:
        if used_synthetic:
            print("[smoke] WARN — synthetic data produced 0 trades, schema OK. PASS-with-warn.")
            return 0
        print("[smoke] FAIL — real EUR data produced 0 trades, signals likely broken.")
        return 1

    print(f"[smoke] PASS — schema OK, {result['trades']} trades. "
          f"(synthetic={used_synthetic})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
