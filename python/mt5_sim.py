#!/usr/bin/env python3
"""MT5-faithful bracket EA simulator for EURUSD — CORE ENGINE (EBB port).

Ported from GoldBigBrain mt5_sim.py. Functionally identical state machine;
only instrument constants and default cost parameters were retuned for
EURUSD on a Vantage Standard STP account.

Key differences vs GBB:
  - Symbol = EURUSD (digits=5). 1 point = 0.00001, 1 pip = 0.0001.
  - Default spread = 0.00007 (~0.7 pip, typical Vantage EURUSD median).
  - Default stops_level / freeze_level = 5 points = 0.00005 (Vantage tight).
  - Variable-spread parquet path → data/eurusd_spread_*.parquet (optional).
  - Bias-correction defaults to 1.0 until EUR-specific calibration is run.
  - 24/5 trading; no XAU-specific daily-settlement gap logic exists here.
"""

import json
import os
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Broker constraints (Vantage EURUSD — Standard STP account)
# ---------------------------------------------------------------------------
# 1 point on EURUSD (5-digit broker) = 0.00001 USD/EUR.
# 1 pip = 10 points = 0.0001.
# stops_level / freeze_level: minimum distance for a stop order or SL/TP
#   modification away from the current price. EURUSD on Vantage is very
#   tight (typically 0-2 points). Use 5 pts as a conservative default;
#   override via costs['stops_level'].
PT_PRICE = 0.00001
PIP_PRICE = 0.0001
STOPS_LEVEL_POINTS = 5
FREEZE_LEVEL_POINTS = 5
STOPS_LEVEL_PRICE = STOPS_LEVEL_POINTS * PT_PRICE   # 0.00005
FREEZE_LEVEL_PRICE = FREEZE_LEVEL_POINTS * PT_PRICE  # 0.00005


# ---------------------------------------------------------------------------
# Per-EA bias-correction coefficients (sim PF -> MT5-aligned PF)
# ---------------------------------------------------------------------------
_BIAS_COEFF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'sim_bias_coefficients.json')

# Defaults for EBB: 1.0 (no correction) until EUR calibration is performed.
# GBB used 0.96 (ar) / 0.82 (v5) — those were XAUUSD-specific and should NOT
# be carried over without re-calibrating.
_DEFAULT_BIAS_COEFFS = {
    'ar': {'pf_correction': 1.0, 'n': 0, 'source': 'EBB-default-uncalibrated'},
    'v5': {'pf_correction': 1.0, 'n': 0, 'source': 'EBB-default-uncalibrated'},
}


def _load_bias_coefficients() -> dict:
    """Load per-EA-family PF bias-correction coefficients from JSON.

    Falls back to hard-coded EBB defaults (1.0 = no correction) if file
    missing/corrupt and emits a warning.
    """
    try:
        with open(_BIAS_COEFF_PATH, 'r') as f:
            data = json.load(f)
        coeffs = {k: v for k, v in data.items() if not k.startswith('_')}
        for fam in ('ar', 'v5'):
            if fam not in coeffs or 'pf_correction' not in coeffs[fam]:
                raise ValueError(f"missing family '{fam}' in {_BIAS_COEFF_PATH}")
            float(coeffs[fam]['pf_correction'])
        return coeffs
    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError) as e:
        warnings.warn(
            f"[ebb_mt5_sim] Could not load {_BIAS_COEFF_PATH} ({e!r}); "
            f"using EBB defaults (pf_correction=1.0).",
            RuntimeWarning,
        )
        return _DEFAULT_BIAS_COEFFS


SIM_BIAS_COEFFICIENTS = _load_bias_coefficients()

for _fam, _meta in SIM_BIAS_COEFFICIENTS.items():
    print(f"[ebb_mt5_sim] PF bias coefficient for '{_fam}': "
          f"{_meta['pf_correction']:.4f} (n={_meta.get('n', '?')})")


# ---------------------------------------------------------------------------
# Variable-spread replay (bar-by-bar MT5-history spread)
# ---------------------------------------------------------------------------
_SPREAD_SERIES_CACHE: dict = {}
_SPREAD_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                 'eurusd_spread_2020_2026.parquet'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                 'eurusd_spread_2020_2026.csv'),
]


def _load_spread_series() -> Optional[dict]:
    """Load per-bar EURUSD spread series from parquet (preferred) or csv."""
    if 'series' in _SPREAD_SERIES_CACHE:
        return _SPREAD_SERIES_CACHE['series']
    for p in _SPREAD_PATHS:
        p = os.path.abspath(p)
        if not os.path.exists(p):
            continue
        try:
            if p.lower().endswith('.parquet'):
                import pandas as pd
                df = pd.read_parquet(p)
            else:
                import pandas as pd
                df = pd.read_csv(p)
            t = np.asarray(df['time'], dtype=np.int64)
            sp_pts = np.asarray(df['spread_points'], dtype=np.int32)
            sp_price = np.asarray(df['spread_price'], dtype=np.float32)
            order = np.argsort(t, kind='stable')
            series = {
                'time': t[order],
                'spread_points': sp_pts[order],
                'spread_price': sp_price[order],
                'path': p,
            }
            _SPREAD_SERIES_CACHE['series'] = series
            print(f"[ebb_mt5_sim] variable-spread series loaded from {p} "
                  f"({len(t):,} bars, med={int(np.median(sp_pts))} pts, "
                  f"p95={int(np.percentile(sp_pts, 95))} pts)")
            return series
        except Exception as e:
            warnings.warn(
                f"[ebb_mt5_sim] failed to load spread series from {p}: {e!r}",
                RuntimeWarning,
            )
    _SPREAD_SERIES_CACHE['series'] = None
    return None


def _build_m1_spread_array(m1_time: np.ndarray, series_time: np.ndarray,
                           series_spread_price: np.ndarray,
                           fallback: float) -> np.ndarray:
    """Build a per-M1-bar spread array (price units) via vectorized lookup."""
    idx = np.searchsorted(series_time, m1_time, side='left')
    n = len(series_time)
    in_range = (idx < n)
    safe_idx = np.where(in_range, idx, 0)
    exact_hit = in_range & (series_time[safe_idx] == m1_time)
    sp = np.full(len(m1_time), float(fallback), dtype=np.float64)
    sp[exact_hit] = series_spread_price[safe_idx[exact_hit]].astype(np.float64)
    return sp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PendingOrder:
    type: str
    price: float
    sl: float
    tp: float
    placed_bar: int
    expiry_bars: int
    be_dist: float = 0.0
    trail_dist: float = 0.0


@dataclass
class Position:
    type: str
    entry_price: float
    sl: float
    tp: float
    entry_bar: int
    max_hold_m1: int
    be_dist: float
    trail_dist: float
    be_done: bool = False
    high_water: float = 0.0
    low_water: float = 0.0
    spread_cost: float = 0.0


@dataclass
class TradeResult:
    entry_bar_m1: int
    exit_bar_m1: int
    entry_price: float
    exit_price: float
    pnl: float
    type: str
    exit_reason: str
    m5_signal_bar: int


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------
IDLE = 0
BRACKET_PENDING = 1
POSITION_OPEN = 2


# ---------------------------------------------------------------------------
# SimEngine
# ---------------------------------------------------------------------------
class SimEngine:
    """Core bracket-EA simulator (EURUSD-tuned, MT5-faithful)."""

    def __init__(self, m1_data: dict, m5_data: dict, config: dict, costs: dict,
                 ea_family: Optional[str] = None):
        for key in ('high', 'low', 'close', 'open'):
            if m1_data.get(key) is None or len(m1_data[key]) == 0:
                raise ValueError(f"M1 data '{key}' is missing or empty")

        self.m1_h = np.ascontiguousarray(m1_data['high'], dtype=np.float64)
        self.m1_l = np.ascontiguousarray(m1_data['low'], dtype=np.float64)
        self.m1_c = np.ascontiguousarray(m1_data['close'], dtype=np.float64)
        self.m1_o = np.ascontiguousarray(m1_data['open'], dtype=np.float64)
        self.n_m1 = len(self.m1_h)

        self.m1_dates = m1_data.get('dates', None)

        self.m5_h = np.ascontiguousarray(m5_data['high'], dtype=np.float64)
        self.m5_l = np.ascontiguousarray(m5_data['low'], dtype=np.float64)
        self.m5_c = np.ascontiguousarray(m5_data['close'], dtype=np.float64)
        self.m5_o = np.ascontiguousarray(m5_data['open'], dtype=np.float64)
        self.m5_hours = m5_data.get('hours', None)
        self.m5_dates = m5_data.get('dates', None)
        self.m5_years = m5_data.get('years', None)
        self.m5_atr14 = m5_data.get('atr14', None)
        self.n_m5 = len(self.m5_h)

        self.bracket_bars = config.get('bracket_bars', 3)
        self.max_hold_m1 = config.get('max_hold_bars', 60) * 5
        self.max_trades_per_day = config.get('max_trades_per_day', 20)
        self.daily_loss_cap_pct = config.get('daily_loss_cap_pct', 5.0)
        self.starting_balance = config.get('starting_balance', 10000.0)

        # EURUSD defaults: ~0.7 pip spread, ~0.2 pip slippage, 0 commission.
        self.spread = costs.get('spread', 0.00007)
        self.slippage = costs.get('slippage', 0.00002)
        self.commission = costs.get('commission', 0.0)

        use_variable_spread = costs.get('use_variable_spread', True)
        self.spread_at_m1 = None
        self.spread_median = self.spread
        m1_time_unix = m1_data.get('time_unix', None)
        if use_variable_spread and m1_time_unix is not None and len(m1_time_unix) == self.n_m1:
            series = _load_spread_series()
            if series is not None:
                try:
                    sp_arr = _build_m1_spread_array(
                        np.asarray(m1_time_unix, dtype=np.int64),
                        series['time'], series['spread_price'],
                        fallback=self.spread,
                    )
                    self.spread_at_m1 = sp_arr
                    self.spread_median = float(np.median(series['spread_price']))
                except Exception as e:
                    warnings.warn(
                        f"[ebb_mt5_sim] variable-spread build failed ({e!r}); "
                        f"falling back to fixed spread={self.spread}.",
                        RuntimeWarning,
                    )

        self.stops_level = costs.get('stops_level', STOPS_LEVEL_PRICE)
        self.freeze_level = costs.get('freeze_level', FREEZE_LEVEL_PRICE)
        self.rejected_for_stops_level = 0
        self.rejected_for_freeze_level = 0

        fam = ea_family if ea_family is not None else config.get('ea_family', 'v5')
        fam = (fam or 'v5').lower()
        if fam not in SIM_BIAS_COEFFICIENTS:
            warnings.warn(
                f"[ebb_mt5_sim] Unknown ea_family={fam!r}; defaulting to 'v5' "
                f"coefficient. Known families: {sorted(SIM_BIAS_COEFFICIENTS)}.",
                RuntimeWarning,
            )
            fam = 'v5'
        self.ea_family = fam
        self.pf_correction = float(SIM_BIAS_COEFFICIENTS[fam]['pf_correction'])

    def run(self, m5_signals: list, m5_to_m1_map: dict) -> dict:
        signal_map = {sig[0]: sig for sig in m5_signals}

        m1_h = self.m1_h
        m1_l = self.m1_l
        m1_c = self.m1_c
        n_m1 = self.n_m1
        spread = self.spread
        spread_arr = self.spread_at_m1
        spread_trigger = self.spread_median
        slippage = self.slippage
        commission = self.commission
        bracket_bars = self.bracket_bars
        max_hold_m1 = self.max_hold_m1
        max_tpd = self.max_trades_per_day
        daily_cap_pct = self.daily_loss_cap_pct
        stops_level_price = self.stops_level
        freeze_level_price = self.freeze_level

        self.rejected_for_stops_level = 0
        self.rejected_for_freeze_level = 0

        state = IDLE
        pending_buy: Optional[PendingOrder] = None
        pending_sell: Optional[PendingOrder] = None
        position: Optional[Position] = None
        bracket_placed_m5 = -1
        signal_m5_bar = -1

        trade_list = []
        balance = self.starting_balance
        current_date = None
        daily_trades = 0
        daily_pnl = 0.0

        year_stats = {}

        def _record_trade(result: TradeResult):
            nonlocal balance, daily_trades, daily_pnl
            trade_list.append(result)
            balance += result.pnl
            daily_trades += 1
            daily_pnl += result.pnl

            yr = self._get_year_for_m1(result.entry_bar_m1, m5_to_m1_map)
            if yr not in year_stats:
                year_stats[yr] = {'wins': 0, 'losses': 0, 'gross_win': 0.0,
                                  'gross_loss': 0.0, 'pnl': 0.0, 'trades': 0}
            ys = year_stats[yr]
            ys['trades'] += 1
            ys['pnl'] += result.pnl
            if result.pnl > 0:
                ys['wins'] += 1
                ys['gross_win'] += result.pnl
            elif result.pnl < 0:
                ys['losses'] += 1
                ys['gross_loss'] += abs(result.pnl)

        for m5_i in range(self.n_m5):
            d = self.m5_dates[m5_i] if self.m5_dates is not None else None
            if d is not None and d != current_date:
                current_date = d
                daily_trades = 0
                daily_pnl = 0.0

            if m5_i not in m5_to_m1_map:
                continue
            m1_start, m1_end = m5_to_m1_map[m5_i]
            m1_end = min(m1_end, n_m1)

            # ---- POSITION_OPEN ----
            if state == POSITION_OPEN:
                pos = position
                closed = False

                for m1k in range(m1_start, m1_end):
                    bars_held = m1k - pos.entry_bar

                    if pos.type == 'long':
                        if m1_h[m1k] > pos.high_water:
                            pos.high_water = m1_h[m1k]

                        sl_hit = m1_l[m1k] <= pos.sl
                        tp_hit = m1_h[m1k] >= pos.tp
                        if sl_hit:
                            pnl = (pos.sl - pos.entry_price) - commission
                            _record_trade(TradeResult(
                                pos.entry_bar, m1k, pos.entry_price, pos.sl,
                                pnl, 'long', 'sl', signal_m5_bar))
                            closed = True
                            break

                        if tp_hit:
                            pnl = (pos.tp - pos.entry_price) - commission
                            _record_trade(TradeResult(
                                pos.entry_bar, m1k, pos.entry_price, pos.tp,
                                pnl, 'long', 'tp', signal_m5_bar))
                            closed = True
                            break

                        if pos.be_dist > 0 and not pos.be_done:
                            if (pos.high_water - pos.entry_price) >= pos.be_dist:
                                new_sl = pos.entry_price + pos.spread_cost
                                if (m1_c[m1k] - new_sl) >= freeze_level_price:
                                    pos.sl = new_sl
                                    pos.be_done = True
                                else:
                                    self.rejected_for_freeze_level += 1

                        if pos.trail_dist > 0 and pos.be_done:
                            new_sl = pos.high_water - pos.trail_dist
                            if new_sl > pos.sl:
                                if (m1_c[m1k] - new_sl) >= freeze_level_price:
                                    pos.sl = new_sl
                                else:
                                    self.rejected_for_freeze_level += 1

                        if bars_held >= max_hold_m1:
                            close_price = m1_c[m1k]
                            unrealized = close_price - pos.entry_price
                            if unrealized < 0:
                                pnl = unrealized - commission
                                _record_trade(TradeResult(
                                    pos.entry_bar, m1k, pos.entry_price, close_price,
                                    pnl, 'long', 'time_stop', signal_m5_bar))
                                closed = True
                                break

                    else:  # short
                        if m1_l[m1k] < pos.low_water:
                            pos.low_water = m1_l[m1k]

                        sl_hit = m1_h[m1k] >= pos.sl
                        tp_hit = m1_l[m1k] <= pos.tp
                        if sl_hit:
                            pnl = (pos.entry_price - pos.sl) - commission
                            _record_trade(TradeResult(
                                pos.entry_bar, m1k, pos.entry_price, pos.sl,
                                pnl, 'short', 'sl', signal_m5_bar))
                            closed = True
                            break

                        if tp_hit:
                            pnl = (pos.entry_price - pos.tp) - commission
                            _record_trade(TradeResult(
                                pos.entry_bar, m1k, pos.entry_price, pos.tp,
                                pnl, 'short', 'tp', signal_m5_bar))
                            closed = True
                            break

                        if pos.be_dist > 0 and not pos.be_done:
                            if (pos.entry_price - pos.low_water) >= pos.be_dist:
                                new_sl = pos.entry_price - pos.spread_cost
                                if (new_sl - m1_c[m1k]) >= freeze_level_price:
                                    pos.sl = new_sl
                                    pos.be_done = True
                                else:
                                    self.rejected_for_freeze_level += 1

                        if pos.trail_dist > 0 and pos.be_done:
                            new_sl = pos.low_water + pos.trail_dist
                            if new_sl < pos.sl:
                                if (new_sl - m1_c[m1k]) >= freeze_level_price:
                                    pos.sl = new_sl
                                else:
                                    self.rejected_for_freeze_level += 1

                        if bars_held >= max_hold_m1:
                            close_price = m1_c[m1k]
                            unrealized = pos.entry_price - close_price
                            if unrealized < 0:
                                pnl = unrealized - commission
                                _record_trade(TradeResult(
                                    pos.entry_bar, m1k, pos.entry_price, close_price,
                                    pnl, 'short', 'time_stop', signal_m5_bar))
                                closed = True
                                break

                if closed:
                    state = IDLE
                    position = None
                continue

            # ---- BRACKET_PENDING ----
            if state == BRACKET_PENDING:
                bracket_age = m5_i - bracket_placed_m5
                if bracket_age > bracket_bars:
                    pending_buy = None
                    pending_sell = None
                    state = IDLE
                else:
                    filled = False
                    fill_m1k = -1
                    for m1k in range(m1_start, m1_end):
                        _sp = spread_arr[m1k] if spread_arr is not None else spread

                        if pending_buy is not None and m1_h[m1k] >= pending_buy.price:
                            entry_price = pending_buy.price + _sp + slippage
                            position = Position(
                                type='long',
                                entry_price=entry_price,
                                sl=pending_buy.sl,
                                tp=pending_buy.tp,
                                entry_bar=m1k,
                                max_hold_m1=max_hold_m1,
                                be_dist=pending_buy.be_dist,
                                trail_dist=pending_buy.trail_dist,
                                be_done=False,
                                high_water=entry_price,
                                low_water=entry_price,
                                spread_cost=_sp,
                            )
                            pending_buy = None
                            pending_sell = None
                            state = POSITION_OPEN
                            filled = True
                            fill_m1k = m1k
                            break

                        if pending_sell is not None and m1_l[m1k] <= pending_sell.price:
                            entry_price = pending_sell.price - _sp - slippage
                            position = Position(
                                type='short',
                                entry_price=entry_price,
                                sl=pending_sell.sl,
                                tp=pending_sell.tp,
                                entry_bar=m1k,
                                max_hold_m1=max_hold_m1,
                                be_dist=pending_sell.be_dist,
                                trail_dist=pending_sell.trail_dist,
                                be_done=False,
                                high_water=entry_price,
                                low_water=entry_price,
                                spread_cost=_sp,
                            )
                            pending_buy = None
                            pending_sell = None
                            state = POSITION_OPEN
                            filled = True
                            fill_m1k = m1k
                            break

                    if filled and fill_m1k + 1 < m1_end:
                        pos = position
                        for m1k in range(fill_m1k + 1, m1_end):
                            if pos.type == 'long':
                                if m1_l[m1k] <= pos.sl:
                                    pnl = (pos.sl - pos.entry_price) - commission
                                    _record_trade(TradeResult(
                                        pos.entry_bar, m1k, pos.entry_price, pos.sl,
                                        pnl, 'long', 'sl', signal_m5_bar))
                                    state = IDLE
                                    position = None
                                    break
                                if m1_h[m1k] >= pos.tp:
                                    pnl = (pos.tp - pos.entry_price) - commission
                                    _record_trade(TradeResult(
                                        pos.entry_bar, m1k, pos.entry_price, pos.tp,
                                        pnl, 'long', 'tp', signal_m5_bar))
                                    state = IDLE
                                    position = None
                                    break
                            else:
                                if m1_h[m1k] >= pos.sl:
                                    pnl = (pos.entry_price - pos.sl) - commission
                                    _record_trade(TradeResult(
                                        pos.entry_bar, m1k, pos.entry_price, pos.sl,
                                        pnl, 'short', 'sl', signal_m5_bar))
                                    state = IDLE
                                    position = None
                                    break
                                if m1_l[m1k] <= pos.tp:
                                    pnl = (pos.entry_price - pos.tp) - commission
                                    _record_trade(TradeResult(
                                        pos.entry_bar, m1k, pos.entry_price, pos.tp,
                                        pnl, 'short', 'tp', signal_m5_bar))
                                    state = IDLE
                                    position = None
                                    break

                    continue

            # ---- IDLE: check for new signal ----
            if daily_trades >= max_tpd:
                continue
            if daily_pnl < 0 and abs(daily_pnl) >= balance * daily_cap_pct / 100.0:
                continue

            if m5_i not in signal_map:
                continue

            sig = signal_map[m5_i]
            _, buy_level, sell_level, sl_dist, tp_dist, be_dist, trail_dist = sig

            current_price = self.m5_c[m5_i]
            stops_level = stops_level_price

            buy_trigger_ok = buy_level is not None and \
                (buy_level - (current_price + spread_trigger)) >= stops_level
            sell_trigger_ok = sell_level is not None and \
                (current_price - sell_level) >= stops_level

            if buy_trigger_ok and (sl_dist < stops_level or tp_dist < stops_level):
                buy_trigger_ok = False
                self.rejected_for_stops_level += 1
            if sell_trigger_ok and (sl_dist < stops_level or tp_dist < stops_level):
                sell_trigger_ok = False
                self.rejected_for_stops_level += 1

            if not (buy_trigger_ok or sell_trigger_ok):
                continue

            pending_buy = None
            pending_sell = None

            if buy_trigger_ok:
                pending_buy = PendingOrder(
                    type='buy_stop',
                    price=buy_level,
                    sl=buy_level - sl_dist,
                    tp=buy_level + tp_dist,
                    placed_bar=m5_i,
                    expiry_bars=bracket_bars,
                    be_dist=be_dist,
                    trail_dist=trail_dist,
                )

            if sell_trigger_ok:
                pending_sell = PendingOrder(
                    type='sell_stop',
                    price=sell_level,
                    sl=sell_level + sl_dist,
                    tp=sell_level - tp_dist,
                    placed_bar=m5_i,
                    expiry_bars=bracket_bars,
                    be_dist=be_dist,
                    trail_dist=trail_dist,
                )

            if pending_buy is not None or pending_sell is not None:
                state = BRACKET_PENDING
                bracket_placed_m5 = m5_i
                signal_m5_bar = m5_i

        if state == POSITION_OPEN and position is not None:
            last_m1 = n_m1 - 1
            if position.type == 'long':
                pnl = (m1_c[last_m1] - position.entry_price) - commission
            else:
                pnl = (position.entry_price - m1_c[last_m1]) - commission
            _record_trade(TradeResult(
                position.entry_bar, last_m1, position.entry_price, m1_c[last_m1],
                pnl, position.type, 'end_of_data', signal_m5_bar))
            state = IDLE
            position = None

        results = self._compute_results(trade_list, year_stats)
        results['pf_raw'] = results['pf']
        results['pf'] = round(results['pf'] * self.pf_correction, 4)
        results['pf_correction'] = self.pf_correction
        results['ea_family'] = self.ea_family
        results['rejected_for_stops_level'] = self.rejected_for_stops_level
        results['rejected_for_freeze_level'] = self.rejected_for_freeze_level
        for _yr, _ys in results.get('per_year', {}).items():
            _ys['pf_raw'] = _ys['pf']
            _ys['pf'] = round(_ys['pf'] * self.pf_correction, 4)
        return results

    def _get_year_for_m1(self, m1_bar: int, m5_to_m1_map: dict) -> int:
        if self.m1_dates is not None and m1_bar < len(self.m1_dates):
            d = self.m1_dates[m1_bar]
            if hasattr(d, 'year'):
                return d.year
        if self.m5_years is not None:
            approx_m5 = m1_bar // 5
            approx_m5 = min(approx_m5, len(self.m5_years) - 1)
            return int(self.m5_years[approx_m5])
        return 0

    @staticmethod
    def _compute_results(trade_list: list, year_stats: dict) -> dict:
        n_trades = len(trade_list)
        if n_trades == 0:
            return {
                'trades': 0, 'pf': 0.0, 'pnl': 0.0, 'win_rate': 0.0,
                'gross_win': 0.0, 'gross_loss': 0.0,
                'wins': 0, 'losses': 0,
                'trade_list': [],
                'per_year': {},
                'max_dd': 0.0,
                'exit_reasons': {},
            }

        pnls = np.array([t.pnl for t in trade_list])
        wins = int(np.sum(pnls > 0))
        losses = int(np.sum(pnls < 0))
        gross_win = float(np.sum(pnls[pnls > 0]))
        gross_loss = float(np.sum(np.abs(pnls[pnls < 0])))
        total_pnl = float(np.sum(pnls))
        pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        wr = wins / n_trades

        cum_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        per_year = {}
        for yr, ys in sorted(year_stats.items()):
            yr_pf = ys['gross_win'] / ys['gross_loss'] if ys['gross_loss'] > 0 else (
                999.0 if ys['gross_win'] > 0 else 0.0)
            per_year[yr] = {
                'trades': ys['trades'],
                'pf': round(yr_pf, 4),
                'pnl': round(ys['pnl'], 2),
                'win_rate': round(ys['wins'] / max(ys['trades'], 1), 4),
                'wins': ys['wins'],
                'losses': ys['losses'],
            }

        exit_reasons = {}
        for t in trade_list:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            'trades': n_trades,
            'pf': round(pf, 4),
            'pnl': round(total_pnl, 2),
            'win_rate': round(wr, 4),
            'gross_win': round(gross_win, 2),
            'gross_loss': round(gross_loss, 2),
            'wins': wins,
            'losses': losses,
            'max_dd': round(max_dd, 2),
            'exit_reasons': exit_reasons,
            'trade_list': trade_list,
            'per_year': per_year,
        }


# ---------------------------------------------------------------------------
# Utility: build m5_to_m1_map from timestamps
# ---------------------------------------------------------------------------
def build_m5_to_m1_map(m1_times: np.ndarray, m5_times: np.ndarray) -> dict:
    mapping = {}
    m1_idx = 0
    n_m1 = len(m1_times)
    n_m5 = len(m5_times)

    for m5_i in range(n_m5):
        t_start = m5_times[m5_i]
        t_end = m5_times[m5_i + 1] if m5_i + 1 < n_m5 else m5_times[m5_i] + 300

        while m1_idx < n_m1 and m1_times[m1_idx] < t_start:
            m1_idx += 1

        start = m1_idx
        end = start
        while end < n_m1 and m1_times[end] < t_end:
            end += 1

        mapping[m5_i] = (start, end)

    return mapping


def build_m5_to_m1_map_uniform(n_m1: int, n_m5: int, bars_per_m5: int = 5) -> dict:
    mapping = {}
    for m5_i in range(n_m5):
        m1_start = m5_i * bars_per_m5
        m1_end = min(m1_start + bars_per_m5, n_m1)
        mapping[m5_i] = (m1_start, m1_end)
    return mapping


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("ebb_mt5_sim.py — EURUSD core engine loaded OK")
    print("Classes: PendingOrder, Position, TradeResult, SimEngine")
    print("Helpers: build_m5_to_m1_map, build_m5_to_m1_map_uniform")

    np.random.seed(42)
    n_m1 = 1000
    n_m5 = 200
    # EURUSD-scale price walk centered around 1.05.
    prices = 1.05 + np.cumsum(np.random.randn(n_m1) * 0.00005)
    m1_data = {
        'open': prices,
        'high': prices + np.abs(np.random.randn(n_m1) * 0.0001),
        'low': prices - np.abs(np.random.randn(n_m1) * 0.0001),
        'close': prices + np.random.randn(n_m1) * 0.00003,
    }
    m5_data = {
        'open': prices[::5][:n_m5],
        'high': np.array([m1_data['high'][i*5:(i+1)*5].max() for i in range(n_m5)]),
        'low': np.array([m1_data['low'][i*5:(i+1)*5].min() for i in range(n_m5)]),
        'close': prices[4::5][:n_m5],
        'hours': np.tile(np.arange(24), n_m5 // 24 + 1)[:n_m5],
        'dates': np.array([None] * n_m5),
        'years': np.full(n_m5, 2024),
    }

    config = {'bracket_bars': 3, 'max_hold_bars': 60}
    costs = {'spread': 0.00007, 'slippage': 0.00002, 'commission': 0.0,
             'use_variable_spread': False}

    engine = SimEngine(m1_data, m5_data, config, costs)
    m5_map = build_m5_to_m1_map_uniform(n_m1, n_m5)

    signals = []
    for i in range(10, n_m5, 40):
        c = m5_data['close'][i]
        signals.append((i, c + 0.00005, c - 0.00005, 0.0005, 0.0008, 0.0003, 0.0002))

    result = engine.run(signals, m5_map)
    print(f"\nSmoke test: {result['trades']} trades, PF={result['pf']}, "
          f"PnL={result['pnl']}, WR={result['win_rate']}")
    if result['exit_reasons']:
        print(f"Exit reasons: {result['exit_reasons']}")
    print("PASS" if result['trades'] > 0 else "WARN: no trades")
