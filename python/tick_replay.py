#!/usr/bin/env python3
"""Tick-faithful replay engine for EURUSD — EBB port of GBB tick_replay.py.

Architecture: phase_2a_tick_replay.md (in GBB)
Baseline ref: mt5_sim.py (in EBB python/)

EBB-specific deltas vs GBB:
  - STOPS_LEVEL_PRICE / FREEZE_LEVEL_PRICE = 0.00005 (5 EURUSD points)
  - Default lot=0.01, contract_size=100_000 (EURUSD standard contract)
  - load_ticks() expected schema unchanged (time int64 ms, bid/ask float)

Phases:
  2A.3 -- pure-Python state-machine engine, scans full ticks (slow)
  2A.4a -- signal-window sliced pure-Python engine
  2A.4b -- numba @njit kernel + signal-window slicing (fast path)

Public API mirrors SimEngine.run() for drop-in swap.
"""

from __future__ import annotations

import time as _time
import datetime
import numpy as np
import pyarrow.parquet as pq
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Broker constants (EURUSD — mirror mt5_sim.py)
# ---------------------------------------------------------------------------
STOPS_LEVEL_PRICE = 0.00005   # 5 pts × 0.00001 USD/pt
FREEZE_LEVEL_PRICE = 0.00005
MAX_GAP_SECONDS = 60          # gap > 60 s → weekend/session gap → close position

# Exit-reason codes
_EXIT_NONE = 0
_EXIT_TP = 1
_EXIT_SL = 2
_EXIT_TIME = 3
_EXIT_SESSION_GAP = 4
_EXIT_END_OF_DATA = 5
_EXIT_UNFILLED = 6
EXIT_CODE_TO_STR = {
    _EXIT_NONE: 'none',
    _EXIT_TP: 'tp',
    _EXIT_SL: 'sl',
    _EXIT_TIME: 'time',
    _EXIT_SESSION_GAP: 'session_gap',
    _EXIT_END_OF_DATA: 'end_of_data',
    _EXIT_UNFILLED: 'unfilled',
}

_DIR_NONE = 0
_DIR_LONG = 1
_DIR_SHORT = 2

NAN64 = np.float64(np.nan)


@dataclass
class TradeResult:
    signal_idx: int
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl_gross: float
    pnl_net: float
    exit_reason: str
    entry_time_ms: int
    exit_time_ms: int
    spread_at_fill: float


# ---------------------------------------------------------------------------
# Tick loader
# ---------------------------------------------------------------------------

def load_ticks(parquet_path: str) -> dict:
    t0 = _time.perf_counter()
    print(f"[ebb_tick_replay] Loading {parquet_path} ...")

    table = pq.read_table(parquet_path, columns=['time', 'bid', 'ask'])

    import pyarrow as pa
    time_col = table.column('time')
    time_ms = time_col.cast(pa.int64()).to_numpy().astype(np.int64)

    bid = table.column('bid').to_numpy().astype(np.float64)
    ask = table.column('ask').to_numpy().astype(np.float64)

    mid = (bid + ask) * 0.5
    spread = (ask - bid).astype(np.float32)

    elapsed = _time.perf_counter() - t0
    n = len(time_ms)
    mem_mb = (time_ms.nbytes + bid.nbytes + ask.nbytes + mid.nbytes + spread.nbytes) / 1e6
    print(f"[ebb_tick_replay] Loaded {n:,} ticks in {elapsed:.2f}s  |  mem={mem_mb:.1f} MB")

    return {
        'time_ms': time_ms,
        'bid': bid,
        'ask': ask,
        'mid': mid,
        'spread': spread,
    }


def load_ticks_multi(parquet_paths: list) -> dict:
    parts = [load_ticks(p) for p in parquet_paths]

    time_ms = np.concatenate([p['time_ms'] for p in parts])
    bid = np.concatenate([p['bid'] for p in parts])
    ask = np.concatenate([p['ask'] for p in parts])
    order = np.argsort(time_ms, kind='stable')
    time_ms = time_ms[order]
    bid = bid[order]
    ask = ask[order]
    mid = (bid + ask) * 0.5
    spread = (ask - bid).astype(np.float32)
    print(f"[ebb_tick_replay] merged {len(parts)} files -> {len(time_ms):,} ticks")
    return {
        'time_ms': time_ms,
        'bid': bid,
        'ask': ask,
        'mid': mid,
        'spread': spread,
    }


# ---------------------------------------------------------------------------
# M5 -> tick map  (vectorized + dict variants)
# ---------------------------------------------------------------------------

def build_m5_to_tick_arrays(tick_time_ms: np.ndarray,
                            m5_time_unix: np.ndarray) -> tuple:
    m5_open_ms = (m5_time_unix.astype(np.int64) * 1000)
    m5_close_ms = m5_open_ms + 300_000
    starts = np.searchsorted(tick_time_ms, m5_open_ms, side='left').astype(np.int64)
    ends = np.searchsorted(tick_time_ms, m5_close_ms, side='left').astype(np.int64)
    return starts, ends


def build_m5_to_tick_map(tick_time_ms: np.ndarray, m5_time_unix: np.ndarray) -> dict:
    starts, ends = build_m5_to_tick_arrays(tick_time_ms, m5_time_unix)
    return {int(i): (int(starts[i]), int(ends[i])) for i in range(len(starts))}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(ticks: dict) -> dict:
    time_ms = ticks['time_ms']
    spread = ticks['spread']

    days_ms = (time_ms // 86_400_000).astype(np.int32)
    unique_days, day_counts = np.unique(days_ms, return_counts=True)

    day_labels = []
    for d in unique_days:
        dt = datetime.datetime.utcfromtimestamp(int(d) * 86400)
        day_labels.append(dt.strftime('%Y-%m-%d'))

    per_day = dict(zip(day_labels, day_counts.tolist()))

    weekend_ticks = 0
    for d, cnt in zip(unique_days, day_counts):
        dt = datetime.datetime.utcfromtimestamp(int(d) * 86400)
        if dt.weekday() >= 5:
            weekend_ticks += int(cnt)

    diff_ms = np.diff(time_ms)
    gap_mask = diff_ms > 60_000
    n_gaps = int(np.sum(gap_mask))
    gap_sizes_s = (diff_ms[gap_mask] / 1000).astype(int)

    # EURUSD spread band: filter 0 < spread < 0.005 (50 pips), much tighter than XAU.
    valid_spread = spread[(spread > 0) & (spread < 0.005)]
    spread_stats = {
        'median': float(np.median(valid_spread)),
        'p5': float(np.percentile(valid_spread, 5)),
        'p95': float(np.percentile(valid_spread, 95)),
        'mean': float(np.mean(valid_spread)),
    }

    return {
        'total_ticks': len(time_ms),
        'per_day': per_day,
        'weekend_ticks': weekend_ticks,
        'n_gaps_over_60s': n_gaps,
        'largest_gaps_s': sorted(gap_sizes_s.tolist(), reverse=True)[:10],
        'spread': spread_stats,
    }


# ---------------------------------------------------------------------------
# Single-signal replay (pure Python — kept for tape parity).
# ---------------------------------------------------------------------------

def _replay_one_signal(
    ticks: dict,
    signal: dict,
    m5_to_tick_map: dict,
    costs: dict,
    max_hold_m5: int = 24,
) -> TradeResult:
    time_ms = ticks['time_ms']
    bid_arr = ticks['bid']
    ask_arr = ticks['ask']
    n_ticks = len(time_ms)

    m5_i = signal['m5_i']
    buy_level = signal.get('buy_level')
    sell_level = signal.get('sell_level')
    sl_buy = signal.get('sl_buy')
    tp_buy = signal.get('tp_buy')
    sl_sell = signal.get('sl_sell')
    tp_sell = signal.get('tp_sell')
    be_dist = signal.get('be_dist', 0.0) or 0.0
    trail_dist = signal.get('trail_dist', 0.0) or 0.0

    # EURUSD defaults: lot=0.01, contract=100_000 EUR/lot.
    lot = costs.get('lot_size', 0.01)
    contract = costs.get('contract_size', 100_000)

    _UNFILLED = TradeResult(
        signal_idx=m5_i, direction='none', entry_price=0.0, exit_price=0.0,
        sl=0.0, tp=0.0, pnl_gross=0.0, pnl_net=0.0,
        exit_reason='unfilled', entry_time_ms=0, exit_time_ms=0,
        spread_at_fill=0.0,
    )

    placement_m5 = m5_i + 1
    if placement_m5 not in m5_to_tick_map:
        return _UNFILLED

    scan_start, _ = m5_to_tick_map[placement_m5]

    end_m5 = placement_m5 + max_hold_m5
    if end_m5 in m5_to_tick_map:
        _, scan_end = m5_to_tick_map[end_m5]
    else:
        scan_end = n_ticks
    scan_end = min(scan_end, n_ticks)
    if scan_start >= scan_end:
        return _UNFILLED

    def _valid_bracket(entry, sl, tp, direction):
        if direction == 'long':
            return (entry - sl) >= STOPS_LEVEL_PRICE and (tp - entry) >= STOPS_LEVEL_PRICE
        return (sl - entry) >= STOPS_LEVEL_PRICE and (entry - tp) >= STOPS_LEVEL_PRICE

    filled = False
    direction = None
    entry_price = 0.0
    sl_level = 0.0
    tp_level = 0.0
    entry_tick_idx = -1
    spread_at_fill = 0.0

    for i in range(scan_start, scan_end):
        cur_bid = bid_arr[i]
        cur_ask = ask_arr[i]
        cur_time = time_ms[i]
        if i > scan_start and (cur_time - time_ms[i - 1]) > 60_000:
            return _UNFILLED
        if buy_level is not None and sl_buy is not None and tp_buy is not None:
            if cur_ask >= buy_level and _valid_bracket(buy_level, sl_buy, tp_buy, 'long'):
                filled = True; direction = 'long'
                entry_price = cur_ask; sl_level = sl_buy; tp_level = tp_buy
                spread_at_fill = float(cur_ask - cur_bid)
                entry_tick_idx = i
                break
        if sell_level is not None and sl_sell is not None and tp_sell is not None:
            if cur_bid <= sell_level and _valid_bracket(sell_level, sl_sell, tp_sell, 'short'):
                filled = True; direction = 'short'
                entry_price = cur_bid; sl_level = sl_sell; tp_level = tp_sell
                spread_at_fill = float(cur_ask - cur_bid)
                entry_tick_idx = i
                break

    if not filled:
        return _UNFILLED

    entry_time_ms_val = int(time_ms[entry_tick_idx])
    time_stop_ms = entry_time_ms_val + max_hold_m5 * 300_000
    be_done = False
    high_water = entry_price
    low_water = entry_price
    spread_cost = spread_at_fill
    exit_price = 0.0
    exit_reason = 'time'
    exit_time_ms_val = entry_time_ms_val
    exited = False

    for i in range(entry_tick_idx + 1, scan_end):
        cur_bid = bid_arr[i]; cur_ask = ask_arr[i]
        cur_time = int(time_ms[i])
        if (cur_time - int(time_ms[i - 1])) > 60_000:
            exit_price = cur_bid if direction == 'long' else cur_ask
            exit_reason = 'session_gap'
            exit_time_ms_val = cur_time
            exited = True
            break
        if cur_time >= time_stop_ms:
            unrealized = (cur_bid - entry_price) if direction == 'long' \
                else (entry_price - cur_ask)
            if unrealized < 0:
                exit_price = cur_bid if direction == 'long' else cur_ask
                exit_reason = 'time'
                exit_time_ms_val = cur_time
                exited = True
                break
        if direction == 'long':
            if cur_ask > high_water:
                high_water = cur_ask
            if cur_bid <= sl_level:
                exit_price = cur_bid; exit_reason = 'sl'
                exit_time_ms_val = cur_time; exited = True; break
            if cur_bid >= tp_level:
                exit_price = cur_bid; exit_reason = 'tp'
                exit_time_ms_val = cur_time; exited = True; break
            if be_dist > 0 and not be_done:
                if (high_water - entry_price) >= be_dist:
                    new_sl = entry_price + spread_cost
                    if new_sl > sl_level and (cur_bid - new_sl) >= FREEZE_LEVEL_PRICE:
                        sl_level = new_sl; be_done = True
            if trail_dist > 0 and be_done:
                new_sl = high_water - trail_dist
                if new_sl > sl_level and (cur_bid - new_sl) >= FREEZE_LEVEL_PRICE:
                    sl_level = new_sl
        else:
            if cur_bid < low_water:
                low_water = cur_bid
            if cur_ask >= sl_level:
                exit_price = cur_ask; exit_reason = 'sl'
                exit_time_ms_val = cur_time; exited = True; break
            if cur_ask <= tp_level:
                exit_price = cur_ask; exit_reason = 'tp'
                exit_time_ms_val = cur_time; exited = True; break
            if be_dist > 0 and not be_done:
                if (entry_price - low_water) >= be_dist:
                    new_sl = entry_price - spread_cost
                    if new_sl < sl_level and (new_sl - cur_ask) >= FREEZE_LEVEL_PRICE:
                        sl_level = new_sl; be_done = True
            if trail_dist > 0 and be_done:
                new_sl = low_water + trail_dist
                if new_sl < sl_level and (new_sl - cur_ask) >= FREEZE_LEVEL_PRICE:
                    sl_level = new_sl

    if not exited:
        last_i = min(scan_end - 1, n_ticks - 1)
        exit_price = bid_arr[last_i] if direction == 'long' else ask_arr[last_i]
        exit_reason = 'time'
        exit_time_ms_val = int(time_ms[last_i])

    if direction == 'long':
        pnl_gross = (exit_price - entry_price) * lot * contract
    else:
        pnl_gross = (entry_price - exit_price) * lot * contract
    spread_cost_usd = spread_at_fill * lot * contract
    pnl_net = pnl_gross - spread_cost_usd

    return TradeResult(
        signal_idx=m5_i, direction=direction,
        entry_price=entry_price, exit_price=exit_price,
        sl=sl_level, tp=tp_level,
        pnl_gross=pnl_gross, pnl_net=pnl_net,
        exit_reason=exit_reason,
        entry_time_ms=entry_time_ms_val,
        exit_time_ms=exit_time_ms_val,
        spread_at_fill=spread_at_fill,
    )


# ===========================================================================
# Numba JIT kernel
# ===========================================================================
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return deco


@njit(cache=True, fastmath=False)
def _replay_signals_jit(
    tick_time_ms,
    tick_bid,
    tick_ask,
    sig_m5_idx,
    sig_buy,
    sig_sell,
    sig_sl_dist,
    sig_tp_dist,
    sig_be_dist,
    sig_trail_dist,
    sig_date_ord,
    sig_year,
    m5_tick_start,
    m5_tick_end,
    bracket_bars,
    max_hold_m5,
    max_tpd,
    daily_cap_pct,
    starting_balance,
    lot,
    contract,
    stops_level,
    freeze_level,
    out_signal_m5,
    out_dir,
    out_entry_price,
    out_exit_price,
    out_sl,
    out_tp,
    out_pnl_net,
    out_exit_code,
    out_entry_ms,
    out_exit_ms,
    out_year,
    out_spread,
):
    n_ticks = tick_time_ms.shape[0]
    n_sig = sig_m5_idx.shape[0]
    n_m5 = m5_tick_start.shape[0]
    bal = starting_balance
    cur_date = -1
    daily_trades = 0
    daily_pnl = 0.0
    next_free_m5 = -1
    n_out = 0

    for k in range(n_sig):
        m5_i = sig_m5_idx[k]
        if m5_i < next_free_m5:
            continue

        d = sig_date_ord[k]
        if d != cur_date:
            cur_date = d
            daily_trades = 0
            daily_pnl = 0.0

        if daily_trades >= max_tpd:
            continue
        if daily_pnl < 0.0 and (-daily_pnl) >= bal * daily_cap_pct / 100.0:
            continue

        buy_lvl = sig_buy[k]
        sell_lvl = sig_sell[k]
        sl_dist = sig_sl_dist[k]
        tp_dist = sig_tp_dist[k]
        be_dist = sig_be_dist[k]
        trail_dist = sig_trail_dist[k]

        buy_ok = (buy_lvl == buy_lvl) and (sl_dist >= stops_level) and (tp_dist >= stops_level)
        sell_ok = (sell_lvl == sell_lvl) and (sl_dist >= stops_level) and (tp_dist >= stops_level)
        if not (buy_ok or sell_ok):
            continue

        sl_buy = buy_lvl - sl_dist if buy_ok else 0.0
        tp_buy = buy_lvl + tp_dist if buy_ok else 0.0
        sl_sell = sell_lvl + sl_dist if sell_ok else 0.0
        tp_sell = sell_lvl - tp_dist if sell_ok else 0.0

        placement_m5 = m5_i + 1
        if placement_m5 >= n_m5:
            continue
        scan_start = m5_tick_start[placement_m5]

        end_m5 = placement_m5 + max_hold_m5
        if end_m5 >= n_m5:
            scan_end = n_ticks
        else:
            scan_end = m5_tick_end[end_m5]
        if scan_end > n_ticks:
            scan_end = n_ticks
        if scan_start >= scan_end:
            continue

        bracket_expire_tick = scan_end
        bracket_expire_m5 = placement_m5 + bracket_bars
        if bracket_expire_m5 >= n_m5:
            bracket_expire_m5 = n_m5 - 1

        filled = False
        direction = _DIR_NONE
        entry_price = 0.0
        sl_level = 0.0
        tp_level = 0.0
        entry_tick_idx = -1
        spread_at_fill = 0.0

        i = scan_start
        prev_t = tick_time_ms[i] if i < n_ticks else 0
        while i < bracket_expire_tick:
            cur_bid = tick_bid[i]
            cur_ask = tick_ask[i]
            cur_time = tick_time_ms[i]
            if i > scan_start and (cur_time - prev_t) > 60_000:
                break
            prev_t = cur_time

            if buy_ok and cur_ask >= buy_lvl:
                filled = True
                direction = _DIR_LONG
                entry_price = cur_ask
                sl_level = sl_buy
                tp_level = tp_buy
                spread_at_fill = cur_ask - cur_bid
                entry_tick_idx = i
                break

            if sell_ok and cur_bid <= sell_lvl:
                filled = True
                direction = _DIR_SHORT
                entry_price = cur_bid
                sl_level = sl_sell
                tp_level = tp_sell
                spread_at_fill = cur_ask - cur_bid
                entry_tick_idx = i
                break

            i += 1

        if not filled:
            cand = m5_i + bracket_bars + 1
            if cand > next_free_m5:
                next_free_m5 = cand
            continue

        entry_time_ms_val = tick_time_ms[entry_tick_idx]
        time_stop_ms = entry_time_ms_val + max_hold_m5 * 300_000

        be_done = False
        high_water = entry_price
        low_water = entry_price
        spread_cost = spread_at_fill
        exit_price = 0.0
        exit_code = _EXIT_TIME
        exit_time_val = entry_time_ms_val
        exited = False

        prev_t = entry_time_ms_val
        j = entry_tick_idx + 1
        while j < scan_end:
            cur_bid = tick_bid[j]
            cur_ask = tick_ask[j]
            cur_time = tick_time_ms[j]

            if (cur_time - prev_t) > 60_000:
                if direction == _DIR_LONG:
                    exit_price = cur_bid
                else:
                    exit_price = cur_ask
                exit_code = _EXIT_SESSION_GAP
                exit_time_val = cur_time
                exited = True
                break
            prev_t = cur_time

            if cur_time >= time_stop_ms:
                if direction == _DIR_LONG:
                    unrealized = cur_bid - entry_price
                    if unrealized < 0.0:
                        exit_price = cur_bid
                        exit_code = _EXIT_TIME
                        exit_time_val = cur_time
                        exited = True
                        break
                else:
                    unrealized = entry_price - cur_ask
                    if unrealized < 0.0:
                        exit_price = cur_ask
                        exit_code = _EXIT_TIME
                        exit_time_val = cur_time
                        exited = True
                        break

            if direction == _DIR_LONG:
                if cur_ask > high_water:
                    high_water = cur_ask
                if cur_bid <= sl_level:
                    exit_price = cur_bid
                    exit_code = _EXIT_SL
                    exit_time_val = cur_time
                    exited = True
                    break
                if cur_bid >= tp_level:
                    exit_price = cur_bid
                    exit_code = _EXIT_TP
                    exit_time_val = cur_time
                    exited = True
                    break
                if be_dist > 0.0 and not be_done:
                    if (high_water - entry_price) >= be_dist:
                        new_sl = entry_price + spread_cost
                        if new_sl > sl_level and (cur_bid - new_sl) >= freeze_level:
                            sl_level = new_sl
                            be_done = True
                if trail_dist > 0.0 and be_done:
                    new_sl = high_water - trail_dist
                    if new_sl > sl_level and (cur_bid - new_sl) >= freeze_level:
                        sl_level = new_sl
            else:
                if cur_bid < low_water:
                    low_water = cur_bid
                if cur_ask >= sl_level:
                    exit_price = cur_ask
                    exit_code = _EXIT_SL
                    exit_time_val = cur_time
                    exited = True
                    break
                if cur_ask <= tp_level:
                    exit_price = cur_ask
                    exit_code = _EXIT_TP
                    exit_time_val = cur_time
                    exited = True
                    break
                if be_dist > 0.0 and not be_done:
                    if (entry_price - low_water) >= be_dist:
                        new_sl = entry_price - spread_cost
                        if new_sl < sl_level and (new_sl - cur_ask) >= freeze_level:
                            sl_level = new_sl
                            be_done = True
                if trail_dist > 0.0 and be_done:
                    new_sl = low_water + trail_dist
                    if new_sl < sl_level and (new_sl - cur_ask) >= freeze_level:
                        sl_level = new_sl

            j += 1

        if not exited:
            last_idx = scan_end - 1
            if last_idx >= n_ticks:
                last_idx = n_ticks - 1
            if direction == _DIR_LONG:
                exit_price = tick_bid[last_idx]
            else:
                exit_price = tick_ask[last_idx]
            exit_code = _EXIT_TIME
            exit_time_val = tick_time_ms[last_idx]

        if direction == _DIR_LONG:
            pnl_gross = (exit_price - entry_price) * lot * contract
        else:
            pnl_gross = (entry_price - exit_price) * lot * contract
        pnl_net = pnl_gross - spread_at_fill * lot * contract

        out_signal_m5[n_out] = m5_i
        out_dir[n_out] = direction
        out_entry_price[n_out] = entry_price
        out_exit_price[n_out] = exit_price
        out_sl[n_out] = sl_level
        out_tp[n_out] = tp_level
        out_pnl_net[n_out] = pnl_net
        out_exit_code[n_out] = exit_code
        out_entry_ms[n_out] = entry_time_ms_val
        out_exit_ms[n_out] = exit_time_val
        out_year[n_out] = sig_year[k]
        out_spread[n_out] = spread_at_fill
        n_out += 1

        bal += pnl_net
        daily_trades += 1
        daily_pnl += pnl_net

        cur_m5 = placement_m5
        while cur_m5 < n_m5 - 1:
            ce = m5_tick_end[cur_m5]
            if ce > 0 and ce <= n_ticks and tick_time_ms[ce - 1] >= exit_time_val:
                break
            cur_m5 += 1
        if cur_m5 > next_free_m5:
            next_free_m5 = cur_m5

    return n_out


# ===========================================================================
# TickReplayEngine
# ===========================================================================

class TickReplayEngine:
    """Tick-level replay engine. Three modes:
      - run_full(): pure-Python state machine, full tick scan (slow baseline)
      - run_sliced(): pure-Python state machine, signal-window slicing
      - run_jit(): numba JIT kernel + signal-window slicing (fast path)
    Default run() = run_jit() if numba available, else run_sliced().
    """

    def __init__(self, tick_data: dict, m5_data: dict, config: dict, costs: dict,
                 ea_family: Optional[str] = None,
                 m5_to_tick_map=None,
                 m5_tick_start=None,
                 m5_tick_end=None):
        self.ticks = tick_data
        self.m5_data = m5_data
        self.config = config
        self.costs = costs
        self.ea_family = ea_family

        if m5_tick_start is not None and m5_tick_end is not None:
            self.m5_tick_start = m5_tick_start
            self.m5_tick_end = m5_tick_end
            self.m5_to_tick_map = None
        elif 'time' in m5_data:
            starts, ends = build_m5_to_tick_arrays(
                tick_data['time_ms'], np.asarray(m5_data['time'], dtype=np.int64)
            )
            self.m5_tick_start = starts
            self.m5_tick_end = ends
            self.m5_to_tick_map = m5_to_tick_map
        else:
            self.m5_tick_start = None
            self.m5_tick_end = None
            self.m5_to_tick_map = m5_to_tick_map or {}

    def _signal_in_coverage(self, m5_i: int) -> bool:
        if self.m5_tick_start is None:
            return True
        if m5_i + 1 >= len(self.m5_tick_start):
            return False
        start = self.m5_tick_start[m5_i + 1]
        end = self.m5_tick_end[m5_i + 1]
        return start < end

    def run(self, m5_signals, m5_to_tick_map=None):
        if _HAS_NUMBA and self.m5_tick_start is not None:
            return self.run_jit(m5_signals)
        return self.run_sliced(m5_signals, m5_to_tick_map)

    def run_full(self, m5_signals, m5_to_tick_map=None):
        return self._run_pure_py(m5_signals, m5_to_tick_map, sliced=False)

    def run_sliced(self, m5_signals, m5_to_tick_map=None):
        return self._run_pure_py(m5_signals, m5_to_tick_map, sliced=True)

    def _run_pure_py(self, m5_signals, m5_to_tick_map, sliced: bool):
        if m5_to_tick_map is None:
            if self.m5_to_tick_map is None and self.m5_tick_start is not None:
                self.m5_to_tick_map = {
                    int(i): (int(self.m5_tick_start[i]), int(self.m5_tick_end[i]))
                    for i in range(len(self.m5_tick_start))
                }
            tick_map = self.m5_to_tick_map
        else:
            tick_map = m5_to_tick_map

        cfg = self.config
        costs = self.costs
        bracket_bars = int(cfg.get('bracket_bars', 4))
        max_hold_m5 = int(cfg.get('max_hold_bars', 24))
        max_tpd = int(cfg.get('max_trades_per_day', 20))
        daily_cap_pct = float(cfg.get('daily_loss_cap_pct', 5.0))
        starting_balance = float(cfg.get('starting_balance', 10000.0))
        lot = costs.get('lot_size', 0.01)
        contract = costs.get('contract_size', 100_000)

        m5_dates = self.m5_data.get('dates', None)
        n_m5 = self.m5_data.get('n_m5', 0)
        if not n_m5:
            n_m5 = len(m5_dates) if m5_dates is not None else (
                max(tick_map.keys()) + 1 if tick_map else 0)

        signal_map = {sig[0]: sig for sig in m5_signals}

        IDLE = 0
        BRACKET_PENDING = 1
        POSITION_OPEN = 2
        state = IDLE
        bracket_placed_m5 = -1
        signal_m5_bar = -1
        active_result: Optional[TradeResult] = None
        current_date = None
        daily_trades = 0
        daily_pnl = 0.0
        balance = starting_balance
        trade_list = []
        year_stats = {}

        ticks_time_ms = self.ticks['time_ms']
        n_ticks = len(ticks_time_ms)

        def _get_year(m5_i: int):
            if m5_dates is not None and m5_i < len(m5_dates):
                d = m5_dates[m5_i]
                if d is not None:
                    return d.year if hasattr(d, 'year') else None
            return None

        def _record(trade: TradeResult, origin_m5: int):
            nonlocal balance, daily_trades, daily_pnl
            pnl = trade.pnl_net
            balance += pnl
            daily_trades += 1
            daily_pnl += pnl
            yr = _get_year(origin_m5)
            if yr is not None:
                ys = year_stats.setdefault(yr, {
                    'wins': 0, 'losses': 0, 'gross_win': 0.0,
                    'gross_loss': 0.0, 'pnl': 0.0, 'trades': 0,
                })
                ys['trades'] += 1
                ys['pnl'] += pnl
                if pnl > 0:
                    ys['wins'] += 1
                    ys['gross_win'] += pnl
                elif pnl < 0:
                    ys['losses'] += 1
                    ys['gross_loss'] += abs(pnl)
            trade_list.append(trade)

        for m5_i in range(n_m5):
            if m5_dates is not None and m5_i < len(m5_dates):
                d = m5_dates[m5_i]
                if d is not None and d != current_date:
                    current_date = d
                    daily_trades = 0
                    daily_pnl = 0.0

            if m5_i not in tick_map:
                continue

            tick_start, tick_end = tick_map[m5_i]
            if state == POSITION_OPEN:
                exit_ms = active_result.exit_time_ms
                if exit_ms == 0:
                    continue
                if tick_end > tick_start and tick_end <= n_ticks:
                    bar_end_ms = int(ticks_time_ms[tick_end - 1])
                elif tick_start < n_ticks:
                    bar_end_ms = int(ticks_time_ms[tick_start])
                else:
                    continue
                if exit_ms <= bar_end_ms:
                    _record(active_result, signal_m5_bar)
                    state = IDLE
                    active_result = None
                else:
                    continue

            if state == BRACKET_PENDING:
                bracket_age = m5_i - bracket_placed_m5
                if bracket_age > bracket_bars:
                    state = IDLE
                    active_result = None
                elif active_result is not None and active_result.direction != 'none':
                    state = POSITION_OPEN
                    continue
                else:
                    continue

            if state == IDLE:
                if daily_trades >= max_tpd:
                    continue
                if daily_pnl < 0 and abs(daily_pnl) >= balance * daily_cap_pct / 100.0:
                    continue
                if m5_i not in signal_map:
                    continue
                sig = signal_map[m5_i]
                _, buy_level, sell_level, sl_dist, tp_dist, be_dist, trail_dist = sig
                buy_ok = (buy_level is not None
                          and sl_dist >= STOPS_LEVEL_PRICE
                          and tp_dist >= STOPS_LEVEL_PRICE)
                sell_ok = (sell_level is not None
                           and sl_dist >= STOPS_LEVEL_PRICE
                           and tp_dist >= STOPS_LEVEL_PRICE)
                if not (buy_ok or sell_ok):
                    continue
                signal_dict = {
                    'm5_i': m5_i,
                    'buy_level':  buy_level  if buy_ok  else None,
                    'sell_level': sell_level if sell_ok else None,
                    'sl_buy':  (buy_level  - sl_dist) if (buy_ok  and buy_level  is not None) else None,
                    'tp_buy':  (buy_level  + tp_dist) if (buy_ok  and buy_level  is not None) else None,
                    'sl_sell': (sell_level + sl_dist) if (sell_ok and sell_level is not None) else None,
                    'tp_sell': (sell_level - tp_dist) if (sell_ok and sell_level is not None) else None,
                    'be_dist':    be_dist,
                    'trail_dist': trail_dist,
                }
                if sliced:
                    placement = m5_i + 1
                    upper = placement + bracket_bars + max_hold_m5
                    sub_map = {k: tick_map[k] for k in range(placement, upper + 1)
                               if k in tick_map}
                    result = _replay_one_signal(
                        self.ticks, signal_dict, sub_map, costs, max_hold_m5=max_hold_m5
                    )
                else:
                    result = _replay_one_signal(
                        self.ticks, signal_dict, tick_map, costs, max_hold_m5=max_hold_m5
                    )
                bracket_placed_m5 = m5_i
                signal_m5_bar = m5_i
                active_result = result
                if result.direction != 'none' and result.exit_reason != 'unfilled':
                    state = POSITION_OPEN
                else:
                    state = BRACKET_PENDING

        if state == POSITION_OPEN and active_result is not None and active_result.direction != 'none':
            _record(active_result, signal_m5_bar)

        return self._compute_results(trade_list, year_stats)

    def run_jit(self, m5_signals):
        if not _HAS_NUMBA:
            return self.run_sliced(m5_signals)
        if self.m5_tick_start is None:
            raise RuntimeError("m5_tick_start arrays missing -- pass into __init__")

        cfg = self.config
        costs = self.costs
        bracket_bars = int(cfg.get('bracket_bars', 4))
        max_hold_m5 = int(cfg.get('max_hold_bars', 24))
        max_tpd = int(cfg.get('max_trades_per_day', 20))
        daily_cap_pct = float(cfg.get('daily_loss_cap_pct', 5.0))
        starting_balance = float(cfg.get('starting_balance', 10000.0))
        lot = float(costs.get('lot_size', 0.01))
        contract = float(costs.get('contract_size', 100_000))

        m5_dates = self.m5_data.get('dates', None)
        n_m5 = len(self.m5_tick_start)
        if m5_dates is None:
            raise RuntimeError("m5_data['dates'] required for JIT path (daily reset key)")

        m5_t_start = np.asarray(self.m5_tick_start, dtype=np.int64)
        m5_t_end = np.asarray(self.m5_tick_end, dtype=np.int64)

        filtered_idx = []
        for sig in m5_signals:
            m5_i = int(sig[0])
            if m5_i + 1 >= n_m5:
                continue
            if m5_t_start[m5_i + 1] >= m5_t_end[m5_i + 1]:
                continue
            filtered_idx.append(sig)
        m5_signals = filtered_idx
        s = len(m5_signals)
        if s == 0:
            return self._compute_results([], {})

        sig_m5_idx = np.empty(s, dtype=np.int64)
        sig_buy = np.empty(s, dtype=np.float64)
        sig_sell = np.empty(s, dtype=np.float64)
        sig_sl_dist = np.empty(s, dtype=np.float64)
        sig_tp_dist = np.empty(s, dtype=np.float64)
        sig_be_dist = np.empty(s, dtype=np.float64)
        sig_trail_dist = np.empty(s, dtype=np.float64)
        sig_date_ord = np.empty(s, dtype=np.int64)
        sig_year = np.empty(s, dtype=np.int32)

        for k, sig in enumerate(m5_signals):
            m5_i = int(sig[0])
            sig_m5_idx[k] = m5_i
            sig_buy[k] = sig[1] if sig[1] is not None else NAN64
            sig_sell[k] = sig[2] if sig[2] is not None else NAN64
            sig_sl_dist[k] = sig[3]
            sig_tp_dist[k] = sig[4]
            sig_be_dist[k] = sig[5] if sig[5] is not None else 0.0
            sig_trail_dist[k] = sig[6] if sig[6] is not None else 0.0
            d = m5_dates[m5_i] if m5_i < len(m5_dates) else None
            if d is None:
                sig_date_ord[k] = -1
                sig_year[k] = 0
            else:
                if hasattr(d, 'toordinal'):
                    sig_date_ord[k] = d.toordinal()
                    sig_year[k] = d.year
                else:
                    sig_date_ord[k] = -1
                    sig_year[k] = 0

        out_signal_m5 = np.zeros(s, dtype=np.int64)
        out_dir = np.zeros(s, dtype=np.int32)
        out_entry_price = np.zeros(s, dtype=np.float64)
        out_exit_price = np.zeros(s, dtype=np.float64)
        out_sl = np.zeros(s, dtype=np.float64)
        out_tp = np.zeros(s, dtype=np.float64)
        out_pnl_net = np.zeros(s, dtype=np.float64)
        out_exit_code = np.zeros(s, dtype=np.int32)
        out_entry_ms = np.zeros(s, dtype=np.int64)
        out_exit_ms = np.zeros(s, dtype=np.int64)
        out_year = np.zeros(s, dtype=np.int32)
        out_spread = np.zeros(s, dtype=np.float64)

        n_out = _replay_signals_jit(
            self.ticks['time_ms'],
            self.ticks['bid'],
            self.ticks['ask'],
            sig_m5_idx, sig_buy, sig_sell,
            sig_sl_dist, sig_tp_dist, sig_be_dist, sig_trail_dist,
            sig_date_ord, sig_year,
            m5_t_start, m5_t_end,
            np.int64(bracket_bars),
            np.int64(max_hold_m5),
            np.int64(max_tpd),
            float(daily_cap_pct),
            float(starting_balance),
            lot, contract,
            STOPS_LEVEL_PRICE, FREEZE_LEVEL_PRICE,
            out_signal_m5, out_dir, out_entry_price, out_exit_price,
            out_sl, out_tp, out_pnl_net, out_exit_code,
            out_entry_ms, out_exit_ms, out_year, out_spread,
        )

        trade_list = []
        year_stats = {}
        for k in range(n_out):
            d = int(out_dir[k])
            dir_str = 'long' if d == _DIR_LONG else ('short' if d == _DIR_SHORT else 'none')
            ex_code = int(out_exit_code[k])
            tr = TradeResult(
                signal_idx=int(out_signal_m5[k]),
                direction=dir_str,
                entry_price=float(out_entry_price[k]),
                exit_price=float(out_exit_price[k]),
                sl=float(out_sl[k]),
                tp=float(out_tp[k]),
                pnl_gross=float(out_pnl_net[k] + out_spread[k] * lot * contract),
                pnl_net=float(out_pnl_net[k]),
                exit_reason=EXIT_CODE_TO_STR.get(ex_code, 'time'),
                entry_time_ms=int(out_entry_ms[k]),
                exit_time_ms=int(out_exit_ms[k]),
                spread_at_fill=float(out_spread[k]),
            )
            trade_list.append(tr)
            yr = int(out_year[k])
            if yr > 0:
                ys = year_stats.setdefault(yr, {
                    'wins': 0, 'losses': 0, 'gross_win': 0.0,
                    'gross_loss': 0.0, 'pnl': 0.0, 'trades': 0,
                })
                ys['trades'] += 1
                ys['pnl'] += tr.pnl_net
                if tr.pnl_net > 0:
                    ys['wins'] += 1
                    ys['gross_win'] += tr.pnl_net
                elif tr.pnl_net < 0:
                    ys['losses'] += 1
                    ys['gross_loss'] += abs(tr.pnl_net)

        return self._compute_results(trade_list, year_stats)

    def _empty_result(self) -> dict:
        return {
            'trades': 0, 'pf': 0.0, 'pnl': 0.0, 'win_rate': 0.0,
            'gross_win': 0.0, 'gross_loss': 0.0,
            'wins': 0, 'losses': 0,
            'max_dd': 0.0,
            'exit_reasons': {},
            'trade_list': [],
            'per_year': {},
        }

    @staticmethod
    def _compute_results(trade_list: list, year_stats: dict) -> dict:
        n_trades = len(trade_list)
        if n_trades == 0:
            return {
                'trades': 0, 'pf': 0.0, 'pnl': 0.0, 'win_rate': 0.0,
                'gross_win': 0.0, 'gross_loss': 0.0,
                'wins': 0, 'losses': 0,
                'max_dd': 0.0,
                'exit_reasons': {},
                'trade_list': [],
                'per_year': {},
            }
        pnls = np.array([t.pnl_net for t in trade_list])
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
