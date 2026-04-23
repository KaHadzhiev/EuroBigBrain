"""
mt5_sim_strategies.py — Signal generation for all 20 entry types (EBB port).

Ported verbatim from GoldBigBrain. Strategies are entry-types and are
instrument-agnostic — they operate on ATR-relative offsets, not absolute
price units, so no XAU→EUR rescaling is needed at this layer.

Output format: list of (m5_bar_idx, buy_level_or_None, sell_level_or_None)
The core sim engine handles position management, fills, and exits.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Individual strategy entry functions
# Each returns (buy_level, sell_level) or None to skip the bar.
# ---------------------------------------------------------------------------

def _atr_bracket(ctx, config, i, a):
    bo = config.get('bracket_offset', 0.3)
    buy = ctx['c_v'][i] + bo * a
    sell = ctx['c_v'][i] - bo * a
    return buy, sell


def _null_bracket(ctx, config, i, a):
    bo = config.get('bracket_offset', 0.3)
    buy = ctx['c_v'][i] + bo * a
    sell = ctx['c_v'][i] - bo * a
    return buy, sell


def _asian_range(ctx, config, i, a):
    bar_date = ctx['dates'][i]
    ad = ctx['asian_daily']
    if bar_date not in ad.index:
        return None
    ar = ad.loc[bar_date]
    ar_range = ar['asian_high'] - ar['asian_low']
    max_ar = config.get('max_asian_atr', 10.0)
    if ar_range <= 0 or ar_range > max_ar * a:
        return None
    return ar['asian_high'], ar['asian_low']


def _momentum_long(ctx, config, i, a):
    if ctx['ret5'][i] <= 0:
        return None
    bo = config.get('bracket_offset', 0.3)
    return ctx['c_v'][i] + bo * a, None


def _momentum_short(ctx, config, i, a):
    if ctx['ret5'][i] >= 0:
        return None
    bo = config.get('bracket_offset', 0.3)
    return None, ctx['c_v'][i] - bo * a


def _fade_long(ctx, config, i, a):
    if ctx['rsi14'][i] >= 35:
        return None
    bo = config.get('bracket_offset', 0.3)
    return ctx['c_v'][i] + bo * a, None


def _fade_short(ctx, config, i, a):
    if ctx['rsi14'][i] <= 65:
        return None
    bo = config.get('bracket_offset', 0.3)
    return None, ctx['c_v'][i] - bo * a


def _ema_cross_long(ctx, config, i, a):
    if ctx['ema8'][i] <= ctx['ema21'][i]:
        return None
    bo = config.get('bracket_offset', 0.3)
    return ctx['c_v'][i] + bo * a, None


def _ema_cross_short(ctx, config, i, a):
    if ctx['ema8'][i] >= ctx['ema21'][i]:
        return None
    bo = config.get('bracket_offset', 0.3)
    return None, ctx['c_v'][i] - bo * a


def _breakout_range(ctx, config, i, a):
    lb = config.get('lookback', 12)
    if lb == 12:
        buy = ctx['roll12_high'][i]
        sell = ctx['roll12_low'][i]
    elif lb == 24:
        buy = ctx['roll24_high'][i]
        sell = ctx['roll24_low'][i]
    else:
        buy = ctx['roll48_high'][i]
        sell = ctx['roll48_low'][i]
    if np.isnan(buy) or np.isnan(sell):
        return None
    return buy, sell


def _vol_spike_bracket(ctx, config, i, a):
    vm = config.get('vol_mult', 2.0)
    vol_ma = ctx['vol_ma20'][i]
    if np.isnan(vol_ma) or ctx['vol_v'][i] < vm * vol_ma:
        return None
    bo = config.get('bracket_offset', 0.3)
    buy = ctx['c_v'][i] + bo * a
    sell = ctx['c_v'][i] - bo * a
    return buy, sell


def _rsi_long(ctx, config, i, a):
    if ctx['rsi14'][i] <= 55:
        return None
    bo = config.get('bracket_offset', 0.3)
    return ctx['c_v'][i] + bo * a, None


def _rsi_short(ctx, config, i, a):
    if ctx['rsi14'][i] >= 45:
        return None
    bo = config.get('bracket_offset', 0.3)
    return None, ctx['c_v'][i] - bo * a


def _ema_fade_long(ctx, config, i, a):
    if ctx['ema8'][i] >= ctx['ema21'][i]:
        return None
    bo = config.get('bracket_offset', 0.3)
    return ctx['c_v'][i] + bo * a, None


def _ema_fade_short(ctx, config, i, a):
    if ctx['ema8'][i] <= ctx['ema21'][i]:
        return None
    bo = config.get('bracket_offset', 0.3)
    return None, ctx['c_v'][i] - bo * a


def _vol_quiet_bracket(ctx, config, i, a):
    vol_ma = ctx['vol_ma20'][i]
    if np.isnan(vol_ma) or ctx['vol_v'][i] >= 0.5 * vol_ma:
        return None
    bo = config.get('bracket_offset', 0.3)
    buy = ctx['c_v'][i] + bo * a
    sell = ctx['c_v'][i] - bo * a
    return buy, sell


def _momentum_bracket(ctx, config, i, a):
    if ctx['ret5'][i] <= 0:
        return None
    bo = config.get('bracket_offset', 0.3)
    buy = ctx['c_v'][i] + bo * a
    sell = ctx['c_v'][i] - bo * a
    return buy, sell


def _ema_trend_bracket(ctx, config, i, a):
    if ctx['ema8'][i] <= ctx['ema21'][i]:
        return None
    bo = config.get('bracket_offset', 0.3)
    buy = ctx['c_v'][i] + bo * a
    sell = ctx['c_v'][i] - bo * a
    return buy, sell


def _high_vol_long(ctx, config, i, a):
    vm = config.get('vol_mult', 2.0)
    vol_ma = ctx['vol_ma20'][i]
    if np.isnan(vol_ma) or ctx['vol_v'][i] < vm * vol_ma:
        return None
    if ctx['ret5'][i] <= 0:
        return None
    bo = config.get('bracket_offset', 0.3)
    return ctx['c_v'][i] + bo * a, None


def _high_vol_short(ctx, config, i, a):
    vm = config.get('vol_mult', 2.0)
    vol_ma = ctx['vol_ma20'][i]
    if np.isnan(vol_ma) or ctx['vol_v'][i] < vm * vol_ma:
        return None
    if ctx['ret5'][i] >= 0:
        return None
    bo = config.get('bracket_offset', 0.3)
    return None, ctx['c_v'][i] - bo * a


# --- NEW (2026-04-23) ---

def _cci_fade_long(ctx, config, i, a):
    """CCI < -200 (oversold) → enter long bracket above current close."""
    cci = ctx.get('cci20')
    if cci is None or not np.isfinite(cci[i]) or cci[i] > -200:
        return None
    bo = config.get('bracket_offset', 0.3)
    return ctx['c_v'][i] + bo * a, None


def _cci_fade_short(ctx, config, i, a):
    """CCI > +200 (overbought) → enter short bracket below current close."""
    cci = ctx.get('cci20')
    if cci is None or not np.isfinite(cci[i]) or cci[i] < 200:
        return None
    bo = config.get('bracket_offset', 0.3)
    return None, ctx['c_v'][i] - bo * a


def _bb_squeeze_long(ctx, config, i, a):
    """Bollinger squeeze: BB width < 0.7×BB-width-MA AND price above upper band → long break."""
    bw = ctx.get('bb_width'); bwm = ctx.get('bb_width_ma'); bbu = ctx.get('bb_upper')
    if bw is None or bwm is None or bbu is None:
        return None
    if not (np.isfinite(bw[i]) and np.isfinite(bwm[i]) and np.isfinite(bbu[i])):
        return None
    if bw[i] >= 0.7 * bwm[i]:
        return None
    if ctx['c_v'][i] <= bbu[i]:
        return None
    bo = config.get('bracket_offset', 0.3)
    return ctx['c_v'][i] + bo * a, None


def _bb_squeeze_short(ctx, config, i, a):
    bw = ctx.get('bb_width'); bwm = ctx.get('bb_width_ma'); bbl = ctx.get('bb_lower')
    if bw is None or bwm is None or bbl is None:
        return None
    if not (np.isfinite(bw[i]) and np.isfinite(bwm[i]) and np.isfinite(bbl[i])):
        return None
    if bw[i] >= 0.7 * bwm[i]:
        return None
    if ctx['c_v'][i] >= bbl[i]:
        return None
    bo = config.get('bracket_offset', 0.3)
    return None, ctx['c_v'][i] - bo * a


def _inside_inside_long(ctx, config, i, a):
    """Two consecutive inside bars → bracket entry above bar[t-2].high (long side)."""
    if i < 3:
        return None
    h = ctx['h_v']; l = ctx['lo_v']
    # bar[i-1] inside bar[i-2]
    if not (h[i-1] <= h[i-2] and l[i-1] >= l[i-2]):
        return None
    # bar[i] inside bar[i-1]
    if not (h[i] <= h[i-1] and l[i] >= l[i-1]):
        return None
    # entry = bar[i-2] high + 1 pip
    bo = config.get('bracket_offset', 0.0)
    entry = h[i-2] + bo * a
    return entry, None


def _inside_inside_short(ctx, config, i, a):
    if i < 3:
        return None
    h = ctx['h_v']; l = ctx['lo_v']
    if not (h[i-1] <= h[i-2] and l[i-1] >= l[i-2]):
        return None
    if not (h[i] <= h[i-1] and l[i] >= l[i-1]):
        return None
    bo = config.get('bracket_offset', 0.0)
    entry = l[i-2] - bo * a
    return None, entry


# Aliases — atr_breakout maps to atr_bracket (per EBB Wave-3 naming).
# Both names are accepted by the dispatch table.
_atr_breakout = _atr_bracket


_STRATEGY_FN = {
    'cci_fade_long':       _cci_fade_long,
    'cci_fade_short':      _cci_fade_short,
    'bb_squeeze_long':     _bb_squeeze_long,
    'bb_squeeze_short':    _bb_squeeze_short,
    'inside_inside_long':  _inside_inside_long,
    'inside_inside_short': _inside_inside_short,
    'atr_bracket':       _atr_bracket,
    'atr_breakout':      _atr_breakout,
    'asian_range':       _asian_range,
    'momentum_long':     _momentum_long,
    'momentum_short':    _momentum_short,
    'fade_long':         _fade_long,
    'fade_short':        _fade_short,
    'ema_cross_long':    _ema_cross_long,
    'ema_cross_short':   _ema_cross_short,
    'breakout_range':    _breakout_range,
    'vol_spike_bracket': _vol_spike_bracket,
    'null_bracket':      _null_bracket,
    'rsi_long':          _rsi_long,
    'rsi_short':         _rsi_short,
    'ema_fade_long':     _ema_fade_long,
    'ema_fade_short':    _ema_fade_short,
    'vol_quiet_bracket': _vol_quiet_bracket,
    'momentum_bracket':  _momentum_bracket,
    'ema_trend_bracket': _ema_trend_bracket,
    'high_vol_long':     _high_vol_long,
    'high_vol_short':    _high_vol_short,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_signals(ctx, config, test_indices, probs):
    """
    Generate entry signals for a given strategy configuration.

    See GBB header for full param docs — semantics unchanged here.
    """
    entry_type = config['entry_type']
    vt = config['vt']
    sess_start = config.get('sess_start', 7)
    sess_end = config.get('sess_end', 20)

    strategy_fn = _STRATEGY_FN.get(entry_type)
    if strategy_fn is None:
        raise ValueError(f"Unknown entry_type: {entry_type}")

    atr14 = ctx['atr14']
    hours = ctx['hours']
    dates = ctx['dates']
    h_v = ctx['h_v']
    lo_v = ctx['lo_v']
    signals = []

    # Per-day cross guard — mirrors GBB_AsianRange.mq5:192. Without this,
    # asian_range fires the same level on every bar 7am-20pm.
    current_date = None
    buy_crossed_today = False
    sell_crossed_today = False

    for ii, i in enumerate(test_indices):
        bar_date = dates[i]
        if bar_date != current_date:
            current_date = bar_date
            buy_crossed_today = False
            sell_crossed_today = False

        if probs[ii] < vt:
            continue

        h = hours[i]
        if h < sess_start or h >= sess_end:
            continue

        a = atr14[i]
        if np.isnan(a) or a <= 0:
            continue

        result = strategy_fn(ctx, config, i, a)
        if result is None:
            continue

        buy_level, sell_level = result

        if buy_level is not None and (buy_crossed_today or h_v[i] >= buy_level):
            buy_level = None
        if sell_level is not None and (sell_crossed_today or lo_v[i] <= sell_level):
            sell_level = None
        if buy_level is None and sell_level is None:
            continue

        signals.append((i, buy_level, sell_level))

        if buy_level is not None:
            buy_crossed_today = True
        if sell_level is not None:
            sell_crossed_today = True

    return signals
