# EUR/USD M5 Session Windows — Recommendations & Rationale

**Project:** EuroBigBrain
**Date:** 2026-04-21
**Author:** EBB Research (WG3 session-specialist)
**Scope:** Concrete broker-time session windows for EUR/USD M5 fade vs breakout strategies, with per-window ATR, spread, and suitability notes. Based on BIS 2022 Triennial + BoE April 2025 FX Turnover Survey, WG1 microstructure brief, WG2 archetype catalog, and retail/academic sources cited inline.
**Broker assumption:** Vantage Standard STP. Server time on Vantage is **GMT+2 in winter / GMT+3 in DST** (MT5 "broker time"). Everything here is in **broker time (CET-equivalent)** unless marked GMT. Mapping: GMT 08:00 = broker 10:00 winter / 11:00 DST. Configs should key off the **GMT hour**, then translate at load.

---

## TL;DR — Headline Recommendations

- **Top-2 windows for FADE-style strategies:**
  1. **NY Overlap Fade Window, 14:00-16:00 GMT (broker 16:00-18:00 winter / 17:00-19:00 DST)** — Krohn 2024 fix-flow reversion window. Tight spreads, deep liquidity, clock-anchored institutional rebalance.
  2. **Late NY, 18:00-20:00 GMT (broker 20:00-22:00 winter / 21:00-23:00 DST)** — position-squaring into 17:00 NY close; mean reversion structurally supported; costlier spreads limit size.
- **Top window for BREAKOUT-style strategies:** **London open, 07:00-09:00 GMT (broker 09:00-11:00 winter / 10:00-12:00 DST)** — highest realized-vol burst of the day; Asian-range expansion candidate.
- **Do NOT port "13-20 broker" from GoldBigBrain.** On gold that window was NY-session mean-reversion on a thin, CTA-dominated tape. On EUR/USD the same clock bucket hits the **London/NY overlap**, which is the most **trend-friendly, HFT-priced** window of the whole day — fading it cost us PF=0.95 and a 99.91% DD in the M113 port.
- **Hard dead zone:** 21:00-06:00 GMT. 1.6-3.0 pip spreads, sub-2 pip ATR per M5 bar, thin order book. Skip unless running a dedicated overnight-lull archetype with NFP-level edge size.

---

## 1. Why This Document Exists

The parent project (GoldBigBrain / London Beasts) found a profitable session bucket at **broker hours 13-20 for XAUUSD**. When those literal hour params were ported to EUR/USD (the M113 attempt, 2026-04-21), the result was catastrophic: PF 0.95, DD 99.91%, 2143 trades. Postmortem in `project_m113_xauusd_overfit.md` attributes the wipe to three independent reasons — but the session-window mismatch is the first and largest.

Reasons the same clock bucket means opposite things on the two instruments:

- **Gold's NY session is thin and CTA-dominated** (spot + futures ≈ $200B/day). Retail fade against CTA trend-chasers has historically had edge 13-20 broker.
- **EUR/USD's same clock bucket is the London/NY overlap** — $1.2T/day total pair, peak tick density, HFT market-makers saturated. Fading at these hours means fighting rate-differential flow and real institutional hedging.
- **Gold spread stays ~20 pts 13-20; EUR/USD spread compresses to 0.6-0.9 pip** in the overlap but **widens again after 16:00 GMT as London shuts**. The spread curve shape is different.

So the EBB session windows must be **derived from EUR/USD microstructure**, not ported. Below are the five intraday regimes, broker-time and GMT, with the operational data needed to pick a strategy.

---

## 2. Intraday Regime Map

All ATR figures are **M5 ATR(14)** in pips, typical 2024-2026 non-news observations. Spread figures are **Vantage Standard STP**, quoted pips (round-trip ≈ 2×). Ranges reflect typical variation, not extremes. Sources: Trade That Swing 2024 daily-ATR decomposition, MarketMilk 2026 hourly volatility table, Take-profit.org hourly breakdown, BoE April 2025 turnover survey, WG1 microstructure brief.

| Window | GMT | Broker (winter) | Broker (DST) | M5 ATR (pips) | Spread (pips) | Tick density | Regime character |
|---|---|---|---|---|---|---|---|
| **Asian/late-NY dead zone** | 20:00-06:00 | 22:00-08:00 | 23:00-07:00 | 1.5-3.5 | 1.6-3.0 | 1-3/sec | Range-bound, position-squaring, illiquid |
| **Frankfurt burst** | 06:00-08:00 | 08:00-10:00 | 07:00-09:00 | 4-7 | 1.0-1.4 | 5-12/sec | First Euro-time real flow, range-expansion priming |
| **London momentum** | 08:00-12:00 | 10:00-14:00 | 09:00-13:00 | 6-12 | 0.8-1.1 | 10-25/sec | Peak daily vol burst, directional |
| **London/NY overlap** | 12:00-16:00 | 14:00-18:00 | 13:00-17:00 | 8-14 | 0.6-0.9 | 15-35/sec | Deepest book, trend continuation + fix-flow reversion |
| **NY-only afternoon** | 16:00-20:00 | 18:00-22:00 | 17:00-21:00 | 4-8 | 0.9-1.4 | 5-12/sec | London closed, thinning, position squaring |

---

## 3. Window-by-Window Detail

### 3.1 Frankfurt Open Burst — GMT 06:00-08:00

- **Broker time:** 08:00-10:00 winter / 07:00-09:00 DST.
- **M5 ATR:** 4-7 pips. First real price action of the European day; Asian compression starts to release.
- **Spread:** 1.0-1.4 pips. Most brokers' rollover window ends 22:00-00:00 GMT, so by 06:00 GMT spreads have normalized but not yet compressed to London levels.
- **Flow character:** European corporates fixing overnight books, Frankfurt-based banks opening order flow. Still retail-light; institutional order-routing dominates.
- **Suitability for FADE:** **LOW.** There is no established "NY extreme" to fade against. Any early-session mean reversion is fading pure noise.
- **Suitability for BREAKOUT:** **MEDIUM.** Asian-range breakout logic can be armed here, but the real execution happens at London open. Frankfurt is a "warning shot" — use for volatility-regime classification (ATR-percentile compression detection), not entry.
- **Recommendation:** Skip entries. Use this window to compute Asian-range high/low and arm breakout triggers for the next window.

### 3.2 London Open Momentum — GMT 07:00-12:00 (core 07:00-09:00)

- **Broker time:** 09:00-14:00 winter / 09:00-14:00 DST (London open time shifts with DST).
- **M5 ATR:** 6-12 pips, peaking 08:00-09:00 GMT. TradeThatSwing 2024 and MarketMilk place 07:00-12:00 GMT as the single highest-volatility zone in a typical EUR/USD day (outside news).
- **Spread:** 0.8-1.1 pips. Depth improves rapidly; sub-1-pip is achievable for limit orders in the 08:00-11:00 GMT core.
- **Flow character:** BoE April 2025 survey reports EUR/USD is **25% of London turnover (~$1.01T/day of a $4T London book)**. Structural liquidity shift is real. First major expansion of the day; directional momentum dominates. Bollinger-squeeze releases here with high frequency.
- **Suitability for FADE:** **LOW-MEDIUM.** Fading the open burst is fighting institutional flow. Only tradable as **second-break / sweep-and-reverse** (WG1 archetype #1), not first-impulse fade.
- **Suitability for BREAKOUT:** **HIGH.** Asian-range breakout, compression-expansion, NY-pre-position momentum. Highest structural-edge window of the day for directional strategies.
- **Recommendation:** **Primary breakout window.** Arm Asian-range OCO stops here. Expect 6-10 trades/month on a properly filtered Asian-range breakout strategy (WG2 archetype #1).

### 3.3 London/NY Overlap — GMT 12:00-16:00

- **Broker time:** 14:00-18:00 winter / 13:00-17:00 DST.
- **M5 ATR:** 8-14 pips, peak of the day.
- **Spread:** 0.6-0.9 pips — **tightest spreads of the day**, both major books open simultaneously.
- **Flow character:** Deepest order book globally. US economic releases cluster at 12:30 and 14:00 GMT. Fix-related reversion flow (Krohn JoF 2024) concentrates 14:30-16:00 GMT as US pension/insurance rebalance ahead of London fix (16:00 GMT = 15:00 GMT London-fix clock, depending on product).
- **Suitability for FADE:** **HIGH — but only in the back half (14:00-16:00 GMT).** The first half (12:00-14:00 GMT) is news-contaminated and first-wave-momentum driven; fading it is against institutional flow. The back half is the **Krohn fix-reversion window** — this is the academically-supported structural fade.
- **Suitability for BREAKOUT:** **MEDIUM-HIGH.** NY-open momentum continuation (WG1 archetype #2) is the tradable breakout variant here. Requires a macro filter (US-DE 2Y yield spread direction) to separate signal from chop.
- **Recommendation:** **Primary FADE window = 14:00-16:00 GMT (broker 16:00-18:00 winter / 17:00-19:00 DST).** Primary breakout variant = 13:30-14:30 GMT NY-open continuation with yield-spread filter. Expect 8-12 fade trades/month, 4-8 breakout trades/month.

### 3.4 NY-Only Afternoon — GMT 16:00-20:00

- **Broker time:** 18:00-22:00 winter / 17:00-21:00 DST.
- **M5 ATR:** 4-8 pips, falling through the window.
- **Spread:** 0.9-1.4 pips. London desks close at 16:00-17:00 GMT; book depth halves.
- **Flow character:** NY-only trading; position-squaring into 17:00 NY close (EUR/USD's daily close reference for rollover). Retail contrarian flow is maximal here (Breedon et al. 2025) — retail fades the day's trend as it matures, institutional trend-followers steamroll them.
- **Suitability for FADE:** **MEDIUM.** Mean reversion is structurally present (position squaring), but spread cost eats 30-50% of edge on a M5 retail bot. Only tradable with tight cost budget and disciplined hour filter.
- **Suitability for BREAKOUT:** **LOW.** Momentum has exhausted; thin book means any push gets faded fast.
- **Recommendation:** Secondary fade window, 18:00-20:00 GMT only. Hard cutoff at 20:00 — after that, spread explodes.

### 3.5 Dead Zone (Asian + late-NY) — GMT 20:00-06:00

- **Broker time:** 22:00-08:00 winter / 23:00-07:00 DST.
- **M5 ATR:** 1.5-3.5 pips. On most bars, ATR is at or below round-trip cost.
- **Spread:** 1.6-3.0 pips, often 2.0+ during Tokyo lunch and 22:00-00:00 GMT rollover.
- **Flow character:** Asian session; EUR/USD is not a native Asian pair. Thin book, wide spreads, algorithmic liquidity-provision at low confidence. Prior-day range compression forms here, feeding the next London open.
- **Suitability for FADE:** **DO NOT TRADE.** ATR < cost. Mathematically impossible for a naive fade to be profitable.
- **Suitability for BREAKOUT:** **DO NOT TRADE.** Any break during Asian hours is noise or news-driven spike.
- **Recommendation:** **Hard blackout.** Use these bars only for feature engineering (Asian-range H/L, overnight drift, prior-day close reference). Any strategy entry in this window is a red-flag bug.

---

## 4. Cross-Window Summary

| Strategy archetype | Best window (GMT) | Window name | Expected trades/mo |
|---|---|---|---|
| Fade (Krohn fix-flow) | 14:00-16:00 | NY Overlap Back-Half | 8-12 |
| Fade (position-squaring) | 18:00-20:00 | NY-only Afternoon | 2-5 |
| Breakout (Asian range) | 07:00-09:00 | London Open | 6-10 |
| Breakout (NY momentum) | 13:30-14:30 | NY Open | 4-8 |
| News drift (post-release) | 13:40-15:00 | NY Overlap Front | 3-5 |
| Any | 20:00-06:00 | **BLACKOUT** | 0 |

---

## 5. Why "13-20 broker" Is Wrong For EUR/USD

Broker 13-20 in winter = GMT 11:00-18:00. That hits:

- GMT 11:00-12:00: London lunch lull (ATR trough, pre-NY).
- GMT 12:00-14:00: news burst + first NY wave (most-contaminated 2h of the day).
- GMT 14:00-16:00: Krohn fix-reversion window (real).
- GMT 16:00-17:00: London close.
- GMT 17:00-18:00: thin NY-only.

Only **2 of those 7 hours** have a real structural fade edge; the other 5 include news contamination, directional NY momentum, and post-close thinning. A session mask that treats all 7 equally is **5 hours of noise around 2 hours of signal** — which is exactly what we measured in the M113 port.

The correct EUR/USD analogue of "the profitable gold window" is **14:00-16:00 GMT only (broker 16:00-18:00 winter / 17:00-19:00 DST)**, not 13-20 broker.

---

## 6. DST Handling — Operational Note

Vantage server time is **GMT+2 in winter, GMT+3 in DST** (US DST dates, not EU). For roughly 3 weeks per year (mid-March to late March, early-November) US and EU DST disagree, producing a transient **GMT+2 with US-DST-active** broker clock. This will shift the GMT-anchored windows by ±1 hour on the broker clock for those 3 weeks.

**Implementation rule:** always key session gates off the **UTC/GMT hour** of the candle timestamp, never the broker hour. Translate at config-load time. The `session_mask` function should accept a `gmt_hour` argument, not a `broker_hour`. Any code that reads raw MT5 time and compares to `hour in (13,14,...20)` is a DST-shift bug waiting to fire during the misalignment window.

---

## 7. What Still Has To Be Measured (Gaps)

This document is the **prior**, not the evidence. Every window must be validated on **2020-2026 EUR/USD M5 MT5 every-tick** with per-year PF before any live candidate is declared. Specifically:

- **Frankfurt-burst real ATR** — the 4-7 pip figure is retail consensus; needs MT5 measurement.
- **NY-only afternoon edge decay** — the 18:00-20:00 fade edge is plausibly 2025-era artifact; check 2020-2022 stability.
- **Krohn fix-flow window drift** — academic paper is 2024 data; if pension desks shifted timing (they moved 16:00→15:55 in 2023), the window boundaries may have moved again.
- **Spread curve** at 16:00-17:00 GMT transition — the London-close spread widening is broker-specific; measure on Vantage tick data directly.
- **News-contamination rate by window** — quantify what % of each window's bars are within 30 min of a Tier-1 EUR or USD release. See `news_blackout.py` for the filter; the measurement itself (session × news-density heatmap) is a separate report.

Targets for the measurement phase: per-year PF table for each (archetype × window) tuple, null-tested per WG3 Gate 3, multi-instrument cross-check per C5.

---

**Sources:**

- [EUR/USD Volatility Statistics (Trade That Swing 2024)](https://tradethatswing.com/analyzing-eur-usd-volatility-for-day-trading-purposes/)
- [EUR/USD Volatility Calculator (MarketMilk 2026)](https://marketmilk.babypips.com/symbols/EURUSD/volatility)
- [EUR/USD Volatility by Hour (Take-profit.org)](https://take-profit.org/en/volatility/forex/eur-usd/)
- [BoE FX Turnover Survey April 2025](https://www.bankofengland.co.uk/markets/london-foreign-exchange-joint-standing-committee/results-of-the-semi-annual-fx-turnover-survey-april-2025)
- [Krohn 2024 JoF "FX Fixings and Returns Around the Clock"](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13306)
- [Best Time to Trade EURUSD (ATFX)](https://www.atfx.com/en/analysis/trading-strategies/best-time-to-trade-eurusd)
- WG1 `findings/wg1_market_microstructure.md`
- WG2 `findings/wg2_strategy_archetypes.md`
- M113 postmortem `memory/project_m113_xauusd_overfit.md`
