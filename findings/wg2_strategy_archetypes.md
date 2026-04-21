# WG2 — Strategy Archetype Catalog for EuroBigBrain (EUR/USD M5)

**Author:** WG2 Strategy Archetype Specialist
**Date:** 2026-04-21
**Parent project:** GoldBigBrain (XAUUSD fade_long, PF=1.76 / 14% DD MT5-verified)
**Context:** GBB ported verbatim to EUR/USD wiped the account (PF=0.95, 99.91% DD, 2143 trades). XAU edge is intraday range-fade in gold microstructure and does NOT transfer. This catalog proposes 10 EUR/USD-native archetypes.

---

## TL;DR

- **Top 3 to test first:** (1) NY Reversal Fade, (2) Asian Range Breakout with false-break filter, (3) ML Direction Classifier with cross-asset features. These are the only archetypes with supportive 2024-2026 literature AND a plausible structural reason to persist post-costs.
- **Do NOT port mean-reversion directly:** EUR/USD is 100-1000x deeper than XAU and has a much stronger trend-follower institutional participation at European/NY hours. Fade-only logic will get steam-rolled.
- **Portfolio thesis:** 2-3 uncorrelated edges (fade + breakout + ML gate) outperform one strong edge for a $1k retail account because capital efficiency is gated by DD, not by peak PF. Target combined PF>=1.4 at DD<=15%.

---

## Archetype Catalog

Scoring is ex-ante prior (1-10) of producing a deployable edge for a $1k Vantage STP retail account AFTER costs (EUR/USD typical spread 0.5 pip + 0.1 pip slippage = ~0.7 pip round-trip cost).

### 1. Asian Range Breakout

- **Hypothesis:** Asian session (22:00-06:00 UTC) is low-volume/low-information on EUR/USD; London open (07:00-09:00 UTC) injects institutional order flow that resolves the overnight range directionally. BoE April 2025 FX survey confirms EUR/USD is 25% of London turnover ($1.01T/day), making this the cleanest structural liquidity shift in the pair.
- **Entry/Exit:**
  - At 07:00 UTC: record `asian_high`, `asian_low` over 22:00-06:59 UTC.
  - Place stop-buy at `asian_high + 2 pips`, stop-sell at `asian_low - 2 pips` (2-pip buffer filters wick-only breaks).
  - OCO, valid 07:00-10:00 UTC. Cancel at 10:00 if unfilled.
  - SL = opposite side of range (typically 15-30 pips). TP = 1.5R. Trail after 1R.
  - **False-break filter:** require breakout candle to close beyond level (not just wick).
- **Estimated edge:** 8-12 pips/trade net, 6-10 trades/month (filter kills most days). Net: ~60-90 pips/mo.
- **Failure modes:** (a) Low-volatility regimes (2017, 2023 H1) collapse range and inflate false-break rate; (b) US CPI / ECB days pre-empt the London open and shift the breakout to 12:30-14:00 — strategy misses it; (c) broker variable spread widens at 06:55-07:05 rollover.
- **Validation gate:** Permutation null on `asian_high/asian_low` level (shuffle days): real PF >= 1.5 * null median across 6 yrs. Per-year PF all > 1.0.
- **Score: 7/10.** Best structural story on EUR/USD. Most-documented retail strategy + BoE turnover data supports the liquidity thesis. Downside: also the most-crowded — expect edge compression.

### 2. NY Reversal Fade (2-4h after NY open)

- **Hypothesis:** Post-NY-open momentum (13:30-14:30 UTC) over-extends because retail + CTA flow piles on direction; between 15:00-17:00 UTC, mean-reverting institutional rebalance flow emerges (US pension/insurance FX rebalancing, pre-London-fix positioning). Krohn (2024, JoF) "Foreign Exchange Fixings and Returns around the Clock" documents this clock-based reversion effect.
- **Entry/Exit:**
  - Measure NY-open drift: `drift = close(14:30) - open(13:30)`.
  - If `|drift| > 1.5 * ATR(M5, 20)`, arm fade signal for 15:00-17:00 UTC window.
  - Entry: first M5 candle after 15:00 UTC that closes back inside the 13:30-14:30 range, counter to drift.
  - SL: beyond the 14:30 extreme. TP: VWAP anchored from 13:30. Hard time-stop at 17:00.
- **Estimated edge:** 6-10 pips/trade net, 8-12 trades/month.
- **Failure modes:** (a) Trend days (NFP, FOMC) will run through fade — mandatory news blackout 13:25-13:35 and full NFP day skip; (b) thin liquidity Fridays after 15:00 widens spread cost; (c) any structural shift in US pension FX timing (past observations: they moved from 16:00 to 15:55 in 2023).
- **Validation gate:** Bootstrap p<0.01 vs same-session random entries; per-year PF > 1.1 on 2020-2026; survives ±15min window shift.
- **Score: 8/10.** Has a REAL structural driver (fix-related flows) documented in 2024 academic literature, not just retail folklore. EUR/USD is liquid enough that fade-fills are realistic.

### 3. News-Drift Momentum (post-release ride)

- **Hypothesis:** High-impact USD/EUR releases (NFP, CPI, FOMC, ECB) produce an initial spike followed by a directional "drift" as dealers/HFTs lay off inventory and slower institutional flow aligns. This drift is 15-45 min long. 2025 forex.com commentary confirms consistent directional follow-through on beat/miss NFP releases.
- **Entry/Exit:**
  - At release T (13:30 UTC for NFP/CPI), wait 2 M5 bars (T+10min) to let initial spike resolve.
  - At T+10min: measure `release_drift = close(T+10) - close(T-5)`. If `|release_drift| > 2 * pre-release ATR(20)`, enter in direction of drift at market.
  - SL: 0.8 * release_drift magnitude. TP: 1.8R. Time-stop T+60min.
- **Estimated edge:** 15-25 pips/trade net, but only 3-5 trades/month (8-12 major events). Net ~60-100 pips/mo.
- **Failure modes:** (a) Spread blowout at release (2-5 pip spread for 30-60s — MUST delay entry); (b) "surprise" vs "in-line" prints often reverse within 20 min; (c) slippage on stop-market entries can be 3-8 pips.
- **Validation gate:** Separate per-event-type (NFP / CPI / FOMC / ECB / PMI) — one must hit PF>1.4 on >=20 events 2020-2026. If only aggregate hits, it's noise.
- **Score: 6/10.** Real phenomenon but capacity-constrained (few trades) and execution-hostile for $1k account. Better as a gate/filter than primary edge.

### 4. Friday Close Mean-Reversion

- **Hypothesis:** Friday 18:00-21:00 UTC exhibits position-squaring mean reversion as speculators flatten weekend risk. Historical retail folklore; no strong 2024-2026 academic support.
- **Entry/Exit:** If Friday 12:00-18:00 UTC drift > 1.2 * 30-day Friday ATR, fade at 18:00 UTC toward VWAP of Friday. SL = day high/low, TP = VWAP or 20:30 UTC time-stop.
- **Estimated edge:** 5-8 pips/trade, 2-4 trades/month. Net ~15-30 pips/mo.
- **Failure modes:** (a) Thin liquidity worsens slippage more than the edge is worth; (b) weekly signal count is too small for proper validation (52 opportunities/yr gross, 10-20 net); (c) any Friday with overnight geopolitical risk reverses the thesis.
- **Validation gate:** Per-year PF>1.1 across 2020-2026 (six independent years). Given ~15 signals/yr, any false positives kill significance.
- **Score: 3/10.** Thin signal, weak structural story, brutal SNR. SKIP unless used as trivial ensemble member.

### 5. Intraday VWAP-Anchored Mean Reversion

- **Hypothesis:** During ranging sessions (Asian, mid-London lull 10:30-12:00 UTC), EUR/USD reverts to anchored-VWAP ± 2-sigma bands. VWAP is institutionally-meaningful as execution benchmark. Chartswatcher 2025 and Mastery Trader Academy 2025 articles document the setup for liquid pairs.
- **Entry/Exit:**
  - Anchor VWAP at 07:00 UTC (London open) or 13:30 UTC (NY open).
  - Compute rolling 20-period std-dev of `price - vwap` (VWAP bands).
  - Fade touches of ±2-sigma band during defined "ranging" windows (ADX(14) < 20 filter).
  - SL: ±2.5-sigma. TP: VWAP. Time-stop: 90 minutes.
- **Estimated edge:** 4-7 pips/trade net, 15-20 trades/month.
- **Failure modes:** (a) Regime shift: trending sessions eat fades — ADX filter is mandatory but lagging; (b) band breadth widens exactly when edge is weakest (post-news); (c) XAU VWAP-fade ported here will fail for same reason GBB failed — EUR/USD range structure is different.
- **Validation gate:** Conditional PF must hold across regime classes (low-ADX vs high-ADX) on 2020-2026; strict ADX filter must improve PF by >=20%, else it's coincident noise.
- **Score: 5/10.** Plausible but WG is wary given this is exactly the failure archetype of GBB on EUR/USD. The ADX regime-gate is essential and historically unreliable at M5.

### 6. Round-Number Stop Hunt

- **Hypothesis:** Stop clusters at x.x500 / x.x000 levels produce a predictable "sweep-and-reverse" pattern: price runs through, triggers stops, then reverses as the liquidity pool is exhausted. Market microstructure literature (Osler 2003 baseline, ongoing retail observation) supports clustering; 2024-2026 specific EUR/USD evidence is thin.
- **Entry/Exit:**
  - Define levels: nearest 50-pip round number (x.x500 / x.x000).
  - Signal: price pierces level by >=3 pips and closes back inside within 2 M5 candles.
  - Entry: reversal candle close, direction of reclaim.
  - SL: beyond the wick extreme + 2 pips. TP: next 50-pip level or 1.5R.
- **Estimated edge:** 10-15 pips/trade net, 4-8 trades/month.
- **Failure modes:** (a) True breakouts look identical to sweeps until ~30 min later (survivorship bias in pattern recognition); (b) round levels drift: 1.1000 was significant in 2020, is noise in 2026 at 1.22 regime; (c) broker-specific spread widening near psychological levels.
- **Validation gate:** Bootstrap against random non-round levels at matched volatility: real PF >= 1.4x noise median. Per-year PF > 1.0 on 2020-2026.
- **Score: 5/10.** Folklore-strong but empirically shaky. Worth testing as ensemble filter, not primary.

### 7. Cross-Asset Divergence (DXY vs EUR/USD)

- **Hypothesis:** DXY is 57.6% EUR/USD by construction, so a DXY move not matched by EUR/USD within minutes signals an imminent catch-up trade. MDPI 2025 "Regime-Aware LightGBM" paper explicitly shows cross-asset features contribute most predictive value in FX direction prediction.
- **Entry/Exit:**
  - Compute `implied_eurusd = f(DXY, other_components)` or simpler: `dxy_return(5min)` vs `eurusd_return(5min)`.
  - If `corr(5min rolling, 60 bars) < -0.85` (normal) but current 5-min `eurusd_return * dxy_return > -0.3 * sigma`, EUR/USD is lagging — enter catch-up direction.
  - SL: 10 pips. TP: convergence to historical regression (usually 5-15 pips, 10-40min).
- **Estimated edge:** 6-10 pips/trade, 10-20 trades/month.
- **Failure modes:** (a) Data-feed latency: DXY ticks vary by broker, can create phantom signals; (b) regime breaks: JPY or CHF moves can swing DXY without EUR/USD implication; (c) MT5 doesn't natively quote DXY — need proxy via index or currency basket.
- **Validation gate:** Must beat single-asset EUR/USD-only baseline by >=15% PF to justify added data infrastructure. Cross-validated across DXY proxies.
- **Score: 7/10.** Strong structural logic (arbitrage-like), 2025 literature support, genuinely different info source from pure-price strategies — lowest correlation to archetypes 1-6. Infrastructure cost is the concern.

### 8. Volatility-Regime Breakout (compression -> expansion)

- **Hypothesis:** Bollinger-band squeeze / low-ATR-percentile periods precede explosive moves. Classic pattern but academic evidence is mixed; retail-documented.
- **Entry/Exit:**
  - Compute ATR(M5, 14) percentile over trailing 500 bars. When percentile < 20 for >=20 bars, arm signal.
  - Entry: first M5 candle closing outside prior 20-bar range, in direction of break.
  - SL: opposite side of compression range. TP: 2R or 3x compression-width, whichever first.
- **Estimated edge:** 8-15 pips/trade, 4-8 trades/month.
- **Failure modes:** (a) Compression = news vacuum: break is often news-driven and unrideable by retail stop-entry; (b) false breaks: most compressions break, reverse, break other way; (c) time-of-day interacts strongly with ATR percentile and ADX.
- **Validation gate:** Must outperform "always-in breakout" baseline (no compression filter) by >=25% PF. Per-year stable.
- **Score: 5/10.** Decent hypothesis but execution-hostile. Useful as gate input.

### 9. Day-of-Week / Time-of-Day Seasonality

- **Hypothesis:** Intraday/intraweek seasonality exists in FX (Monday gap, Tuesday reversal, Wednesday consolidation, etc.) — weakly supported literature, strong within-sample fitting risk.
- **Entry/Exit:** Fit a calendar-hour × day-of-week matrix from training data; enter in direction of historical mean return whenever current hour-slot has `|mean_return| > 2 * se`.
- **Estimated edge:** Unknown — unclear separation from noise without multi-testing correction. Claimed 3-8 pips/trade.
- **Failure modes:** (a) Multiple-testing nightmare: 7 days × 24 hours = 168 buckets, expect 8 false positives at p<0.05 just by chance; (b) within-sample seasonality rarely survives OOS (GBB cfg74 hour-slot edge survived, but that's XAU and rare); (c) regime-dependent — DST / ECB meeting schedule changes shift timing.
- **Validation gate:** Bonferroni-corrected p<0.001 per slot; walk-forward 3yr/6mo must hold across 6+ folds.
- **Score: 3/10.** Pattern likely exists but separating from noise at $1k account scale is near-impossible. Use as prior for other archetypes, not standalone.

### 10. ML Direction Classifier (LightGBM tabular)

- **Hypothesis:** Tabular ML on engineered features (price returns, volatility, session dummies, cross-asset, calendar) can extract a weak but persistent direction prior. MDPI 2025 regime-aware LightGBM paper achieves useful edge with HMM-gated features; GoldBigBrain's own ML gate (AUC=0.72 across 2020-2026) validated the approach on XAU.
- **Entry/Exit:**
  - Features: (a) returns at 5m/15m/60m/4h; (b) ATR percentile; (c) session one-hots; (d) day-of-week; (e) distance-to-round-number; (f) DXY 5m correlation residual; (g) H1 RSI; (h) hour-of-day.
  - Label: sign of return over next 30 min, direction only.
  - Model: LightGBM, 200 trees, depth 6, L1/L2 regularized. Walk-forward train 3yr / test 6mo.
  - Trade rule: enter in predicted direction when `p > 0.58` (tunable), stop-entry at market. SL=12 pips, TP=20 pips, time-stop 60min.
- **Estimated edge:** 4-8 pips/trade net, 20-40 trades/month.
- **Failure modes:** (a) Lookahead bias in feature construction (session VWAP, ATR boundaries) — mandatory strict audit; (b) AUC=0.55 on EUR/USD is realistic ceiling, edge after costs is razor-thin; (c) model drift — features lose predictive power faster on FX than on XAU (6-12 mo half-life).
- **Validation gate:** Walk-forward 6/6 PASS required (same standard as GBB). OOS AUC >= 0.56. Deflated Sharpe >= 1.0.
- **Score: 7/10.** Highest-upside archetype if executed rigorously. GBB journey proved the framework works; EUR/USD is just a re-fit. Primary risk is repeating the XAU-to-EUR porting mistake (overfit features).

---

## Synthesis

### TOP 5 to test first (ranked)

1. **NY Reversal Fade (Score 8)** — strongest structural driver, academic 2024 support (Krohn JoF), liquid execution window, clean clock-based validation.
2. **ML Direction Classifier (Score 7)** — reuse GBB framework, highest upside, orthogonal info to session-logic strategies.
3. **Asian Range Breakout (Score 7)** — cleanest retail-accessible structural edge, most-documented. Expect crowded but viable.
4. **Cross-Asset Divergence (Score 7)** — lowest correlation to 1-3, strongest 2025 literature support for cross-asset as top feature class, real arbitrage-style logic.
5. **News-Drift Momentum (Score 6)** — real but capacity-constrained; best used as gate/filter on other strategies, deploy as its own edge only if 1-4 are dry.

### Bottom 3 to SKIP

1. **Friday Close Mean-Reversion (3)** — too few signals for validation, weak story.
2. **Day-of-Week Seasonality (3)** — multiple-testing swamps signal at retail scale.
3. **VWAP Mean Reversion (5)** — this is the GBB failure archetype ported; unless we find structural reason EUR/USD is different from XAU mean-reversion failure, it repeats the 99.91% DD wipe.

### Combined Portfolio Thesis

**Yes — 2-3 uncorrelated edges BEAT one strong edge for a $1k retail account.**

Rationale:
- For $1k account, max acceptable DD is ~15-20% ($150-200) before psychological/survival breakdown.
- A single PF=1.7 strategy (GBB caliber) typically carries 12-18% DD in OOS. Tail adds 5-10%.
- Two uncorrelated PF=1.35 edges combined approximate a joint PF=1.5 at DD=8-12% — strictly dominating the single-strategy profile on risk-adjusted terms (precedent: london-beasts "On a Leash" PF=1.76 / 14% DD BEATS "Unchained" PF=2.1 / 35% DD on risk-adj).
- Correlation matrix target: NY Reversal Fade vs Asian Breakout should be <0.3 (different sessions, opposite logic); ML vs either should be <0.4 (different signal sources).

**Proposed portfolio:**
- **Leg A:** Asian Range Breakout (London open momentum) — 30% capital allocation.
- **Leg B:** NY Reversal Fade (mean-reversion) — 40% capital allocation.
- **Leg C:** ML gate as META-filter across A+B (veto trades below p=0.45) — acts as risk-off switch rather than independent entry.

Target combined metrics: PF>=1.4, DD<=12%, 20-30 trades/month.

**Hard rules carried over from GBB experience:**
- Null-permutation test MANDATORY for every leg before MT5 deployment (real >=5x null).
- Walk-forward 3yr/6mo minimum, 6/6 folds must PASS.
- Spread+slippage realistic sim before go-live.
- No leg declared "validated" on a single period — always multi-year + per-year breakdown.

---

**Sources:**

- [Foreign Exchange Fixings and Returns around the Clock - Krohn 2024 JoF](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13306)
- [Regime-Aware LightGBM for Forecasting — MDPI 2025](https://www.mdpi.com/2079-9292/15/6/1334)
- [BoE Semi-Annual FX Turnover Survey April 2025](https://www.bankofengland.co.uk/markets/london-foreign-exchange-joint-standing-committee/results-of-the-semi-annual-fx-turnover-survey-april-2025)
- [London Breakout Strategy for EUR/USD backtest — zForex](https://zforex.com/blog/technical-analysis/london-breakout-strategy/)
- [VWAP Strategies for 2025 — Chartswatcher](https://chartswatcher.com/pages/blog/6-powerful-vwap-trading-strategies-for-2025)
- [VWAP Reversion Guide 2025 — Mastery Trader Academy](https://masterytraderacademy.com/vwap-reversion-strategy-trading-guide/)
- [Wavelet Denoised ResNet + LightGBM for Forex Rate Prediction — arXiv 2102.04861](https://arxiv.org/pdf/2102.04861)
- [EUR/USD Weekly NFP Analysis 2025 — forex.com](https://www.forex.com/en-us/news-and-analysis/eurusd-outlook-euro-weakens-after-nfp-release/)
- [EUR/USD Strategy Guide 2026 — FXNX](https://fxnx.com/en/blog/eur-usd-strategy-master-2026-s-market)
- [Six Market Microstructure Research Papers 2024 — Global Trading](https://www.globaltrading.net/research-on-the-web-in-2024/)
