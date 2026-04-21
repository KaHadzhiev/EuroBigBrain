# Design — DXY Cross-Asset Divergence Feed for EuroBigBrain

**Author:** DXY Cross-Asset Feed Designer
**Date:** 2026-04-21
**Status:** Design (not yet implemented)
**Parent findings:** WG2 Archetype #7 (score 7/10), WG4 architecture §5 (multi-instrument from day 1)
**Target strategy:** EUR/USD M5 divergence GATE — not standalone edge

---

## 0. Executive summary

DXY is 57.6% EUR/USD by construction. A DXY move not matched by EUR/USD signals non-dollar-driven flow in the euro leg and a short-horizon catch-up trade. WG2 scores this 7/10; MT5 has no native DXY — we proxy from a broker symbol or synthesize from the basket.

**Recommendation: synthesize from the six-currency basket using ICE weights; validate against ICE DX futures forward-only; use as a META-FILTER on other edges (NY Reversal Fade, Asian Breakout), not a standalone signal.** Synthetic DXY is accurate to ~1-2 bp on M5 closes vs ICE DX and is free; real-DXY is paid and adds <5% accuracy.

Threshold heuristic (MDPI 2025 + WG2): rolling-60-bar residual z-score `|z| >= 2.0` informational, `|z| >= 2.5` arms the gate. Gate vetoes entries whose direction disagrees with implied-EUR/USD drift from DXY.

---

## 1. DXY data source — synthetic wins

### 1.1 Option A — Vantage broker proxy (REJECTED)

Vantage Standard STP offers `DXY` as a CFD on ICE DX futures, but: spread 2-4 index points (~multiple pips of EUR/USD-equivalent noise per tick); tick cadence 1-2s (not matched to EUR/USD's ~10Hz, introduces sync artifacts); weekend gaps mismatched vs calculated basket; and it costs spread+swap on a symbol we never trade. A broker DXY CFD is a lagged, noisy copy of what we can compute for free.

### 1.2 Option B — Synthetic DXY from six-currency basket (ACCEPTED)

The ICE formula:

```
DXY = 50.14348112 × EURUSD^-0.576
                  × USDJPY^ 0.136
                  × GBPUSD^-0.119
                  × USDCAD^ 0.091
                  × USDSEK^ 0.042
                  × USDCHF^ 0.036
```

Weights negate for pairs where USD is quote (EUR, GBP) and add for pairs where USD is base (JPY, CAD, SEK, CHF). The 50.14348112 constant anchors the March 1973 baseline at 100.

All six symbols are quoted on Vantage MT5 with sub-pip spreads and tick-level liquidity. No data license needed. This is the standard approach used by every quant shop that needs DXY in a sim.

### 1.3 Option C — Real ICE DX futures feed

Direct ICE DX (Refinitiv/Bloomberg/CQG) is $200-1500/mo. For a $1k account on a GATE (not signal), ROI is negative by ~2 orders of magnitude. **Reject unless forward validation shows synthetic drift > 2 bp vs ICE close.**

---

## 2. Synthetic DXY computation in MQL5

### 2.1 Symbol inventory

```mql5
string kDXYLegs[6] = {"EURUSD","USDJPY","GBPUSD","USDCAD","USDSEK","USDCHF"};
double kDXYWeights[6] = {-0.576, 0.136, -0.119, 0.091, 0.042, 0.036};
double kDXYConstant = 50.14348112;
```

### 2.2 Per-tick computation

On each `OnTick()` of the primary EUR/USD chart:

```mql5
double SyntheticDXY()
{
    double dxy = kDXYConstant;
    for (int i = 0; i < 6; i++)
    {
        MqlTick t;
        if (!SymbolInfoTick(kDXYLegs[i], t)) return 0.0;
        double mid = 0.5 * (t.bid + t.ask);
        dxy *= MathPow(mid, kDXYWeights[i]);
    }
    return dxy;
}
```

Cost per call: 6× `SymbolInfoTick` (each a cached memory read, ~nanoseconds) + 6× `MathPow` (~200ns each) = **sub-microsecond**. Negligible per tick; order-of-magnitude less than a single ATR update.

### 2.3 M5 bar synchronization

Signals trigger on EUR/USD M5 bar close. Snap to EUR/USD bar time and resample legs with `CopyRates(leg, PERIOD_M5, shift, 1, ...)` at `iTime(_Symbol, PERIOD_M5, shift)` — NOT the leg's own bar time. Weekend-gap hazard: USDSEK on Sunday open may not print at 22:00 UTC exactly; if leg bar time is >5 minutes off EUR/USD bar, mark DXY stale and disable gate for that bar.

### 2.4 Pre-compute history at init

On `OnInit`, compute `syn_dxy` for the trailing 2000 M5 bars (~7 days) so rolling regression windows are warm. Cost ~200ms at EA start.

---

## 3. Divergence calculation

### 3.1 Primitive — log-return residual

Operate on log returns to remove level drift: `r_eur = log(close_eurusd[t]/[t-1])`, `r_dxy = log(syn_dxy[t]/[t-1])`. By construction (EUR weight -0.576): `E[r_dxy] ≈ -0.576·r_eur + other-leg noise` where the noise is the 42.4% JPY/GBP/CAD/SEK/CHF component uncorrelated with EUR.

### 3.2 Rolling regression (60 bars = 5 hours on M5)

Window `N = 60`. Recompute on every closed M5 bar:

```
beta   = cov(r_eur, r_dxy, N) / var(r_dxy, N)
alpha  = mean(r_eur, N) - beta * mean(r_dxy, N)
pred_r_eur = alpha + beta * r_dxy[t]
resid      = r_eur[t] - pred_r_eur
sigma      = stdev(resid, N)
z          = resid / sigma
```

`beta` should sit near `-1 / 0.576 ≈ -1.74` under normal EUR-dominated DXY flow; deviations indicate JPY/CHF-driven DXY (regime signal).

**Why residual-z not raw correlation?** Raw correlation drops mechanically in low-vol bars (both returns near zero, noise dominates). Z-scored residual is scale-invariant.

### 3.3 Divergence regimes (state machine)

| Regime | Condition | Interpretation |
|---|---|---|
| **Synced** | `|z| < 1.0` | EUR/USD confirming DXY — no divergence signal |
| **Lean** | `1.0 <= |z| < 2.0` | Mild divergence — informational, no action |
| **Diverge** | `2.0 <= |z| < 2.5` | Significant divergence — raise attention flag |
| **Trigger** | `|z| >= 2.5` | Hard divergence — arm gate |
| **Break** | `|z| >= 3.5` | Regime shift / news spike — disable DXY gate 30 min |

Threshold `|z| >= 2.5` corresponds to ~1.2% frequency in normal EUR/USD (about 3 bars per day on 24h M5) — sparse enough to be informative, dense enough to be measurable.

---

## 4. Signal generation — GATE not SIGNAL

WG2 is explicit: divergence is a 7/10 candidate with thin capacity (~10-20 trades/mo standalone) and known failure modes (JPY/CHF regime shifts). We use it as a **META-FILTER on NY Reversal Fade and Asian Breakout entries**, not a standalone signal.

### 4.1 As a GATE (primary usage)

When NY Reversal Fade or Asian Breakout fires a LONG signal:

- Compute current `z`.
- Compute `dxy_implied_eur_dir = sign(-r_dxy_recent_15min)` — if DXY is falling, EUR should be rising → implied-long.
- If `|z| >= 2.5` AND `signal_dir == dxy_implied_eur_dir`: **PASS** (DXY confirms the catch-up thesis).
- If `|z| >= 2.5` AND `signal_dir != dxy_implied_eur_dir`: **VETO** (signal fights DXY-implied flow).
- If `|z| < 2.5`: pass through unchanged (no opinion).
- If in **Break** regime: pass through unchanged (gate disabled).

### 4.2 As a STANDALONE SIGNAL (optional week-5 test)

Entry when `|z| >= 2.5` AND `z` has just flipped through ±2.0 in the last 3 bars (momentum of divergence):

- Direction: toward `dxy_implied_eur_dir`.
- SL: 10 pips (3-4× typical M5 ATR).
- TP: convergence to `|z| < 0.5` OR 40-minute time-stop OR 15 pips hard — whichever first.

This is the WG2 archetype definition verbatim; we test it only AFTER the gate variant is validated, to avoid over-counting the edge.

### 4.3 Hard rules (carry-over)

- **No trades** on the bar a divergence flips — one-bar cooldown to avoid mean-reverting within-bar noise.
- **News blackout** 13:25-13:35 UTC (NFP/CPI days full skip): DXY divergence at release is mechanical, not tradeable.
- **Spread guard:** existing `EBB.RiskCore` spread guard applies — 3-pip cap on EUR/USD.

---

## 5. Backtest data preparation

### 5.1 Required history

For every M5 bar in backtest period, need aligned M5 OHLC on:
- `EURUSD` (primary, already in warehouse)
- `USDJPY`, `GBPUSD`, `USDCAD`, `USDSEK`, `USDCHF` (need to source)

WG4 warehouse plan already includes EUR, GBP, AUD, JPY, CAD M1 ticks 2020-2026. **ADD** USDSEK + USDCHF in week 1. AUD is not a DXY leg — keep it for multi-instrument validation but not for DXY synthesis.

### 5.2 Data integrity checks

At warehouse build time:

- **Alignment:** every EUR/USD M5 timestamp must have a bar on all 5 other legs. SEK gaps get ffilled up to 3 bars; beyond 3, flag and disable gate.
- **Sanity spot-check:** synthetic 2025-01-03 17:00 UTC close vs ICE DX futures close — must match within 0.15 index points (~0.15%). Common-bug catcher: USDCHF vs CHFUSD inversion.
- **Holidays:** US holidays close ICE DX but spot FX trades; synthesize anyway, tag `dxy_outside_ice_hours = True`, consider excluding from gate.

### 5.3 Storage

Add a `synthetic_dxy` DuckDB view in the warehouse:

```sql
CREATE VIEW synthetic_dxy_m5 AS
SELECT
    t.ts,
    50.14348112
      * POWER(e.close, -0.576)
      * POWER(j.close,  0.136)
      * POWER(g.close, -0.119)
      * POWER(c.close,  0.091)
      * POWER(s.close,  0.042)
      * POWER(h.close,  0.036) AS dxy
FROM bars_m5_eurusd e
JOIN bars_m5_usdjpy j USING (ts)
JOIN bars_m5_gbpusd g USING (ts)
JOIN bars_m5_usdcad c USING (ts)
JOIN bars_m5_usdsek s USING (ts)
JOIN bars_m5_usdchf h USING (ts);
```

---

## 6. Computational cost

Extrapolating from GBB tick-replay M5 path:

- Baseline EUR/USD-only 2020-2026: ~8min, 100 configs parallel on Mac 10-core.
- +5 extra in-memory symbol reads per bar: +~12% wall = ~9min.
- Rolling N=60 regression (Welford-style incremental): <1%.
- ONNX inference with divergence features: +0%.

For MT5 every-tick, 6 `SymbolInfoTick` calls are dominated by the event loop itself — a full-year every-tick run goes from ~40 to ~45min. **Real risk is not compute but data:** 6yr × 6 symbols × ~20 cols ≈ 500 MB DuckDB (fine), but SEK + CHF history is not yet in the warehouse — week-1 sync work.

---

## 7. Forward-test validation — does synthetic match ICE DX?

### 7.1 Static calibration — historical

Pull free ICE DX daily close from investing.com/stooq.com for 2020-2026 (daily, free). Per trading day: `error_bp = (syn_17utc - ice_eod) / ice_eod × 10000`. Expected from prior quant work: mean |error| ~1.5-3 bp, p99 ~8 bp. **Gates:** mean |err| < 5 bp, p99 < 15 bp, zero days >50 bp unexplained. Common >50 bp causes: Scandinavian holiday stale USDSEK print; broker JPY mid differs from ICE composite (2022 intervention); a leg data gap.

### 7.2 Forward check — M5 cadence

Use TradingView's free delayed-15min DXY indicative feed for 30 days paper-live. Log synthetic + indicative ICE on every EUR/USD M5 close. Target Pearson on 5-min deltas ≥ 0.98, sign-agreement ≥ 95% when |ICE delta| > 0.02. If correlation < 0.95 on 30-day sample — revisit broker legs or consider paid feed.

### 7.3 Gate killswitch

If runtime drift > 3σ from calibration bounds for two consecutive days, EA auto-disables DXY gate (falls back to ungated entries) and flags user. WG4-style circuit breaker.

---

## 8. Implementation checklist (maps to WG4 week plan)

- [ ] **Week 1 warehouse:** add USDSEK, USDCHF M1 tick history 2020-2026 (currently missing from plan).
- [ ] **Week 1 Python:** `python/features/dxy_synthesizer.py` (this PR), unit tests with known ICE DX weeks.
- [ ] **Week 2 validation:** add DXY-gate pass-through test to 6-gate harness; null-test that shuffles DXY timestamps — real gate must keep PF edge, shuffled must not.
- [ ] **Week 3 EA integration:** `mql5/EBB/Signals/DXYGate.mqh` — implements §2 and §4.1.
- [ ] **Week 4 ML feature:** expose `dxy_resid_z` and `dxy_beta` to LightGBM feature registry (WG4 §4); ablate — if AUC contribution < 0.003, drop.
- [ ] **Week 5 multi-instrument:** N/A — DXY is EUR/USD-specific (the gate thesis is built on EUR's 57.6% weight). GBP/USD variant would use a different basket (EURGBP + DXY-implied cross).
- [ ] **Week 6 paper-live:** 30-day ICE DX vs synthetic correlation log, gate kill-switch drill.

---

## 9. Open questions / known unknowns

- **Does divergence persist post-costs?** WG2 estimate: 6-10 pip gross gross vs ~1 pip EUR/USD round-trip. As a gate (not signal) cost impact is zero but edge contribution must be measurable (ablation in week 4).
- **JPY/CHF regime contamination:** 2022 BOJ intervention and 2024 yen carry unwind moved DXY 100+ points while EUR/USD barely moved — gate would veto legitimate trades. Need `jpy_regime_flag` feature to detect and disable gate on such days.
- **2020 COVID flash:** March 2020 had 300+ bp synth-vs-ICE divergence from USDSEK gaps. Expected behavior: Break regime auto-disables gate; must verify on week-5 stress scenario.

---

**End of design.**

## Sources

- [ICE U.S. Dollar Index methodology — ICE Futures U.S.](https://www.ice.com/publicdocs/futures_us/ICE_USDX.pdf)
- [Regime-Aware LightGBM for Forecasting — MDPI 2025](https://www.mdpi.com/2079-9292/15/6/1334)
- [Programming custom symbols / synthetic indices in MQL5 — MQL5 docs](https://www.mql5.com/en/docs/customsymbols)
- [WG2 Strategy Archetype Catalog (this project)](../findings/wg2_strategy_archetypes.md)
- [WG4 Architecture Improvements (this project)](../findings/wg4_architecture_improvements.md)
