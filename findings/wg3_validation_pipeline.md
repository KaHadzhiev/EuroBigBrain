# WG3 — EuroBigBrain Validation Pipeline

**Author:** Anti-Overfit / Validation-Pipeline Architect
**Date:** 2026-04-21
**Scope:** EUR/USD M5 algorithmic strategy candidate gating, pre-deploy.

---

## TL;DR

- **11 sequential gates**, each a hard-fail: Gate 1 (in-sample sanity) → Gate 11 (live kill-switch). No candidate reaches MT5 live without passing every gate; results logged to `validation_ledger.parquet` with immutable run-ID.
- **Block-bootstrap is the only null test that matters for EURUSD M5.** cfg74 taught us session-shuffle nulls test the *wrong hypothesis* for reversion strategies — prescription: **match the null to the archetype** (reversion → direction-flip + block-bootstrap; trend → session-shuffle; ML → label-permutation).
- **P-hacking budget: 200 backtests total before Deflated Sharpe ratio adjustment makes "PF=1.4 in-sample" worthless.** Bailey/López de Prado DSR formula is implemented in `validation/dsr.py`; anything below DSR=0.95 threshold is rejected regardless of raw PF.
- **Multi-instrument gate is the M113 killer.** EuroBigBrain must run unchanged on GBPUSD and AUDUSD with PF ≥ 1.1 each, OR the strategy must have a documented economic reason it's EUR-specific (e.g., ECB release timing). "It only works on EUR" is a red flag, not a feature.
- **Build infrastructure FIRST, strategies SECOND.** The entire `validation/` package (dsr, bootstrap, walk-forward, cost-stress, regime-split, monitor) must exist and be unit-tested before the first candidate sweep. Target: 2 weeks of pure plumbing before any strategy code.

---

## 1. Gate 1 — In-Sample Sanity (cheap, fast, discriminating)

Bare minimum to waste more compute on. Apply to full history 2018-01-01 → 2025-06-30 (reserve final 6 months for Gate 10).

| Metric | Threshold | Rationale |
|---|---|---|
| Profit Factor | ≥ 1.30 | Below this is unusable after costs |
| Trade count | ≥ 500 | Statistics need N; M5 EURUSD easily yields this over 7yrs |
| Max DD | ≤ 15% of starting equity | Beyond this, position sizing breaks account in live |
| Avg trade duration | 15 min – 48h | Shorter = spread dominates; longer = M5 is wrong TF |
| Sharpe (daily returns) | ≥ 0.8 | Conservative — deflated version applied at Gate 2 |
| Win rate | not checked | Intentional: reversion vs trend have very different win rates |

**Fail here = reject immediately, no further compute.**

---

## 2. Gate 2 — Bootstrap p-value + Deflated Sharpe

**Method:** Stationary block bootstrap (Politis & Romano 1994) on per-trade PnL returns. EURUSD M5 autocorrelation decays to <0.05 by lag 20 bars in volatility-clustered returns. Use `arch.bootstrap.optimal_block_length` to compute per-strategy optimal block length; expected range 10–30 trades.

**Parameters:**
- `n_resamples = 10,000`
- `block_length = auto` via Politis/White 2004
- `null_hypothesis: mean(per_trade_return) = 0`
- Reject null if observed_mean_PnL > 99th percentile of resampled means (**p < 0.01**)

**Deflated Sharpe Ratio (Bailey/López de Prado 2014):**
Also compute DSR given N independent trials in the candidate sweep:

```
DSR = SR_observed × √(skew-kurt adjustment) − E[max SR | N, T] / σ(SR)
```

Require **DSR > 0.95** (≈ 95% confidence real edge after selection-bias correction). The `N` argument to DSR is the **number of distinct hypotheses tested across the whole project**, not just in the current sweep — we carry the counter in `validation_ledger.parquet`.

**Implement:** `validation/bootstrap.py`, `validation/dsr.py`.

---

## 3. Gate 3 — Archetype-Specific Null Test

The cfg74 lesson: **a session-shuffle null tests "does the hour-of-day matter" — irrelevant for a strategy whose edge is price reversion.** Null must target the mechanism actually claimed.

| Archetype | Primary null | Secondary null |
|---|---|---|
| Reversion (fade) | Direction-flip (multiply all signals × −1; PF must collapse) | Sign-shuffled returns |
| Breakout / momentum | Session-shuffle (rotate hour labels randomly) | Block-shuffle of features |
| ML meta-filter | Label permutation (shuffle y before training, retrain, re-backtest) | Bar-shifted features |
| Mean-revert band | Price-path bootstrap (IID resample M5 returns, rebuild series) | Direction-flip |

**Pass criteria:** observed PF ≥ 1.5× the 95th-percentile of null PF distribution (500 permutations minimum). **Not** the 5× blanket rule from GoldBigBrain — that was over-conservative for some archetypes and under for others.

**Implement:** `validation/null_tests.py` with dispatch on declared `strategy.archetype` attribute.

---

## 4. Gate 4 — Walk-Forward Optimization

**Configuration:**
- Total history: 7 years (2018-01 → 2025-01, held-out 2025-01→2025-06 reserved)
- IS window: 18 months (training/tuning)
- OOS window: 6 months (evaluation)
- Step: 6 months (rolling)
- Windows: 11 IS/OOS pairs

**Pass criteria:**
- **≥ 8 of 11 OOS windows must have PF ≥ 1.2** (73%)
- Mean OOS PF ≥ 1.3
- No OOS window with DD > 25%
- OOS PF correlation with IS PF ≥ 0.4 (if tuning in IS is random, correlation is zero — we want tuning to actually transfer)

**Implement:** `validation/walk_forward.py`. Parallel across windows on Mac prefixes.

---

## 5. Gate 5 — Per-Year Stability

Split by calendar year 2018–2024 (7 full years). For each year independently compute PF and DD.

**Pass criteria:**
- **Minimum 5/7 years PF ≥ 1.15** (GoldBigBrain "On a Leash" benchmark: 7/7)
- At most 1 year PF < 1.0, and that year must have **documented economic rationale** (e.g., 2020 COVID, 2022 war-onset volatility) — not "it just was a bad year"
- Worst year DD ≤ 20%

This is the **M113 pattern** killer: if PF is driven by one freak year (e.g., 2023 alone carries the whole strategy), reject.

**Implement:** `validation/yearly.py` (emits per-year CSV + visual heatmap).

---

## 6. Gate 6 — Cost Stress Test

Every backtest must be re-run under worst-case cost assumptions:

| Stress | Modification |
|---|---|
| Spread ×2 | Vantage Standard EURUSD avg ~0.4 pips → stress to 0.8 pips |
| Slippage on stops | 50% of stops fill 1 pip worse than level |
| Commission re-added | $7/lot round-trip (Vantage Pro) even if using Standard |
| News-spike spread | 5 pips for 2 bars around scheduled releases (NFP, ECB, FOMC) |

**Pass criteria:** Under combined worst case, **PF ≥ 1.20** and DD not worse than 1.5× baseline DD.

If the strategy loses its edge under realistic adverse costs, it's a spread-fitted artifact — this is exactly what killed several GoldDigger variants in live.

**Implement:** `validation/cost_stress.py` — shim layer over the existing MT5 simulator that injects cost perturbations.

---

## 7. Gate 7 — Parameter Robustness (Edge-Cliff Detector)

For each tunable parameter `p` with "best" value `p*`, run backtests at `p* × {0.8, 0.9, 1.0, 1.1, 1.2}`. If any param has only one value that works, the strategy sits on a cliff.

**Pass criteria:**
- **No parameter where PF drops > 30% within ±20% range**
- Parameter surface must be a plateau, not a spike
- Visualize all univariate sweeps to `findings/robustness_<cfg>.png` for visual inspection

Extra check: run a **±10% multi-param jitter (100 perturbations)**. 95th-percentile PF should remain ≥ 1.2. This catches strategies where all params must line up simultaneously.

**Implement:** `validation/robustness.py`.

---

## 8. Gate 8 — Multi-Instrument Generalization (THE M113 KILLER)

The M113 lesson: a strategy that wiped 4 non-XAU accounts at 96–99% DD is not a strategy, it's an XAU-curve-fit.

**Test unchanged on:** GBPUSD M5, AUDUSD M5, USDCHF M5. (Not JPY — different tick size.)

**Pass criteria (pick one):**
- **Tier A (ideal):** PF ≥ 1.1 and DD ≤ 20% on ALL three. Strategy is structural.
- **Tier B (acceptable with justification):** PF ≥ 1.0 on 2/3 AND written economic rationale for why EUR-specific (ECB schedule, London/NY overlap specificity, EUR-funded carry mechanics). Rationale must be pre-registered before the cross-test — no post-hoc "oh yes this is why".
- **Tier C (reject):** Any instrument produces DD > 40% or negative PF on unchanged parameters. This is the M113 signature.

**Implement:** `validation/cross_instrument.py` — single-arg (strategy class) → 3-instrument report.

---

## 9. Gate 9 — Regime Sensitivity

Classify each month by:
- **VIX regime:** low < 15, normal 15–25, high > 25
- **DXY regime:** trending (|20-day slope| > 1.5σ) vs ranging
- **News density:** high (Tier-1 news weeks) vs quiet
- **Volatility regime:** EURUSD ATR(14) below or above 90-day median

**Pass criteria:**
- Strategy must be **profitable (PF ≥ 1.1) in at least 2 of the 3 VIX regimes**
- Must be profitable in both trending and ranging DXY
- If reversion strategy: tested specifically in ranging regimes (should win) AND trending (must not wipe)
- If breakout strategy: inverse

This catches "my strategy works great (but only in a quiet 2019-like regime)."

**Implement:** `validation/regime_split.py` — uses a regime lookup table generated once from VIX/DXY data.

---

## 10. Gate 10 — Forward Held-Out Test (Final Deploy Gate)

Never touched during Gates 1–9. Period: **2025-07-01 → 2026-04-01** (9 months held out).

No tuning, no parameter adjustment, no "we noticed this so we fixed it". Run the frozen config and accept the result.

**Pass criteria:**
- PF ≥ 1.25
- DD ≤ 1.3× worst walk-forward DD
- Trade count consistent with extrapolated IS frequency (± 30%)
- No single week accounts for > 25% of total PnL

If Gate 10 passes → candidate is **deploy-grade**. If it fails → the strategy is dead, do not iterate on it, do not try nearby configs (that creates leakage into the held-out set). Start over with a new idea.

---

## 11. Live-Monitoring Kill Switches

Post-deploy, the strategy is monitored by `live_monitor.py` (runs every 15 min off MT5 account history):

| Signal | Action |
|---|---|
| Rolling 60-day PF < 1.00 | Auto-pause, alert |
| Single-day DD > 5% of equity | Auto-pause, alert |
| Rolling 30-day DD > 1.5× backtest worst | Auto-pause, alert |
| 3 consecutive weeks with zero trades | Alert only (regime shift possible) |
| Trade frequency < 50% of backtest expectation (20-day rolling) | Alert — execution issue or regime |
| Spread-cost as % of PnL > 2× backtest | Alert — broker conditions changed |

All pauses write to a `kill_log.parquet` with root-cause entry required before any un-pause.

---

## P-Hacking Budget

**The problem:** GoldBigBrain ran ~1,000 backtests. With that N, the 95th percentile of best-in-sample PF under pure noise is approximately:

```
E[max PF | N=1000, T=7yrs M5] ≈ 1.55 (empirical, from random-signal simulation)
```

Meaning: **any PF below 1.55 on a 1000-test sweep carries zero evidence of real edge by selection-bias math alone.**

**Deflated-Sharpe-adjusted thresholds** (for EURUSD M5, 7yrs, ~2000 trades/yr):

| Backtests run (N) | Required raw PF for 95% conf |
|---|---|
| 10 | 1.25 |
| 100 | 1.40 |
| 1,000 | 1.60 |
| 10,000 | 1.85 |

**EuroBigBrain budget: 200 total backtests across the project's lifetime before Gate 2 DSR calculation** forces us to need PF > 1.45 raw in-sample. We hit that, we slow down, go back to theory, consult working group on mechanism.

Budget enforcement: the `validation_ledger.parquet` increments a counter on every backtest touching held-out data or the primary sweep set. Once N=200, any new backtest must include a pre-registration note in `preregistrations/` explaining the specific hypothesis being tested — this forces deliberation and stops reflexive parameter jiggling.

---

## Infrastructure to BUILD FIRST (before any strategy)

**Target: 2 weeks of pure plumbing. No strategy code until all of these exist + have passing unit tests.**

### Week 1 — Statistics package
1. `validation/bootstrap.py` — stationary block bootstrap, auto block-length, p-value API
2. `validation/dsr.py` — Deflated Sharpe Ratio, trial-count-aware
3. `validation/null_tests.py` — archetype dispatcher (flip, session-shuffle, label-perm, price-bootstrap)
4. `validation/ledger.py` — `validation_ledger.parquet` writer, enforces p-hacking budget
5. Unit tests: feed known-noise series → every gate must correctly reject

### Week 2 — Pipeline plumbing
6. `validation/walk_forward.py` — rolling IS/OOS harness, parallel across prefixes
7. `validation/yearly.py` — per-year + per-regime splitter
8. `validation/cost_stress.py` — cost perturbation shim over MT5 sim
9. `validation/cross_instrument.py` — GBPUSD/AUDUSD/USDCHF runner
10. `validation/regime_split.py` + regime lookup table generator (VIX/DXY data)
11. `validation/robustness.py` — univariate + multivariate parameter sweeps
12. `live_monitor.py` — kill-switch daemon

### Acceptance test for the infrastructure itself
Build a **known-good synthetic strategy** (trend-following on a trending synthetic series) and a **known-bad strategy** (coin-flip on real EURUSD). The pipeline MUST:
- Pass the known-good on Gates 1–9 (will skip 10, no held-out on synthetic)
- Reject the known-bad no later than Gate 2 or Gate 3

Only after both acceptance tests pass may real EuroBigBrain strategy candidates enter the pipeline.

---

## Summary

Eleven gates, archetype-matched nulls, Deflated Sharpe as the gatekeeper against p-hacking, cross-instrument as the M113 killer, held-out 9 months as the deploy gate, and kill-switch automation post-deploy. Infrastructure-first so we never again retrofit validation to a strategy we already like.

The GoldBigBrain survivor ("On a Leash", PF=1.76, 7/7 green years) would comfortably pass every one of these gates. Every rejected GoldBigBrain candidate (cfg74 reversion, M113, W148s, the 182-cfg sweep trio) would have been killed at Gate 2, 3, 5, or 8 — mostly at Gate 8 (multi-instrument). If we had this pipeline a month ago, we'd have saved 100+ hours of compute and avoided the reversal confusion on cfg74.

Sources:
- [The Deflated Sharpe Ratio (Bailey & López de Prado)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [The Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- [How to Use the Sharpe Ratio (López de Prado, Lipton, Zoonekynd, Sep 2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5520741)
- [arch.bootstrap.optimal_block_length — arch 7.2.0 docs](https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html)
- [Bootstrap Methods for Finance: Review and Analysis (Cogneau & Zakamouline 2010)](https://quantdevel.com/BootstrappingTimeSeriesData/Papers/Cogneau,%20Zakamouline%20(2010)%20-%20Bootstrap%20Methods%20for%20Finance.pdf)
- [Block length selection in the bootstrap for time series (Politis & White)](https://www.sciencedirect.com/science/article/abs/pii/S0167947399000146)
