# Prior Lessons for ML-EUR (next bot project, drafted 2026-04-22)

Per HARD RULE `feedback_apply_prior_lessons_to_new_project.md` — this doc MUST be read and applied before writing a single fire script for the ML-EUR pivot.

## What we already know (don't re-discover)

### Gates / thresholds (HARD numbers, not negotiable)
- **Null-test ratio**: real PF ≥ **5×** null median PF. EBB GBP failed at 4.34× (rejected). EBB EUR failed at 1.03× (rejected).
- **Min trade count for trustable PF**: **n ≥ 300** trades. n<150 = noise zone, don't even compute stats.
- **Multi-instrument support**: must clear PF≥1.0 on **≥2 non-target instruments** at the same params before any deploy talk.
- **Walk-forward**: 3yr train / 6mo test, **6/6 OOS years profitable** required. fg+rsisl cleared this; cfg74 didn't.
- **Bootstrap p-value**: **p<0.0001** on ≥1000 resamples. Reverses some null-FAILs (cfg74 case).
- **DD ceiling**: live max DD < **15%**. Beast Unchained at 35% was rejected for that reason alone.
- **Spread+slippage realism**: re-run final config with broker-realistic spread (Vantage XAU = check `reference_instrument_specs.md`, EUR M5 spread is much tighter).

### Anti-patterns (DON'T do these)
- ❌ "Run an exploratory sweep and see what survives" — that's the EBB failure mode. Design from prior knowledge first.
- ❌ Fine-grain a PF=1.10 candidate. PF<1.30 with n<200 = ceiling; fine-grain doesn't lift it.
- ❌ Relax VT to lift trade count. Wipeouts every time. Gold and EBB Wave-6 both confirmed.
- ❌ Save multi-instrument testing for "Wave 5". Do it Wave 2-3. M113 redux is the most-likely outcome for any port.
- ❌ Use Claude as inter-wave sequencer overnight. Mac-side chained driver only.
- ❌ Test low-VT (≤0.20) on a new instrument without instrument-native ATR calibration. Wave-6 GBP gave 99% DD across all 3 SLs at VT=0.20.
- ❌ Test fade strategies on EUR. EUR is **persistent / trending intraday** (Wave-3 atr_breakout-EUR confirmed); fade family wiped (Waves 1+2). Fade is a XAU-microstructure quirk, not portable.
- ❌ Re-test fade_long on any non-XAU instrument. M113 wiped EUR/GBP/JPY/XAG at 96-99.85% DD.

### Instrument fingerprints (microstructure)
- **XAU**: mean-reverting intraday → fade strategies viable (cfg74, beast).
- **EUR**: persistent / trending intraday → breakout/momentum family viable, fade family wipes.
- **GBP**: similar to EUR (clusters with EUR on atr_breakout — both directionally beat noise but underpowered for sample size).
- **JPY/CHF**: insufficient data (Wave 5 fired only 1 cfg each; both <PF 1.0).

### What worked once (the GBB ML recipe — the basis for ML-EUR)
- LightGBM meta-filter on 6 years XAU, AUC=0.72 multi-year (`project_goldbigbrain_walkforward.md`).
- 26/26 quarters profitable on validation (`project_goldbigbrain_regime_validation.md`).
- 6/6 OOS years pass; PF=1.27 @ threshold 0.38 (`project_goldbigbrain_ml_results.md`).
- Time-of-day = 65% of signal weight; MACD/h4_rsi were dead weight (`project_goldbigbrain_ablation.md`).
- ANE skip for trees: LightGBM/XGBoost get 0× from CoreML, only NNs benefit (`reference_ane_skip_for_trees.md`).

### Compute budget
- Mac M5: 4 prefixes parallel max for MT5 every-tick (RAM cap). Also viable: GPU + 8 Python workers for sim/feature pipelines.
- Win: 6 MT5 instances if RAM permits (3.3 GB peak per metatester64). NEEDS a fire script for parity with Mac — currently doesn't exist (12 Win-hours wasted on EBB because of this).

## ML-EUR minimum 4-wave plan (designed from prior knowledge)

**Wave 1 — Feature engineering + AUC sanity check** (30 min)
- Pull EURUSD M5 OHLCV 2020-2026 from MT5 history.
- Compute the same feature set GBB used (time-of-day primary; ATR / range / vol-rate secondary; skip MACD/h4_rsi).
- Train 5-fold CV LightGBM. **GATE: mean AUC ≥ 0.65** (lower bar than XAU 0.72 because EUR is trickier).
- If AUC < 0.65 → ML-EUR is dead, pivot to either H1 EUR or stop project.

**Wave 2 — Walk-forward 3yr/6mo** (1 hr)
- 6 OOS folds covering 2020-2026.
- **GATE: ≥ 5/6 OOS folds with PF > 1.0**.
- If <5/6 → ML-EUR is dead.

**Wave 3 — Multi-instrument transfer** (1 hr)
- Train on EUR, test on GBP, USDJPY, USDCHF (all available in Mac MT5).
- **GATE: PF ≥ 1.0 on ≥ 2 of 3 transfer instruments**.
- If only 1/3 → flag M113-redux risk; demote to slot specialist (EUR-only).

**Wave 4 — Null + bootstrap + realistic-cost** (1 hr)
- Permutation null test: shuffle session labels 5+ times. **GATE: real PF ≥ 5× null median**.
- Bootstrap 1000 resamples. **GATE: p < 0.0001**.
- Re-run final config with EUR realistic spread/slippage. **GATE: PF still > 1.0**.
- If all gates pass → ML-EUR is a deploy candidate. Hand off to user for deploy decision.

**Pre-stage**: write `run_ml_eur_overnight.sh` that chains Wave1→Wave2→Wave3→Wave4 with `&&`, each wave exits cleanly when manifest written, parser between waves emits `next_wave_ok.txt` or `kill.txt` to drive the chain. Claude only checks in periodically.

## Decision points encoded in advance
- Wave 1 AUC < 0.65 → kill ML-EUR, move to H1 EUR or stop.
- Wave 2 OOS < 5/6 → kill ML-EUR.
- Wave 3 transfer 0-1/3 → demote to EUR-only slot specialist; lower deploy bar requires ≥3 yr forward + extra null margin.
- Wave 4 any gate fail → kill ML-EUR.
- All 4 waves pass → ML-EUR is a candidate. STILL "research mode, no deploy" until user explicitly says go (per `feedback_no_deploy_research_mode.md`).
