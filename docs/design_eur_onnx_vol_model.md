# Design — EUR/USD ONNX Volatility Model for EBB_Core

_Version 1.0 · 2026-04-21 · Authoring basis: GoldBigBrain walkforward audit, bootstrap framework, and `findings/LESSONS_FROM_GOLDBIGBRAIN.md` §1.3 (lookahead), §3.1 (vol-filter AUC), §3.2 (walk-forward schedule), §6.1 (EUR-specific feature delta); WG4 architecture note._

**Non-goal:** port `vol_model_6yr.onnx` to EUR. Gold's model was trained on XAU M5 microstructure — features like `atr_ratio_14_50` scale to $1–3 movements and `minutes_since_session` peaks at the London-PM / NY-AM fix. On EUR/USD that output is noise with false confidence.

**Goal:** build `vol_model_eur_6yr.onnx`, a LightGBM binary classifier predicting `max_excursion_next_5bars > 1.5 × ATR(14)`, ONNX-exported, consumed by `EBB_Core.mq5` as a veto gate before any mechanical entry. Same role gold's model plays: **predict WHEN, never WHICH DIRECTION** (AUC 0.67 vs 0.51 settled this).

---

## 1. Feature engineering for EUR/USD

EUR's microstructure differs from gold in four dimensions — each forces a feature or a re-scaling.

| Axis | XAUUSD | EURUSD | Feature implication |
|---|---|---|---|
| Spread vs ATR | 18pt / ATR ~200pt (9%) | 1.1pip / ATR ~8pip (14%) | `spread_z` becomes more informative; absolute-cost features must be ratio-based |
| Session peaks | 13–20 GMT monotone ramp | Twin peaks: 07–10 London open, 13–16 LDN/NY overlap | Hour-of-day encoded as Fourier pair (sin/cos) plus explicit session dummies |
| Event sensitivity | Geopolitics + CPI + gold futures roll | ECB, FOMC, NFP, DXY, Bund yield | Event-proximity feature + DXY/Bund correlation |
| Weekly rhythm | No carry | Rollover at 21:00; Wed triple-swap | Day-of-week encoded; avoid 20:00–21:30 Wed entries |

**Candidate feature list (38 features, all causal):**

_Microstructure (9):_ `atr5`, `atr14`, `atr_ratio_5_14`, `atr_ratio_14_50`, `bb_width_20`, `bb_width_z`, `spread_z_1h`, `tick_count_z_1h`, `m5_body_wick_ratio`.

_Vol regime (6):_ `realized_vol_1h`, `realized_vol_4h`, `realized_vol_ratio_1h_4h`, `hurst_128`, `sample_entropy_64`, `wavelet_energy_L0` (Daubechies-4, 64-bar window).

_Temporal (8):_ `hour_sin`, `hour_cos`, `minute_of_hour`, `minutes_since_london_open`, `minutes_since_ny_open`, `day_of_week_sin`, `day_of_week_cos`, `is_london_ny_overlap` (binary).

_Cross-asset (5):_ `dxy_ret_1h`, `dxy_ret_4h`, `eurgbp_ret_1h` (EUR-specific stress), `bund_yield_change_1h`, `vix_level_z` (risk-on/off).

_Asian-range / structural (4):_ `dist_to_asian_high_atr`, `dist_to_asian_low_atr`, `asian_range_atr`, `is_asian_range_broken`.

_Event proximity (3):_ `minutes_to_next_high_impact`, `minutes_since_last_high_impact`, `in_30min_post_release` (binary). Calendar: Forex Factory high-impact EUR/USD.

_Micro-momentum (3):_ `rsi_14`, `rsi_5_minus_14`, `ema_8_slope_atr`.

**Rationale ranking (what should dominate):** `minutes_since_london_open`, `wavelet_energy_L0`, `atr_ratio_5_14`, `hour_sin/cos`, `realized_vol_ratio_1h_4h`, `dxy_ret_1h`, `spread_z_1h`. Gold's top-feature profile (`minutes_since_session`, `wavelet_energy_L0`, `atr_ratio_14_50`, `hour`, `sample_entropy_64`, `hurst_128`) suggests temporal + vol-regime + entropy will carry EUR too; the new entrants are `dxy_ret_1h` (EUR ≡ 1/DXY to first order) and `spread_z_1h` (retail crowding signal — EUR's spread widens on real liquidity events).

**Dead weight we skip from the start** (gold ablation results): MACD, h4_rsi, h4_trend, any EMA ribbon. Adds noise, inflates LightGBM's `num_leaves` without AUC gain.

---

## 2. Lookahead-bias prevention checklist

Gold's swing-detection bug (`high[i-lookback:i+lookback+1]`) faked AUC 0.50 → 0.69. We do not repeat it. Every feature passes **all** of these before merge:

1. **Window shape audit.** Every rolling computation is `series[i-N : i+1]` or `series.shift(1).rolling(N)`. No `i+k` for any k≥1. Grep for `+1` indexing on close/high/low arrays and inspect each hit.
2. **Shift discipline.** All cross-series merges use `.shift(1)` before joining (DXY, Bund, event calendar). Event-proximity features computed from the timestamped schedule only — never `groupby(date).transform('min')` which leaks same-day future events.
3. **Unit test per feature.** `tests/features/test_<name>.py`: given synthetic series with known structure, feature at index T must equal value computed from `series[:T+1]` only. Parametrize over random indices; fail on any mismatch.
4. **Permutation-invariance check.** For each feature, permute values AFTER index T and recompute at T; value must be identical. If it changes, feature peeks forward. CI job.
5. **Target isolation.** `__target__ = (max_excursion[t+1:t+6] > 1.5 × atr[t])` lives in a separate builder that runs AFTER feature materialization. Features cannot import the target module. Import-graph linted.
6. **Purge gap.** Walk-forward splits have a 24-bar (2h) purge between train-end and test-start so overlapping target windows can't leak label info.
7. **Feature-importance sanity.** When a single feature's gain > 3× the second-place feature, auto-flag and require manual re-audit. (Gold's swing bug produced exactly this signature.)
8. **Replay parity test.** Train a model, run inference on bar T using full dataframe; then truncate dataframe to `[:T+1]` and re-run; outputs must match to 1e-6. Single flag of failure = reject.

Every check is a CI gate. A PR that adds a feature must include the unit test; the permutation-invariance job runs on every commit.

---

## 3. Walk-forward training schedule

Replicates gold's 6-fold expanding-window pattern (LESSONS §3.2, 6/6 OOS years passed, AUC 0.698–0.736).

| Fold | Train | Test (OOS) | Purge |
|---|---|---|---|
| 1 | 2020 | 2021 | 24 M5 bars (2h) |
| 2 | 2020–2021 | 2022 | 24 bars |
| 3 | 2020–2022 | 2023 | 24 bars |
| 4 | 2020–2023 | 2024 | 24 bars |
| 5 | 2020–2024 | 2025 | 24 bars |
| 6 | 2020–2025 | 2026-YTD | 24 bars |

Internal validation: last 10% of each train slice for LightGBM early-stopping (mirrors `walkforward_audit_vol_model.py` lines 166–184). Params: `learning_rate=0.02`, `num_leaves=63`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`, `min_data_in_leaf=200`, `num_boost_round=2000`, `early_stopping=50`.

**AUC thresholds:**

- **Hard floor:** AUC ≥ 0.60 on every OOS fold. Below this, signal is indistinguishable from noise given EUR's efficiency.
- **Target band:** AUC 0.63–0.70 per fold. Gold hit 0.698–0.736; EUR is more efficient and more crowded, expect ~0.03–0.05 AUC below gold.
- **Over-ceiling flag:** AUC > 0.75 → lookahead suspect. Re-run §2 checklist 1, 4, 8 before accepting.
- **Permutation null:** shuffle `__target__` 200× per fold, report `(real_AUC - null_mean_AUC) / null_std`. Must be ≥ 3σ every fold.

Deploy-candidate gate: ≥ 5/6 folds in the 0.60–0.75 band, zero folds below 0.55, permutation z-score ≥ 3σ on ≥ 5/6 folds. Fails = pivot features or instrument, do NOT deploy the model.

Production model: retrain on full 2020–2026 AFTER the walk-forward gates pass. The "released" model has seen everything; the walk-forward exists to prove the training procedure generalizes, not to ship a train-2020-2023 model.

---

## 4. Multi-instrument generalization test

Gold's edge did NOT generalize (M113 wipe on XAG/GBP/JPY at 96–99.85% DD — memory `project_m113_xauusd_overfit`). EUR must clear the inverse test up-front.

**Protocol:**

1. Train on EUR/USD 2020–2024 with the full walk-forward procedure above.
2. Freeze the final EUR-trained model.
3. Build GBP/USD and USD/JPY M5 feature matrices using the SAME feature code, SAME normalization constants (per-symbol ATR-based, so `atr_ratio_5_14` is already unitless; `dxy_ret_1h` is shared).
4. Compute AUC on GBP 2024–2026 and JPY 2024–2026. No retraining.
5. **Decision table:**

| Result | Interpretation | Action |
|---|---|---|
| EUR AUC ≥ 0.60, GBP+JPY AUC ≥ 0.58 | Feature set captures FX vol regime broadly | Ship, label as "FX-general vol model" |
| EUR AUC ≥ 0.60, GBP OR JPY AUC 0.55–0.58, other fails | Partial generalization — cross-pair flow features (DXY, Bund) carry some signal | Ship EUR-only, note instrument constraint in deploy config |
| EUR AUC ≥ 0.60, both GBP+JPY < 0.55 | EUR-specific microstructure model | Ship EUR-only, label "EUR-specialist" |
| EUR AUC < 0.60 | Feature set or instrument dead | Pivot: rebuild feature set OR move primary to GBP/DXY (LESSONS §3.4 WG4 week-4 pivot) |

Multi-instrument test runs as part of the CI gate, not a post-hoc experiment.

---

## 5. ONNX export + EBB_Core integration

**Export pipeline:** `skl2onnx.convert_lightgbm(booster, initial_types=[('input', FloatTensorType([None, 38]))], target_opset=17)`. Fixed feature order checked into `configs/models/eur_vol_v1/feature_order.txt`; hash written to `metadata.json` alongside git SHA and data hash.

**Model artifact name:** `vol_model_eur_{YYYY}_{git_sha8}_{data_hash8}.onnx`. EA loads by filename; SHA mismatch at init = refuse to start (gold pattern, LESSONS §4.11-adjacent).

**EBB_Core.mq5 integration (mirrors `GBB_Core.mq5` pattern, LESSONS §3.3, §6.8):**

```
OnInit():
  vol_handle = OnnxCreate("vol_model_eur_<hash>.onnx", 0);
  ArraySetAsSeries(features, false);
  // Assert expected input shape [1, 38]
  if (!OnnxSetInputShape(vol_handle, 0, {1, 38})) Fail();

OnTick() / per-M5-bar:
  BuildFeatures38(features);           // mirrors Python feature code 1:1
  OnnxRun(vol_handle, ONNX_NO_CONVERSION, features, vol_prob);
  if (vol_prob < VolThreshold) return;  // veto — no trade this bar
  DispatchSignal();                     // fade_long / momentum / etc.
```

`VolThreshold` is an EA input, default 0.38 (gold's sweet spot per `project_goldbigbrain_ml_results`); re-tuned during week-4 ablation on EUR. Must re-derive from EUR's PF-vs-threshold curve; do NOT assume 0.38 generalizes.

**Parity test (merge gate):** Python inference and MQL5 ONNX inference on the same 1000 bars must agree within 1e-5 on `vol_prob`. Any divergence = feature-builder drift between Python and MQL5; block merge.

**Feature builder in MQL5:** `EBB/VolFilter.mqh` implements `BuildFeatures38()`. Every Python feature has a line-for-line MQL5 equivalent; ablation tests run both sides on a fixed CSV fixture and assert byte-equal output.

---

## 6. Compute budget

Reference: gold's single walk-forward fold trains on ~500k M5 rows with LightGBM, early-stopping around boost round 400–600, ~60–120 seconds per fold on Mac M5 with `N_JOBS = cpu_count - 1 = 9`.

EUR/USD M5 row count 2020–2026 ≈ 600k (FX trades 24/5, more bars than gold's 24/5 with Sunday gap).

**Per-run estimate:**

- 6 walk-forward folds × ~90s each = 9 minutes.
- 200 permutation-null folds × ~50s (null target converges faster, fewer boost rounds) = ~170 minutes. Parallelize 8-wide on Mac M5 → ~22 minutes wall.
- Multi-instrument inference (GBP, JPY): seconds.
- Feature engineering (cold): ~15 minutes single-threaded; cached to Parquet thereafter.
- ONNX export + parity test: 2 minutes.

**Total wall time per training run: ~45–55 minutes on Mac M5** with `caffeinate -i -s` (LESSONS §4.4) and `N_JOBS=9` (LESSONS §5.13, saturate hardware).

**Hardware:** Mac M5 only. Reason: LESSONS §4.11 — LightGBM gets zero benefit from ANE/CoreML; GPU is useless here. `reference_ane_skip_for_trees` memory explicit. The Windows laptop stays on MT5 every-tick validation of candidate strategies in parallel (LESSONS §4.1).

**Iteration budget to first deploy-candidate model:**

- v0 baseline (all 38 features, no ablation): 1 run = 1h.
- Ablation (drop-one importance test across top-15 features): 15 runs = 12h wall if serial, or 2h wall with 6-way parallel folds on Mac.
- Hyperparameter sweep (`num_leaves` ∈ {31, 63, 127}, `learning_rate` ∈ {0.01, 0.02, 0.05}): 9 runs × 1h = 9h or 2h parallel.
- Multi-instrument + null validation on winner: 1h.

**Total calendar time — clean path: ~1 working day on Mac M5.** With the expected two or three detours (feature bug discovered, AUC ceiling hit, a null-test fails once), **realistic time-to-first-trained-model = 2–3 days**. Fits WG4's week-4 ML slot (line 261) with budget for the week-4 pivot checkpoint if EUR AUC floors below 0.60.

No GPU hours needed. No ANE hours needed. CPU-bound start to finish.

---

_End of design v1.0._
