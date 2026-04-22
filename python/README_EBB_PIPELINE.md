# EBB Pipeline (Tick -> Sim -> MT5 -> Calibrate)

End-to-end EuroBigBrain pipeline cloned 1:1 from GoldBigBrain pattern.
**Calibration target: <5% PF gap between Python sim and MT5 every-tick (vs GBB's 11%).**

---

## Fire Order (run in sequence)

| Step | Script | Input | Output | Wall time (est.) |
|------|--------|-------|--------|------------------|
| 0 | `pull_dukascopy_eurusd.py` | (network) | `data/dukascopy/eurusd_ticks_YYYY-MM.parquet` x76 | ~2h |
| 1 | `merge_dukascopy_eurusd.py` | step 0 | `data/eurusd_ticks_2020_2026.parquet` | ~2 min |
| 2 | `dukascopy_to_m5_bars.py` | step 1 | `data/eurusd_m5_2020_2026.parquet` + `data/EURUSD_M5_6yr.csv` | ~3 min |
| 3 | `build_eurusd_features_6yr.py` | step 2 | `data/eurusd_features_6yr.parquet` | ~5 min |
| 4 | `train_eurusd_lightgbm_6yr.py` | step 3 | `models/eur_tb_h10_6yr.onnx` + `models/eur_tb_h10_6yr_feature_list.txt` + `data/eurusd_oof_probs_6yr.parquet` | ~10 min |
| 5 | `sim_eurusd_6yr.py` | step 4 | `results/sim_eurusd_6yr_threshold_sweep.csv` | ~1 min |
| 6 | `run_mt5_thr_sweep.py` (existing) | step 4 ONNX | `runs/mt5_thr_sweep.json` | ~10 min |
| 7 | `calibrate_sim_vs_mt5.py` | steps 5+6 | `results/calibrate_sim_vs_mt5.csv` + `results/calibrate_metrics.json` + `results/eurusd_isotonic_calibrators.joblib` | <30 sec |

**Pre-step**: copy `models/eur_tb_h10_6yr.onnx` to MT5 EA resource path before re-compile + step 6.

---

## Per-script details

### 1. `merge_dukascopy_eurusd.py`
Concatenates the 76 monthly Dukascopy parquet shards into `data/eurusd_ticks_2020_2026.parquet`.
Sorts by time, dedups on `time` column. Schema preserved (time/bid/ask/last/volume/flags).

### 2. `dukascopy_to_m5_bars.py`
Aggregates ticks (bid mid) into M5 OHLCV bars. Volume = tick count per 5-min bucket
(MT5-aligned). Drops bars with zero ticks. Writes parquet + CSV mirror.

### 3. `build_eurusd_features_6yr.py`
**Same 23 features** as `build_eurusd_features.py` -- this preserves the EBB EA contract.
Reads the new 6yr Dukascopy M5 parquet. Cross-asset features (EURGBP/XAU/DXY) loaded
from GBB's data dir if available; else neutral 0 (matches EA `RequireCrossSymbols=false`).

### 4. `train_eurusd_lightgbm_6yr.py`
Trains on 2020-01..2024-01 (~4yr), tests on 2024-01..2026-04 (~2.3yr OOS).
Triple-Barrier label TRUE-hilo with h=10 / SL=0.7xATR / TP=2.0xATR.
**Exports ONNX with `zipmap=False`** -- mandatory for MQL5 EA per
`project_ebb_DEPLOY_CANDIDATE_h10` (EA expects `[batch, 2 classes]` plain float tensor).
Also dumps OOF probabilities to `data/eurusd_oof_probs_6yr.parquet` for the sim.

### 5. `sim_eurusd_6yr.py`
Python sim of long-only Triple-Barrier on OOF probs. Threshold sweep 0.30..0.50.
**Cost model**: spread = 1.0 pip Vantage (= ~0.5 pip Dukascopy ECN + 0.5 pip retail markup),
slippage 0.2 pips per fill. Entry at next-bar OPEN.

### 6. `run_mt5_thr_sweep.py` (existing)
Fires 4 MT5 every-tick backtests on Instance2, parses HTML reports for PF/trades/DD.
Currently set to 4 thresholds (0.30/0.32/0.35/0.40) over 2025-01..2026-04. **Adjust THRESHOLDS
list to match the sim grid AND the test window to span 2024-01..2026-04 to align with sim OOF.**

### 7. `calibrate_sim_vs_mt5.py`
Compares paired thresholds:
- Per-thr PF delta (target: <5% per gate)
- Multi-metric vector (Spearman rank-corr / KS PnL / Sharpe-band overlap)
- **Fits isotonic regression** + (if >=8 paired thresholds) GBR for auto-correction

---

## Beat-GBB Strategy

GBB hit 11% sim/MT5 PF gap. EBB targets <5% via three combined fixes:

1. **Isotonic regression auto-calibrator** (`isotonic_calibration.py` ported pattern)
   Learns monotone `f(sim_pf) -> mt5_pf`, removes systematic skew without code changes.

2. **Vantage spread MARKUP modeled explicitly** -- not just session-based spread.
   Dukascopy is ECN (~0.5 pip), Vantage Standard adds ~0.5 pip retail markup -> total 1.0 pip.
   GBB only modeled session spread, missed the broker delta entirely.

3. **EURUSD-tuned slippage** -- 0.2 pips per fill, much tighter than XAUUSD's 1.5 pts.
   EURUSD is the most liquid pair -> negligible slippage on micro-lot market orders.

---

## Models / paths

- ONNX (deployed): `~/IdeaProjects/EuroBigBrain/models/eur_tb_h10_6yr.onnx`
- ONNX (current 16mo): `~/IdeaProjects/EuroBigBrain/models/eur_tb_h10.onnx`
- EA: `~/IdeaProjects/EuroBigBrain/mql5/EBB_TripleBarrier.mq5`
  Resource line: `#resource "eur_tb_h10.onnx" as uchar OnnxModelData[]`
  (Update to `eur_tb_h10_6yr.onnx` AND copy the file to MT5's resource path before recompile.)

---

## Memory references

- `project_ebb_DEPLOY_CANDIDATE_h10` -- 16mo MT5 PF=2.21@thr=0.44, PF=1.77@thr=0.35
- 6yr MT5 every-tick: PF=1.05@thr=0.35 (overfit confirmed -> need 6yr retrain = step 4)
- `feedback_null_test_before_mt5` -- MUST null-test the new 6yr model before live deploy
- `feedback_step_back_before_iterating` -- read GBB scripts end-to-end FIRST (done)
