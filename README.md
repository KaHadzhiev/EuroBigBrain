# EuroBigBrain (EBB)

> Sibling research project to [GoldBigBrain](https://github.com/KaHadzhiev/GoldBigBrain).
> EUR/USD M5 portfolio strategy on Vantage Standard STP, $1k demo, research-only.

**Status:** PRE-ALPHA. Design phase complete. `EBB_Core.mq5` skeleton scheduled but not yet built. Zero validated configs. Nothing is deployable. This repository is a lab notebook plus scaffolding — not a trading system.

**Date initialised:** 2026-04-22

---

## What EBB is

EuroBigBrain is the EUR/USD counterpart to GoldBigBrain. The goal is a deploy-grade EUR/USD M5 portfolio strategy that is demonstrably better than the London Beasts (gold) winners on risk-adjusted terms, with validation rigorous enough that cfg74-style reversal confusion and M113-style single-instrument overfit cannot happen again.

**What is NOT inherited:**
- Gold parameters (VT, SL, TP thresholds) — EUR has 1-pip spread vs gold's 18 points, carry-driven vs inflation-hedge, retail-crowded. Numbers re-derive from EUR microstructure.
- ML-predicts-direction — gold's direction model stalled at AUC 0.51. EBB uses ML as a vol veto-filter only (AUC target ≥ 0.60).

**What IS inherited:**
- The GBB_Core architecture (event loop, bracket lifecycle, ATR-scaled sizing).
- Tick-replay engine (Phase 2A — needs the per-day level-crossed guard patch that caused the 0.195 calibration catastrophe).
- ONNX inference harness.
- 80+ memory files of anti-patterns — encoded in `findings/LESSONS_FROM_GOLDBIGBRAIN.md`.

---

## The 5 critical decisions

Distilled from `findings/LESSONS_FROM_GOLDBIGBRAIN.md` and `PLAN.md` §2 (C1-C10). These are load-bearing and non-negotiable; violating any triggers a hard-stop review.

### 1. Multi-year MT5 every-tick is the only ground truth

Python sims inflate PF 1.4–1.8x even after post-rebuild calibration. The M1 sim had median ratio 0.195 vs tick-replay — catastrophic. v1.1 was locked on a 3.5-month forward test claiming PF=1.54; full 2020–2026 every-tick revealed PF=0.91 and 92.7% DD (account wipe). No config gets locked from Python sim. Calibration ratio (`sim_PF / mt5_PF`) must be published in every run report.

### 2. Pipeline BEFORE strategy (validation harness is Week 1-2, not "later")

Validation gates (bootstrap, deflated Sharpe, walk-forward, cross-instrument null, direction-flip, block-bootstrap) must exist, be unit-tested, and be CI-enforced before the first strategy candidate is written. WG3 + WG4 + feedback_fix_verify_before_grid converged on this. Building strategy first and "validating later" is how every prior cycle burned weeks on noise.

### 3. Archetype-matched nulls (cfg74 reversal)

Session-shuffle nulls tested the wrong hypothesis for reversion strategies. cfg74 "failed" 5x session-shuffle yet later passed bootstrap p<0.0001, 7/7 years profitable, direction-flip significant — real edge, window-invariant. Null tests must match the archetype:
- Reversion -> direction-flip + bootstrap (ratio bar 1.5x)
- Breakout -> session-shuffle (ratio bar 2.0x)
- ML -> label-permutation (ratio bar 1.5x)

### 4. Multi-instrument generalisation from line 1 (M113 killer)

M113 fade_long was locked on XAUUSD. Calibrated retest on XAGUSD/EURUSD/GBPUSD/USDJPY: 4/4 accounts wiped at 96–99.85% DD. The "mean-reversion edge" was pure gold microstructure. Every component in EBB takes `symbol` as a parameter. Cross-instrument null (EUR, GBP, AUD, CHF with identical params) is a CI gate; PF >= 1.10 on 2/3 and zero wipeouts (DD > 40%) anywhere required.

### 5. Kill-switch in the core, non-overridable + research-only

Daily loss cap, account DD cap (8% auto-halt, 12% design ceiling), heartbeat watchdog, spread guard — all enforced in `RiskCore.mqh` / `KillSwitch.mqh`, called by `EBB_Core`, never by signal code. No live deploy until user explicitly greenlights. Every winner is "current best candidate", never "ready to deploy."

Bonus axioms (C6 / C9): don't port gold parameters; `workers=1` for any deploy-critical MT5 grid (workers>1 wine-prefix race corrupts CSVs).

---

## Repo layout

```
EuroBigBrain/
|-- PLAN.md                         # Project bible (append-only, sign-off required)
|-- README.md                       # This file
|-- .gitignore
|-- docs/                           # Design docs (one per subsystem)
|   |-- design_ebb_core_refactor.md
|   |-- design_tick_replay_calibration.md
|   |-- design_eur_onnx_vol_model.md
|   |-- design_validation_gates_ci.md
|   |-- design_validation_block_bootstrap.md
|   |-- design_deflated_sharpe.md
|   |-- design_multi_instrument_gate.md
|   |-- design_dxy_cross_asset.md
|   |-- design_data_pipeline.md
|   |-- design_portfolio_combine.md
|   |-- design_kill_switch_sizing.md
|   `-- eur_session_windows.md
|-- findings/                       # Synthesis from WG1-WG4 + GBB memory
|   |-- LESSONS_FROM_GOLDBIGBRAIN.md
|   |-- wg1_market_microstructure.md
|   |-- wg2_strategy_archetypes.md
|   |-- wg3_validation_pipeline.md
|   `-- wg4_architecture_improvements.md
|-- python/                         # Validation harness + data pipeline
|   |-- data/                       # fetch_history, quality_audit
|   |-- features/                   # dxy_synthesizer (DXY cross-asset)
|   |-- filters/                    # news_blackout + tests
|   |-- portfolio/                  # combine, correlation_test
|   `-- validation/                 # bootstrap, walk_forward, deflated_sharpe,
|                                   #   kill_switch_simulator, multi_instrument_null + tests
|-- mql5/
|   |-- EBB/                        # EBB_Core target (KillSwitch.mqh, Sizing.mqh)
|   `-- EuroBigBrain/               # Reserved for compiled EA deliverables
|-- configs/                        # JSON configs (test_smoke.json, future strategy configs)
|-- runs/                           # MT5 grid + Python validation run artefacts (mostly gitignored)
|-- data/                           # Broker M1 ticks, DuckDB warehouse (gitignored)
`-- results/                        # Cross-run aggregated results
```

---

## Current state (2026-04-22)

| Week | Status | Notes |
|---|---|---|
| Week 1 — infra skeleton | IN PROGRESS | Validation modules built (bootstrap, walk-forward, DSR, MI-null, kill-switch sim). EBB_Core.mq5 not yet written. |
| Week 2 — harness + calibration | PENDING | Tick-replay engine patch still owed. 100-config calibration gate not yet fired. |
| Week 3 — first candidate | NOT STARTED | NY Reversal Fade, week 3 deliverable per PLAN.md §4. |
| Week 4–6 | NOT STARTED | Asian Range Breakout, ML vol-filter, portfolio combine, 9-month held-out forward. |

Smoke test runs exist under `runs/multi_instrument_null/test_smoke_*` but produced empty logs — smoke harness scaffolding is in place, real validation has not fired.

---

## How to run (when things exist)

```bash
# Python validation (Windows)
cd C:\Users\kahad\IdeaProjects\EuroBigBrain
python -m pytest python/validation/tests -v
python -m pytest python/filters/tests -v

# MT5 grid (Windows prefix)
bash runs/wave1_eur_fade_long/fire_win_eur.py

# MT5 grid (Mac prefix — via SSH)
ssh kalinhadzhievm5@192.168.100.68 "cd ~/IdeaProjects/EuroBigBrain && python3 runs/wave2_eur_session_x_vt/fire_mac_wave2_v2.py"
```

Tick data and compiled `.ex5` files are not in the repo — fetched or compiled locally.

---

## Disclaimer

**NOT FINANCIAL ADVICE.** EBB is a research project. Nothing here has been validated to production standard. Most of what looks promising in sim will not survive MT5 every-tick multi-year validation — that is the entire point of the harness.

EUR/USD is a different beast from gold: tighter spread, higher retail crowding, carry-driven flows, ECB/Fed schedules, DXY correlations. Gold lessons (methodology, validation rigour, anti-patterns) transfer. Gold numbers do not.

This repository is public for transparency and personal archival. Do not trade off anything here. If you fork it and lose money, that is entirely your own doing.

---

## License

Research-only. No license granted for commercial use. Contact the repo owner for anything beyond "reading the code."
