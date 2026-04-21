# EuroBigBrain — Architecture Improvements (WG4)

**Author:** Architecture Improvement Specialist, 4-agent Working Group
**Date:** 2026-04-21
**Target:** EUR/USD M5 primary, multi-instrument generalization from day 1
**Reference:** GoldBigBrain (XAUUSD, PF=1.76 / 13.9% DD, ~6 months)

---

## TL;DR

- **Throw away the M5 OHLC sim entirely.** It lied (1.66× PF inflation) and cost us months. EuroBigBrain uses tick-replay as the ONLY sim layer, calibrated and unit-tested against MT5 every-tick before any strategy is written.
- **Validation infrastructure before strategies.** Null-test, bootstrap, walk-forward, MC-DD gates are CI jobs that run on every commit from day 1. A strategy that doesn't fit the CI harness can't be merged — no retrofit gates.
- **One EA binary, one config schema, many strategies.** Keep GBB's modular EA pattern but bake risk/kill-switch/daily-loss-cap into the *core* as non-overridable, not per-strategy. Strategies plug in as entry signal classes, nothing more.
- **Stay on LightGBM for ML; don't chase TabNet.** 2026 benchmarks still show gradient boosting wins on financial tabular data with narrower variance and 10–100× cheaper training. Neural nets buy <2% edge for 20× compute.
- **Six-week target to deployment-candidate.** Weeks 1–2 infra, week 3 port cfg74-style baseline, week 4 ML filter, week 5 multi-instrument validation, week 6 paper-live + kill-switch drills. If week 4 doesn't beat null on EUR/USD, pivot instrument don't pile epicycles.

---

## 1. EA Architecture — One Binary, Hard Core + Pluggable Signals

**Decision: ONE mega-EA (`EBB_Core.mq5`), like GBB_Core, but with stricter separation.**

Multiple specialized EAs sound clean but multiply the bug surface: each needs its own kill-switch, its own slippage model, its own position manager. GBB had one and it worked; the failure mode wasn't architecture, it was per-strategy contamination of shared state (e.g., retro-filter trap).

**Structure:**

```
EBB/
  Core.mqh           // event loop, tick handler, global state
  RiskCore.mqh       // position sizing, SL/TP placement, kill switch — NON-OVERRIDABLE
  KillSwitch.mqh     // daily loss cap, account DD cap, heartbeat watchdog
  PositionMgr.mqh    // open/modify/close, BE, trailing
  Session.mqh        // hour/day gating, news blackout
  VolFilter.mqh      // ONNX inference wrapper
  Telemetry.mqh      // structured logging → CSV + journal
  Signals/
    Fade.mqh         // fade_long (cfg74 heir)
    Momentum.mqh
    Breakout.mqh
    Base.mqh         // abstract interface — every signal implements GetSignal(), GetSLTP()
```

**Non-overridable core:** RiskCore and KillSwitch are called by Core, not by signals. A signal returns `{direction, sl_atr_mult, tp_atr_mult, confidence}`; it CANNOT size the position, CANNOT bypass daily loss, CANNOT disable the kill switch. This kills the class of bug where a per-strategy tweak silently disables a safety (GBB had several near-misses here).

**Kill switch baked from day 1** (not bolted on at deployment):

- Daily loss cap: 2% of start-of-day equity, hard close all, halt until next session.
- Account DD cap: 8% from all-time-high equity — halts EA, emails user (using user phone ping rule).
- Heartbeat watchdog: if no tick processed in 90s during market hours, flag in journal.
- Spread guard: reject signals when spread > 2× 1h median (EURUSD typically 0.6–1.5 pips; guard at 3 pips).
- Magic-number isolation: every EA instance owns a magic range, refuses to touch others' positions.

**Day-1 requirement:** kill-switch must be tested with a unit test that forces a synthetic 2% intraday loss and verifies all positions close + new orders rejected. This test is a merge gate.

---

## 2. Backtest Infrastructure — Tick-Replay Only, No M5 Sim

**Decision: skip M5 OHLC sim entirely. Use MT5 every-tick as ground truth, build one internal tick-replay engine that is calibrated to MT5 within ±3% PF before accepting any strategy result.**

GoldBigBrain's M5 sim inflated PF by 1.66× median. Every month we "validated" another false winner that died in MT5. The Phase 2A tick-replay engine is the right answer but had bugs (missing per-day level-crossed guard). We finish that engine on week 1 and gate it with a 100-config calibration suite before using it.

**Pipeline:**

1. Strategy code in Python mirrors EA signal logic exactly (shared config JSON).
2. Tick-replay engine replays M1-tick-reconstructed data; ONNX-compatible inference in-loop.
3. **Calibration gate:** 100 randomly-sampled configs must have |sim_PF − mt5_PF| / mt5_PF ≤ 0.05 on 2023–2025. Until this passes, NO sim result is trusted for ranking.
4. MT5 every-tick is the FINAL gate on any deploy candidate; sim is for screening only.

**Eliminating worker races:**

GBB's workers>1 race condition (wine prefix log stitching) wasted multiple full-grid runs. EuroBigBrain fix:

- Each worker gets its own isolated prefix directory AND a unique magic number AND a unique report filename prefix with UUID.
- Result collector reads only from `{uuid}_{config_id}.xml`, never scans directory.
- Pre-flight test on every run: launch 4 workers, verify 4 distinct UUIDs in output before proceeding.
- Workers=1 remains the fallback for any deploy-critical grid (documented rule).

---

## 3. Validation Infrastructure — Gates Before Strategies

**This is the single biggest time saver vs GoldBigBrain.** GBB added null-tests after strategies were tested, so gates got tuned to existing winners. Cascade: accept p<0.01 for cfg74, reject cfg28 at same bar = confirmation bias.

**Build order (week 1–2, before ANY strategy code):**

1. **Null-test harness** — permutes signal labels within sessions, 1000 runs, reports real/null ratio. CI runs this on every PR. Gate: ratio ≥ 2.0× median, ≥ 5× median for deploy-candidate label.
2. **Bootstrap harness** — resamples trades with replacement, 10k bootstraps, reports PF 95% CI. Gate: CI lower bound > 1.1.
3. **Walk-forward harness** — 6 yearly OOS folds, each trained on prior years only. Gate: ≥ 5/6 years profitable, zero years with DD > 25%.
4. **Monte Carlo DD harness** — resamples trade sequence 10k times, reports p99 DD. Gate: p99 DD ≤ 20% for deploy, ≤ 30% for candidate.
5. **Multi-instrument harness** — same strategy config on EURUSD (primary), GBPUSD, AUDUSD (secondary). Gate: PF > 1.2 on ≥ 2/3 instruments, zero wipeouts (DD > 40%) on any.
6. **Determinism CI** — every commit reruns a reference backtest twice, diffs CSV output byte-for-byte. Gate: identical or commit is rejected.

All six gates run as GitHub Actions on every PR. `main` is the deployment branch — nothing merges without all six green. Branching strategy: one branch per candidate strategy, PR back to main when all gates pass.

This is the Implementation Risk finding from the 2026 arxiv paper applied: same strategy logic, same inputs, same cost specs → must produce same number or your engine is broken.

---

## 4. ML Pipeline — LightGBM + ONNX, Skip the Neural Net Detour

**Decision: LightGBM for all tabular models. No TabNet, no feedforward MLPs, no LSTMs.**

2026 benchmarks are clear: tree ensembles beat neural nets on tabular financial data in both median performance and variance. TabNet and similar deep tabular models "show wide distributions with negative medians" and "rarely achieve top ranks." Neural net training is 10–100× slower with no consistent edge. LightGBM also ONNX-exports cleanly — we already have the inference harness from GBB.

**Pipeline architecture:**

- **Feature engineering as a library**, not a notebook. Every feature is a pure function `(history) → float` with a unit test showing no lookahead (feature at time T computed only from data T-1 and earlier).
- **Feature registry** in `features.yaml`: name, function path, lookback window, description. Adding a feature = PR to the registry.
- **Training pipeline:** single command `ebb train --config cfg.yaml --output model.onnx`. Deterministic seeds. Produces ONNX + feature list + hash.
- **Model versioning:** every model ships as `{name}_{git_sha}_{data_hash}.onnx`. EA loads by hash; mismatch = refuse to start.
- **Retraining cadence:** quarterly auto-retrain on rolling 5-year window, null-test + WF-gated before promotion to production.
- **Meta-filter only:** ML is a veto gate (keep GBB's pattern), never the primary entry signal. Lessons from GBB: hand-crafted signals had zero edge; ML only works as a noise-suppressor on a real edge.

**Concrete feature set for EURUSD M5** (starter, subject to ablation): realized vol 1h/4h, hour-of-day sine/cosine, distance-to-Asian-range-high/low, M5 body/wick ratios, USD index correlation, ECB/Fed event proximity, spread z-score.

---

## 5. Multi-Instrument From Day 1

**Decision: EA is symbol-agnostic from source line 1. Every strategy must pass multi-instrument null-test as part of CI.**

GoldBigBrain tested XAU only for 5 months then discovered edge was XAU-microstructure-specific — a discovery you want to make in week 2, not month 5.

**Implementation:**

- No hardcoded tick sizes, contract sizes, or pip values. Read all from `SymbolInfoDouble(_Symbol, SYMBOL_*)` at init.
- All features normalized by ATR or realized vol, not absolute price.
- All SL/TP in ATR multiples, not points.
- Config schema has `primary_symbol` + `test_symbols[]`. CI runs strategy on all of them.
- **Gate:** strategy candidate PF on primary symbol must be within 40% of PF on at least one secondary, else flagged "instrument-specific edge" and demoted.
- **Data:** M1 tick history for EURUSD, GBPUSD, AUDUSD, USDJPY, USDCAD sourced + calibrated in week 1. Single DuckDB warehouse, queryable by symbol/date range.

This also hedges the scenario where EURUSD turns out to have no 5-min edge in 2026 — we pivot instrument without rebuilding infra.

---

## 6. Position Sizing — Fractional Kelly-Capped Vol-Target

**Decision: volatility-targeted fractional Kelly, capped at 0.5% risk-per-trade.**

Why not plain 0.6% fixed fractional like GBB? GBB's 0.6% was arbitrary. It happened to work. A Kelly-based approach is not arbitrary — it's tied to measured edge.

**Formula:**

```
base_risk = 0.5%  # hard cap
kelly_fraction = max(0, (win_rate * avg_win - loss_rate * avg_loss) / avg_win) / 4   # quarter-Kelly
vol_scalar = target_vol / realized_vol_20d     # clipped to [0.5, 2.0]
position_risk = min(base_risk, kelly_fraction * base_risk) * vol_scalar
```

- Quarter-Kelly because full-Kelly assumes known edge; we have an estimate with large CI.
- Hard cap 0.5% means worst case (bad edge estimate) can't blow up.
- Vol scalar keeps $-risk-per-trade relatively constant across regimes — avoids the GBB surprise where fixed-lot sim hid compounding risk.
- Measured on rolling 500-trade window; updated every Sunday.
- **Sanity:** if realized vol measurement fails, fall back to 0.3% fixed. Never scale up on error.

---

## 7. Drawdown Brake — Dual Layer

**In-EA:** KillSwitch.mqh enforces:

- Per-trade SL: always set, hard max 1.5% risk even if sizing calc says more.
- Daily loss cap: 2% of start-of-day equity → close all, pause until next session.
- Account DD cap: 8% from ATH → halt EA, flag for user review, require manual restart.
- Consecutive-loss brake: 5 losing trades in a row → pause 4 hours.

**In validation pipeline:**

- MC-DD harness auto-fails strategies with p99 DD > 20%.
- Walk-forward auto-fails if any year has DD > 25%.
- Live-paper gate: 30 days paper trading with p95 realized DD < 10% before real money.

Both layers must exist from day 1 because: (a) EA kill switch requires code-level implementation that's painful to retrofit; (b) validation DD gates need to be fair across all candidates, not just the first winner.

---

## 8. Repository Structure + Config Management

```
EuroBigBrain/
  mql5/
    EBB_Core.mq5
    EBB/  (include files)
  python/
    ebb/
      features/       # feature functions + tests
      signals/        # Python mirror of EA signal logic
      tickreplay/     # sim engine
      validation/     # null, bootstrap, wf, mc_dd, multi_inst harnesses
      training/       # ML training pipeline
      cli.py
    tests/            # unit tests — CI required
  configs/
    strategies/       # one YAML per strategy: full params, which signals/features
    models/           # trained ONNX + metadata
  data/
    ticks/            # DuckDB warehouse, partitioned by symbol/year
  ci/
    determinism.yml
    gates.yml
  results/            # per-run artifacts, git-lfs tracked
  findings/           # WG reports, analysis docs
  docs/
    journey.md        # append-only, updated at every phase milestone
    lessons.md        # stable decisions record
```

**Config management:** one YAML per strategy candidate, schema-validated. Config includes: signal name, signal params, risk params, session gating, symbol list, ML model hash. Nothing else goes in EA inputs at runtime — config is source of truth.

**Branching:** `main` = deploy. Feature branches per candidate. PR opens → CI runs all 6 gates → green = mergeable. No direct commits to main. This enforces the "gate before merge" discipline that GBB lacked.

---

## 9. Documentation Discipline

Three docs, clearly separated:

- **`docs/journey.md`** (append-only): weekly progress, what was tried, what failed, why. Written at end of each week, never edited retroactively. Prevents the GBB "retroactive rewrite" pain.
- **`docs/lessons.md`**: stable decisions and hard rules. When a new lesson is locked, append + commit. Never deletes, only strikes out with rationale.
- **`docs/api.md`**: signal interface, config schema, feature contract. Auto-generated from code docstrings.

Per-experiment logs go in `results/YYYYMMDD_experiment-name/` with config, output, verdict, next action. CI refuses to merge if a `results/` dir has no `verdict.md`.

**What to publish externally:** nothing until a deploy-candidate passes live-paper. No premature "we found an edge" posts.

---

## 10. Time-to-First-Deployment-Candidate Target: 6 weeks

Realistic because: infra copy-able from GBB (maybe 30% reusable), validation gates are well-defined now, we know XAU-only intraday-microstructure traps to avoid, LightGBM pipeline is mostly a re-point.

Risk factors: EURUSD may not have a 5-min edge as clean as XAU's fade. Budget for week 4 pivot.

---

## BUILD ORDER (Week 1–6)

**Week 1 — Infra skeleton:**
- Repo layout, CI hooks, config schema, tick data warehouse (EUR/USD, GBP/USD, AUD/USD, USD/JPY M1 ticks 2020–2026).
- EBB_Core.mq5 skeleton: event loop, RiskCore, KillSwitch, PositionMgr (no signals yet).
- Unit tests for kill switch (synthetic 2% loss scenario).
- Worker-race test: 4 prefixes × 100 configs → verify zero collisions.
- Tick-replay engine carried over from GBB Phase 2A, patched per-day level-crossed guard.

**Week 2 — Validation harness:**
- Null, bootstrap, walk-forward, MC-DD, multi-instrument, determinism — all six as CLI commands and CI jobs.
- Calibration gate: 100 random configs on tick-replay vs MT5 every-tick. Must pass ±5% PF before proceeding.
- Determinism CI: double-run reference config, byte-diff CSVs.
- Document all gates in `docs/lessons.md`.

**Week 3 — Baseline strategy (fade_long heir):**
- Port cfg74 logic to EBB/Signals/Fade.mqh + Python mirror.
- Run on EURUSD with minimal param tuning. Submit to CI.
- Expected outcome: likely fails multi-instrument gate or null gate — this is the real test of our infrastructure, not the strategy.
- If baseline passes, label `candidate_v0`. If not, document failure mode, move on.

**Week 4 — ML veto filter:**
- Feature library + registry on EURUSD.
- LightGBM model trained 2020–2024, tested 2025–2026.
- ONNX export, EA integration via VolFilter.mqh.
- Rerun candidate_v0 with and without filter; compare on all gates.
- **Pivot checkpoint:** if EURUSD shows no real edge (null-test ratio < 2.0 across all signals tried), swap primary to GBPUSD or DXY and continue. Do not pile more strategies onto a dead instrument.

**Week 5 — Multi-instrument validation + stress:**
- Run candidate on GBP/USD, AUD/USD, USD/JPY.
- Stress scenarios: 2020-03 COVID flash, 2022 USD/JPY intervention, 2024 yen carry unwind.
- Regime breakdown per year + per instrument.
- MC-DD p99, walk-forward 6-fold.
- Tighten risk params based on stress outputs, not in-sample fit.

**Week 6 — Paper-live + kill-switch drill:**
- Deploy to demo account, Vantage or ICMarkets.
- 5 trading days minimum of paper-live, logging every tick decision.
- Kill-switch drill: manually trigger 2% intraday loss on demo, verify halt + alert.
- Compare paper-live PnL vs tick-replay prediction for same period. Must track within ±10%.
- End-of-week decision: go/no-go for live micro-lot trading.

**Exit criteria for "deployment-candidate":**
1. All six CI gates green on main.
2. 30+ trades in paper-live tracking replay within ±10%.
3. Kill-switch drill passed.
4. Multi-instrument: PF > 1.2 on ≥ 2 of 3 secondaries, zero wipeouts.
5. User (boss) approval — we are in research mode until explicit go.

If week 6 ends without all five: document honestly in `journey.md`, extend one more week max, then re-scope. Don't let EuroBigBrain become another 6-month slog by refusing to admit when a week plan slipped.

---

**End of report.**

## Sources

- [FinRL-X: AI-Native Modular Infrastructure for Quantitative Trading (arxiv)](https://arxiv.org/html/2603.21330v1)
- [AutoQuant: Auditable Expert-System Framework for Execution-Constrained Auto-Tuning (arxiv)](https://arxiv.org/html/2512.22476)
- [Implementation Risk in Portfolio Backtesting (arxiv)](https://arxiv.org/html/2603.20319v1)
- [Tabular Models Benchmark 2026 (AIMultiple)](https://research.aimultiple.com/tabular-models/)
- [Is Boosting Still All You Need for Tabular Data? (Michael Clark, 2026)](https://m-clark.github.io/posts/2026-03-01-dl-for-tabular-foundational/)
- [TabNet vs XGBoost (MLJAR)](https://mljar.com/blog/tabnet-vs-xgboost/)
- [A Closer Look at Deep Learning Methods on Tabular Datasets (arxiv)](https://arxiv.org/html/2407.00956v4)
- [MT5 Risk Management Tools for Prop Traders (ForTraders)](https://www.fortraders.com/blog/mt5-risk-management-tools-for-prop-traders)
- [Programming a Kill Switch in your EA (MQL5 Forum)](https://www.mql5.com/en/forum/466722)
- [Interpretable Hypothesis-Driven Trading: Walk-Forward Validation Framework (arxiv)](https://arxiv.org/html/2512.12924v1)
