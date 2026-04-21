# EuroBigBrain — Master Plan

**Date:** 2026-04-21
**Author:** EBB Auditor (synthesis of WG1–WG4 + LESSONS_FROM_GOLDBIGBRAIN)
**Status:** Project bible. Append-only. Amendments require user sign-off + `lessons.md` entry.
**Primary instrument:** EUR/USD M5 on Vantage Standard STP, $1k demo
**Inherited asset:** GBB_Core architecture, tick-replay engine (Phase 2A, needs patch), ONNX inference harness, 80 memory files of anti-patterns

---

## 1. Mission & Success Criteria

**Mission:** ship a deploy-grade EUR/USD M5 portfolio strategy that is **demonstrably better than London Beasts (gold) on risk-adjusted terms**, with validation so rigorous that cfg74-class reversal-confusion and M113-class XAU-overfit cannot happen again.

**"Better than gold" decomposed into concrete numbers:**

| Axis | London Beasts benchmark | EBB target | Why |
|---|---|---|---|
| Risk-adjusted return (RF = Net / MaxDD) | 15.9 (On a Leash) | **≥ 18** | Risk-adj is what scales safely; raw PnL is vanity |
| Profit Factor (multi-year MT5 every-tick) | 1.76 single strat / 1.87 & Co. portfolio | **≥ 1.45 portfolio, ≥ 1.30 per leg** | WG2 portfolio thesis: 2–3 uncorrelated PF=1.35 beats one PF=1.7 at DD |
| Max DD | 13.9% (On a Leash) | **≤ 12%** | Sub-friendly bar, leaves headroom on a $1k account |
| Trades/month (combined portfolio) | ~25 | **≥ 20** | MQL5 signal-store threshold; enough to be copied profitably |
| Years profitable (per-year OOS table) | 7/7 (cfg74) | **≥ 6/7** with no year PF < 1.0 (Gate 5, WG3) | Regime robustness, not bull-run artifact |
| Multi-instrument PF (GBPUSD, AUDUSD, USDCHF unchanged params) | N/A (gold was single-instrument) | **PF ≥ 1.10 on 2/3, zero wipeouts (DD > 40%) anywhere** | WG3 Gate 8 — the M113 killer |
| Deflated Sharpe Ratio (Bailey/LdP, trial-adjusted) | not computed | **DSR ≥ 0.95** | The gatekeeper against p-hacking across the whole project |
| Held-out 9-month forward (2025-07 → 2026-04, zero-touch) | N/A | **PF ≥ 1.25, DD ≤ 1.3× WF worst** (Gate 10) | The only test that's honest about selection bias |
| Sub-friendliness (PF-to-DD ratio, monthly positive rate) | decent | **≥ 70% profitable months, no >3 losing months in a row** | MQL5 signal-store viability |

**Non-goals:** beating On a Leash on raw PnL, competing with HFT microstructure alpha, trading overnight, trading 24h, predicting direction with ML.

---

## 2. Where WG and LESSONS Converge (non-negotiable)

All five sources independently arrived at these. They are the project's load-bearing axioms. Violating any of them triggers a hard-stop review.

| # | Converged rule | Sources |
|---|---|---|
| C1 | **ML predicts vol (WHEN), not direction (WHICH WAY).** LightGBM veto-filter, not primary signal. | WG4 §4, LESSONS §3.1, project_goldbigbrain_direction_model.md |
| C2 | **Archetype-matched nulls.** Reversion → direction-flip + bootstrap; breakout → session-shuffle; ML → label-perm. The cfg74 reversal proved session-shuffle tests the wrong hypothesis for fades. | WG3 Gate 3, LESSONS §1.7, feedback_null_test_before_mt5.md |
| C3 | **Pipeline BEFORE strategy.** Validation gates must exist, be unit-tested, and be CI-enforced before the first strategy candidate is written. | WG3 §11, WG4 §3, feedback_fix_verify_before_grid.md |
| C4 | **Multi-year MT5 every-tick is the only ground truth.** Python sims inflate 1.4–1.8× even post-rebuild. Single-period "validations" are lies. | WG3 Gate 10, WG4 §2, LESSONS §1.1, §1.2, feedback_multiyear_validation_default.md |
| C5 | **Multi-instrument generalization from line 1.** Every component takes `symbol` as a parameter; cross-instrument null is a CI gate. M113-killer. | WG1 §4, WG3 Gate 8, WG4 §5, LESSONS §1.8, project_m113_xauusd_overfit.md |
| C6 | **Don't port gold parameters.** Port architecture (ATR-scaled bracket, ONNX gate, bracket lifecycle). Re-derive numbers from EUR/USD microstructure. | WG1 §1, WG2 TL;DR, WG4 §1, LESSONS §3.4, §6.8 |
| C7 | **Kill-switch in the core, non-overridable.** Daily loss cap, account DD cap, heartbeat watchdog, spread guard — RiskCore called by Core, not by signals. | WG4 §1, §7, LESSONS §4.8 |
| C8 | **Session-gated, macro-aware, news-blacked-out.** Random-hour mean reversion on EUR/USD is pre-priced away; surviving edges are session-boundary or news-anchored. | WG1 §3, §4, WG2 archetypes 1–3, LESSONS §6.2 |
| C9 | **Workers=1 for deploy-critical MT5 grids.** Workers>1 wine-prefix race corrupts CSVs. | WG4 §2, LESSONS §1.9, §4.2, feedback_workers3_race_corrupts.md |
| C10 | **Research mode — no live deploy until user explicitly greenlights.** Every winner is "current best candidate." | LESSONS §2.7, §5.7, feedback_no_deploy_research_mode.md |

These ten are baked into CI as merge gates (see §6).

---

## 3. Where WG and LESSONS Disagree — Tiebreakers

| Conflict | Position A | Position B | Resolution | Rationale |
|---|---|---|---|---|
| **Portfolio composition** | WG2: 2–3 uncorrelated PF=1.35 legs (A=Asian breakout 30%, B=NY fade 40%, C=ML meta) | LESSONS §2.1: candidate ≠ winner, each must independently pass multi-year MT5 every-tick | **Build 2–3 legs, but each must pass Gates 1–10 INDEPENDENTLY before joining portfolio.** Portfolio layer is a capital-allocation decision on top of individually-validated legs, not a way to rescue weak ones. | Portfolio math only works if each leg is truly positive-expectancy; WG2's premise assumes this, LESSONS enforces it |
| **Top-archetype ordering** | WG1 #1 Asian Range Second-Break, #2 NY Momentum, #3 News Drift Fade | WG2 #1 NY Reversal Fade, #2 ML Direction, #3 Asian Range Breakout | **Week 3 = NY Reversal Fade (WG2 #1). Week 5 = Asian Range Breakout. ML is a META-FILTER, not a standalone candidate.** | WG2 has stronger 2024–2026 academic support (Krohn 2024 fixing flows); WG1's "momentum" is similar mechanism but LESSONS §3.6 shows TP=2 small-move capture generalizes better than directional continuation. ML-as-direction was rejected in gold at AUC 0.51 (LESSONS §3.1) — make it veto-only |
| **Null-test ratio bar** | WG3 Gate 3: observed PF ≥ 1.5× null 95th percentile | LESSONS §2.2 / feedback_null_test_before_mt5: real ≥ 5× null | **Archetype-dependent.** Direction-flip null (reversion): 1.5× is correct (LESSONS §1.7 post-cfg74-reversal). Session-shuffle for breakouts: 2.0× minimum. Label-perm for ML: 1.5×. The "5×" in old memory was a blanket over-correction. | cfg74 passed bootstrap p<0.0001 while "failing" 5× session-shuffle; the 5× rule was wrong, not the strategy |
| **P-hacking budget** | WG3: 200 backtests total before DSR demands PF>1.45 raw | LESSONS: no explicit budget, "saturate hardware 100%" | **200-backtest budget enforced via `validation_ledger.parquet`, DSR recomputed at every new row.** Hardware saturation applies to multi-config sweeps within a single pre-registered hypothesis, NOT to unlimited hypothesis generation. | Saturating cores on unlimited hypotheses = industrial-scale p-hacking. The budget forces theory-before-compute |
| **ML model choice** | WG4: LightGBM, skip TabNet/MLP/LSTM (trees win on tabular, ONNX-clean) | LESSONS §3.1: LightGBM confirmed for vol prediction (AUC 0.67 in gold) | **Agree — LightGBM only.** No disagreement, just consolidation. | No conflict; recording as settled |
| **Tick data provenance** | WG4: DuckDB warehouse of M1 ticks from broker | LESSONS §6.6: "Backtest must use broker's actual weekly data, not Dukascopy's inferred weekend" | **Broker M1 ticks as primary, Dukascopy as cross-check only.** Weekly gaps must come from broker history. | Sunday-gap behavior is broker-specific, can't infer |
| **Kill-switch DD cap** | WG4: 8% account DD | LESSONS target: ≤ 12% max DD | **8% auto-halt, 12% "do not design to exceed".** The 8% kill is a circuit breaker, not a design target. If a strategy regularly approaches 8% in sim, it's already too aggressive. | Brakes and design limits are different numbers |
| **Session window gold port** | WG1: "session-gated + macro-aware from day one" | LESSONS §6.2: "do session-shuffle test early to see if a session label even matters" | **First candidate uses 13:00–17:00 UTC (NY-fix window per WG2 Krohn rationale). Run session-shuffle as DIAGNOSTIC in week 3 to confirm window is meaningful for THIS strategy, not copy gold's 13–20 GMT.** | Session matters generally (WG1 correct), but the specific window is strategy-dependent — verify per-candidate, don't import |

**No unresolvable conflicts found.** Every tension has a defensible middle path.

---

## 4. The 6-Week Build Plan

**Overall philosophy:** Weeks 1–2 are pure plumbing — no strategy code. Week 3 is the first candidate, which tests the plumbing as much as the strategy. Each week has an explicit go/no-go gate; failing a gate triggers a scope decision, not a deadline slip.

### Week 1 — Infrastructure skeleton

**Deliverables:**
- `EBB_Core.mq5` skeleton: event loop, `RiskCore.mqh`, `KillSwitch.mqh`, `PositionMgr.mqh`, `Session.mqh`, `VolFilter.mqh` stub, `Telemetry.mqh` — no signal logic yet
- Symbol-agnostic scaffolding: `SymbolInfoDouble(_Symbol, SYMBOL_*)` at init, ATR-normalized everything
- DuckDB tick warehouse, M1 ticks 2018–2026 for EURUSD/GBPUSD/AUDUSD/USDCHF (WG3 cross-instrument set) sourced from broker + Dukascopy cross-check
- `validation/bootstrap.py`, `validation/dsr.py`, `validation/null_tests.py` with archetype dispatcher, `validation/ledger.py`
- Unit tests: kill-switch synthetic 2% loss scenario, known-noise series rejection test, worker-race test (4 prefixes × 100 configs, verify zero UUID collisions)
- Tick-replay engine: carry over from GBB Phase 2A, **patch the missing per-day level-crossed guard** (the root cause of the 0.195 calibration catastrophe per `project_phase2a_tick_replay_done.md`)

**Go/no-go gate:** CI runs on empty repo with known-noise signal → pipeline correctly rejects at Gate 2. Worker-race pre-flight shows 4 distinct UUIDs.

### Week 2 — Validation harness + calibration

**Deliverables:**
- `validation/walk_forward.py`, `validation/yearly.py`, `validation/cost_stress.py`, `validation/cross_instrument.py`, `validation/regime_split.py`, `validation/robustness.py`
- `live_monitor.py` kill-switch daemon (stubbed; no live account yet)
- **Calibration gate: 100 random configs on tick-replay vs MT5 every-tick. `|sim_PF − mt5_PF| / mt5_PF ≤ 0.05` on 2023–2025 before ANY sim result ranks candidates.**
- EUR-specific ONNX vol-filter training pipeline: port gold feature library (`python/goldbigbrain_features.py`), re-derive ATR-scaling constants for EUR/USD, train LightGBM on 2018–2023 EUR M5, export ONNX, validate AUC on 2024–2025
- Determinism CI: double-run reference config, byte-diff CSVs, rejected on mismatch
- GitHub Actions: all six gate jobs (null, bootstrap, walk-forward, MC-DD, multi-instrument, determinism) wired to PR checks
- Synthetic acceptance test: trend-following on trending synthetic passes Gates 1–9; coin-flip on real EURUSD rejected by Gate 2 or 3

**Go/no-go gate:** Calibration within ±5%. EUR ONNX vol-filter AUC ≥ 0.60 on 2024–2025 OOS (gold hit 0.67; EUR realistically 0.58–0.63). If AUC < 0.55 → vol filter is noise, proceed without it (candidate must clear gates mechanically) or pivot to GBPUSD per WG4 Week 4 pivot clause.

### Week 3 — First candidate: NY Reversal Fade

**Deliverables:**
- `EBB/Signals/NYFade.mqh` + Python mirror in `python/ebb/signals/ny_fade.py` (shared config JSON)
- Logic per WG2 #2: measure NY-open drift 13:30–14:30 UTC; if `|drift| > 1.5 × ATR(M5,20)`, arm fade for 15:00–17:00 UTC window; entry on first M5 close back inside range counter to drift; SL beyond 14:30 extreme; TP to anchored VWAP from 13:30; hard time-stop 17:00 UTC
- News blackout 13:25–13:35, full NFP day skip (hard-coded calendar, no inference)
- Run through FULL gate sequence 1–9. Gate 10 (held-out) deferred to week 6.
- Parameter robustness sweep (Gate 7) on ATR multiplier, time-stop hour, drift threshold — ±20% per param, check for cliffs
- **Pre-register the hypothesis** in `preregistrations/NYFade_v0.md` before running any backtest (forces the Gate 3 null to be archetype-matched: reversion → direction-flip + block-bootstrap, NOT session-shuffle as primary)

**Go/no-go gate:** NYFade passes Gates 1–9 on EURUSD AND multi-instrument Gate 8 (tier A or B). If fails Gate 8 at tier C (any instrument wipeout), reject candidate — do not iterate on nearby configs (creates leakage). Pick next archetype.

### Week 4 — ML veto filter + candidate hardening

**Deliverables:**
- Wire EUR ONNX vol-filter into `VolFilter.mqh`, connect to NYFade via `p_threshold` param
- Rerun NYFade through all gates with and without filter; compare improvement on PF, DD, trade count
- If filter improves PF ≥ 15% and doesn't kill trade count below 15/mo: promote `NYFade_v1` (with filter). Else: ship `NYFade_v0` unfiltered, deprecate ONNX model
- **Pivot checkpoint (WG4 §10):** if EURUSD shows no real edge (null-test ratio < 2.0 across all signals tried OR Gate 8 tier C on the first candidate), swap primary to GBPUSD. Do not pile epicycles on a dead instrument.
- Label-permutation null on the ML filter itself (archetype C3 null from WG3 Gate 3)
- Determinism CI green, ledger now shows N≈50 backtests — DSR bar at this point is PF ≥ 1.33 raw (per WG3 table interpolation)

**Go/no-go gate:** NYFade (filtered or not) has PF ≥ 1.30, DD ≤ 15%, WF 8/11 OOS windows PF ≥ 1.2, Gate 8 tier A or B, DSR ≥ 0.95.

### Week 5 — Second candidate: Asian Range Breakout + portfolio construction

**Deliverables:**
- `EBB/Signals/AsianBreak.mqh` + Python mirror (WG2 #3 / WG1 #1): Asian session 22:00–06:59 UTC records high/low; stop-buy at high+2pip, stop-sell at low-2pip; OCO valid 07:00–10:00 UTC; SL=opposite side of range; TP=1.5R; trail after 1R; **false-break filter: candle must CLOSE beyond level**
- Minimum range filter: Asian range ≥ 25 pips (WG1 failure-mode fix)
- News blackout for any Tier-1 event in 07:00–10:00 window
- Full gate sequence 1–9 for AsianBreak
- Correlation matrix: NYFade vs AsianBreak per-day PnL. Target < 0.3 (different sessions, opposite logic)
- Portfolio construction: 40% capital NYFade, 30% AsianBreak (WG2 suggested weights; ML veto applies to both at p<0.45)
- Joint backtest 2018–2024, joint walk-forward, joint MC-DD p99

**Go/no-go gate:** AsianBreak passes Gates 1–9 independently. Portfolio PF ≥ 1.40, joint DD ≤ 12%, combined trades ≥ 20/mo. If AsianBreak fails, ship NYFade-solo to week 6 and look for leg 2 candidate in parallel.

### Week 6 — Pre-deploy hardening

**Deliverables:**
- **Gate 10 (held-out forward):** 2025-07-01 → 2026-04-01, zero-touch, frozen config. PF ≥ 1.25, DD ≤ 1.3× worst WF DD, no single week > 25% of PnL
- **Deflated Sharpe recomputed** across ALL backtests ever run in the project (read `validation_ledger.parquet`). DSR ≥ 0.95 required
- Cost stress (Gate 6): spread×2, slippage +1pip on 50% of stops, commission re-added, news-spike 5pip for 2 bars — combined worst case PF ≥ 1.20
- Regime sensitivity (Gate 9): profitable in ≥ 2 of 3 VIX regimes, both trending/ranging DXY
- Paper-live on Vantage demo, 5 trading days minimum, every tick decision logged
- Kill-switch drill: manually force 2% intraday loss on demo → verify halt + alert + no new orders
- Tracking-error check: paper-live PnL vs tick-replay prediction for same period within ±10%
- **Dual kill-switch wiring verified live:** in-EA (RiskCore) + live_monitor daemon (external) both pausing independently on triggers
- Final verdict file: `results/YYYYMMDD_ebb_v1/verdict.md` with go/no-go recommendation to user

**Go/no-go gate:** all five exit criteria met (per WG4 §10). If not: `journey.md` entry honestly, extend ONE more week max, then re-scope. Don't let EBB become another 6-month slog.

---

## 5. First Night's Immediate Work Queue — Pending User Approval

**Capacity:** Mac 4 prefixes + Win 3 instances every-tick parallel = 7 items max.
**Status:** QUEUED. NOT auto-launched. User approves in morning (Q1 below).

| # | Machine | Item | Expected runtime | What we learn |
|---|---|---|---|---|
| 1 | Mac-1 | EUR fade_long VT=0.30 sweep, sess=13-17, NY-fade config baseline, 2023–2025 | ~90 min | Does a tight vol-threshold fade trigger at all on EUR? Baseline for #2–4 |
| 2 | Mac-2 | EUR fade_long VT=0.50 (same config) | ~90 min | Trade count vs VT=0.30, validates that VT is an active parameter (not dead) |
| 3 | Mac-3 | EUR fade_long VT=0.70 | ~90 min | Higher selectivity, fewer trades — checks degradation curve |
| 4 | Mac-4 | EUR fade_long VT=1.0 (gate effectively disabled) | ~90 min | **Control:** tells us whether the gold-trained ONNX gates anything sensibly on EUR features. If VT=1.0 ≈ VT=0.30 results, the gate is noise on EUR |
| 5 | Win-1 | EUR fade_long sess=08-12 (London morning), default VT=0.50 | ~2h | Is edge present at London-open window? WG1 hypothesis #1 test |
| 6 | Win-2 | EUR fade_long sess=13-17 (NY fix window, Krohn 2024 rationale) | ~2h | WG2 #2 archetype — the Week-3 candidate's signal floor check |
| 7 | Win-3 | EUR fade_long sess=07-21 (full-day, no session gate) | ~2h | Control: does session-gating matter on EUR? If 07-21 ≈ 13-17 PF, session label is noise (LESSONS §1.7 substitutability test) |

**Total wall-clock:** ≤ 2 hours (Win longer than Mac; parallel). Cost: ~8 Mac-core-hours + ~6 Win-core-hours.

**Why this specific set, tied to lessons:**

- **We don't yet have EUR-trained ONNX** (that's Week 2 work). So #4 VT=1.0 is the **control** that tells us whether even the *gold* ONNX gates anything sensibly on EUR features. If the answer is "VT does nothing on EUR", we know the feature distribution is too different (LESSONS §6.1) — which is itself a valuable finding for Week 2's EUR-ONNX training.
- **#5–7 triangulate on whether session gating matters at all on EUR.** Gold's 13–20 GMT was window-invariant (cfg74 reversal story, LESSONS §1.7). If EUR also shows session-substitutability, we know session-gating is a diagnostic, not an edge source, and we design Week 3 accordingly.
- **We are NOT porting cfg74's VT=0.20 or SL=0.3 or TP=2.0 directly.** Those are gold-ATR-calibrated (LESSONS §3.4). We're sweeping EUR-native VT values [0.30, 0.50, 0.70, 1.0] as multipliers of **EUR M5 ATR**, not of gold's.
- **Workers=1 on all 7 jobs** (LESSONS §1.9) — no wine-prefix race, each job owns its prefix/instance.
- **No redundant parallel work** (LESSONS §2.5): Mac slice = VT sweep, Win slice = session sweep, zero overlap.
- **Every job logs to `validation_ledger.parquet`** so DSR counter increments properly from run 1.
- **Pre-flight `python python/preflight.py`** before launching any of them (LESSONS §4.14).

**What we do NOT learn tonight:** whether any of these are positive-expectancy. This is a **signal-floor probe**, not a candidate test. Candidate testing starts Week 3 after infrastructure is green.

---

## 6. Hard Rules Enforced from Line 1

**Top-10 LESSONS commandments (verbatim from `C:\Users\kahad\.claude\projects\C--Users-kahad\memory\MEMORY.md`):**

| # | Rule | Memory file |
|---|---|---|
| H1 | Multi-year MT5 every-tick or it didn't happen. Show year-by-year every time. | `feedback_multiyear_validation_default.md` |
| H2 | Never trust Python sim without a published MT5 calibration ratio. | `feedback_sim_unreliable_for_ranking.md`, `feedback_dont_patch_m1_sim.md` |
| H3 | Archetype-matched nulls. Session-shuffle is a diagnostic, not a deploy gate. | `feedback_null_test_before_mt5.md`, `project_cfg74_actually_validated_bootstrap.md` |
| H4 | ML predicts vol (WHEN), never direction (WHICH WAY). | `project_goldbigbrain_direction_model.md`, `project_goldbigbrain_ml_results.md` |
| H5 | Workers=1 for every deploy-critical MT5 grid. | `feedback_workers3_race_corrupts.md` |
| H6 | MT5 every-tick caps: Mac=4, Win=3 on 16 GB. Math N×3.3GB + 5GB OS ≤ RAM. | `feedback_mac_mt5_every_tick_worker_cap.md`, `feedback_no_ram_overflow.md` |
| H7 | No redundant parallel work. Every multi-machine launch states slices explicitly. | `feedback_no_redundant_parallel.md` |
| H8 | Never static-blacklist from retro-filter stats. Non-additive trade dependencies. | `feedback_retro_filter_trap.md` |
| H9 | Lookahead-bias audit every top feature. Rolling windows `[i-N:i+1]`, shifts positive. | `feedback_lookahead_bias.md` |
| H10 | Research mode only. Every winner is "current best candidate." | `feedback_no_deploy_research_mode.md` |

**Three EBB-specific hard rules added:**

| # | Rule | Rationale |
|---|---|---|
| E1 | **Symbol-agnostic from line 1.** Every component takes `symbol` as parameter. Multi-instrument null as CI gate on every PR. | Prevents M113-style XAU-overfit (memory: `project_m113_xauusd_overfit.md`). Discovery cost in gold was 5 months; for EBB, must be 2 weeks max |
| E2 | **Pipeline before strategy: no strategy code merges until all 6 CI gates (null, bootstrap, WF, MC-DD, multi-inst, determinism) are green on empty scaffolding.** | WG4 §3 "single biggest time saver vs GBB." Prevents gate-tuning-to-winner confirmation bias |
| E3 | **Pre-register every hypothesis in `preregistrations/`.** Before any sweep, write the archetype classification, the expected null type, and the primary failure mode. Ledger budget enforcement requires it after N=200. | WG3 p-hacking discipline + LESSONS `feedback_check_memory_first.md` (theory before compute) |

---

## 7. What We Will NOT Do

| Anti-pattern | Reference | Why forbidden |
|---|---|---|
| **Port gold parameters blindly** (VT=0.20, SL=0.3, TP=2.0, sess=13–20) | `project_m113_xauusd_overfit.md`, LESSONS §3.4, §6.8 | M113 proved the numbers don't transfer. Port architecture only |
| **Retro-filter cascade** (blacklist "bad hours" from in-sample stats) | `feedback_retro_filter_trap.md` | +$9k → -$9 in one experiment. Trades non-additive |
| **Lookahead-bias features** (centered windows, future shifts) | `feedback_lookahead_bias.md`, LESSONS §1.3 | Swing detection bug inflated AUC 0.50→0.69 on pure noise |
| **Declare "validated" from single-period test** | `project_v11_multiyear_FAIL.md` | v1.1 wiped account on full history after "passing" a 3.5mo test |
| **Deploy without explicit user greenlight** | `feedback_no_deploy_research_mode.md` | Every winner stays "candidate" |
| **Workers>1 on deploy-critical grids** | `feedback_workers3_race_corrupts.md` | Wine prefix log race stitches wrong PF to rows |
| **Feed sim survivors to MT5 before null test** | `feedback_null_test_before_mt5.md` | Hard rule; sim rankings are meaningless without null-gate |
| **Trust Python sim PF without calibration ratio** | `feedback_sim_unreliable_for_ranking.md` | Inflates 1.4–1.8× on V5-family, 13–20× on old mega_screener |
| **Fixed-lot sim DD as deploy-relevant** | LESSONS §1.12 | Understates by 3–4×. Always MT5 every-tick at target RiskPercent |
| **Same-machine duplicate runs with different `--out`** | `feedback_no_redundant_parallel.md` | 100% overlap is not speedup |
| **Retroactively tune validation gates to let a winner through** | WG4 §3 | Confirmation-bias cascade; gates must be frozen before candidates run |
| **Iterate on nearby configs after Gate 10 fail** | WG3 §10 | Creates leakage into held-out set. Dead-strategy = start over with new idea |
| **Static-blacklist hours/days/regimes from retro analysis** | `feedback_retro_filter_trap.md` | See retro-filter cascade above |
| **Trade through Tier-1 news without blackout** | WG1 §4 | 5–10 pip slippage events erase a month of edge |
| **Predict direction with ML as primary signal** | LESSONS §3.1, `project_goldbigbrain_direction_model.md` | AUC stalls at 0.51. ML is a veto-only meta-filter |
| **Commit strategy code before infrastructure is green** | WG4 §3, E2 | Gate-tuning-to-winner confirmation bias |
| **Start in deploy-mode** | `feedback_no_deploy_research_mode.md` | "You're not deployin anything" — user directive 2026-04-20 |

---

## 8. Open Questions for User (Decision-Grade, Morning Review)

**Q1: Approve the 7-item overnight queue as written?**
- Options: YES / NO / modify
- **Recommend: YES** — 2h wall-clock, zero deploy risk, measures EUR fade_long signal floor + whether session/VT parameters are active or noise. Control cells (#4 VT=1.0, #7 sess=07-21) make the experiment honest regardless of outcome.

**Q2: Train EBB-specific ONNX vol model on EUR M5 features in Week 2, or reuse gold ONNX as a control?**
- Options: A) Train EUR-native from scratch / B) Reuse gold ONNX / C) Both in parallel
- **Recommend: C (Both in parallel)** — gold ONNX is already trained and can run this week as a signal-floor probe (queue items #1–4). Week 2 trains EUR-native. Gold ONNX likely outputs noise on EUR features per LESSONS §6.1, but "likely" isn't measured. Running both gives us a quantitative answer on transferability.

**Q3: Multi-instrument scope from Week 1 — EUR/USD only, or include GBPUSD + AUDUSD + USDCHF for cross-gate from day 1?**
- Options: A) EUR-only weeks 1–4, cross-gate week 5 / B) Full 4-instrument warehouse from week 1, cross-gate on every PR
- **Recommend: B (full 4-instrument from week 1)** — catches M113-style XAU-overfit failures 3 weeks earlier. Data warehouse is one-time cost; the marginal compute on cross-instrument CI is ~15 min per PR. WG3 Gate 8 and WG4 §5 both demand this from line 1.

**Q4: First candidate's session window (Week 3 NYFade) — use WG2's 15:00–17:00 UTC (Krohn 2024 fixing-flow rationale), or run a session-shuffle diagnostic FIRST to let data pick the window?**
- Options: A) Frozen 15:00–17:00 per academic rationale / B) Data-driven from session-shuffle / C) Both, compare
- **Recommend: A (frozen 15:00–17:00)** — data-driven selection is in-sample leakage dressed up as "letting the data decide." The academic rationale is the pre-registration (E3). We run the session-shuffle **as a diagnostic after** the main backtest to check substitutability (LESSONS §1.7), not to pick the window.

**Q5: Portfolio build order — Week 5 add AsianBreak as second leg, or validate NYFade standalone through Week 6 held-out first, add leg in Week 7+?**
- Options: A) 2-leg by week 5 per WG4 timeline / B) NYFade-solo through week 6, portfolio phase is week 7+
- **Recommend: B (NYFade-solo through week 6)** — WG4's 6-week timeline is aggressive; LESSONS §2.1 demands each leg independently pass ALL gates including Gate 10 before portfolio combination. Adding a second leg before leg 1 has cleared held-out is exactly the "rescue-weak-with-portfolio" anti-pattern we rejected in §3. Accept the timeline slip.

**Q6: ONNX vol-filter policy if Week 2 EUR-ONNX AUC < 0.55 OOS — proceed without filter, or pivot instrument to GBPUSD (WG4 week-4 clause invoked early)?**
- Options: A) Proceed without filter / B) Pivot to GBPUSD / C) Defer decision until Week 3 NYFade results
- **Recommend: C (defer to Week 3)** — AUC 0.55 is a feature-filter metric, not a strategy metric. NYFade without filter might still pass Gate 1–9 mechanically. Decision needs both data points. If NYFade also fails its gates on EUR → THEN pivot to GBPUSD per WG4 §10. Avoid compounding pivots on a single weak signal.

**Q7: Deflated Sharpe budget — lock at 200 backtests as WG3 proposes, or parameterize based on hardware capacity (we can do 2000 easily)?**
- Options: A) 200 hard cap / B) 2000 hard cap / C) No cap, DSR math does the work
- **Recommend: A (200 hard cap)** — the cap's job is **theoretical discipline**, not compute rationing. At N=2000 the DSR PF bar is 1.60+ raw; no realistic EUR M5 strategy clears that in-sample. 200 is the right ceiling for keeping us honest about hypothesis generation, per WG3's explicit rationale.

---

**End of master plan. Amendments go in `docs/lessons.md` with date + rationale.**
