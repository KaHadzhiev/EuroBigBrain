# EuroBigBrain — 11-Gate Validation Pipeline as CI Jobs

**Author:** Pipeline architect
**Date:** 2026-04-21
**Scope:** Turn WG3's 11 validation gates into concrete CI jobs with triggers, modules, thresholds, and an orchestrator. No candidate earns "candidate" status until Gates 1–9 green; no deploy until 10 green; 11 runs post-deploy.

**Governing principle:** every gate is a BLOCKER. A failed gate does not emit a warning — it exits non-zero and prevents the PR/merge/deploy from progressing. The only exception is Gate 11, which runs in production and issues kill-switch pauses rather than CI failures.

---

## Gate inventory (triggers + modules at a glance)

| # | Gate | Trigger | Module | Wall (Mac M5) | Wall (Win) |
|---|------|---------|--------|--------------:|-----------:|
| 1 | In-sample sanity | pre-commit | `python/validation/sanity.py` | 20 s | 40 s |
| 2 | Bootstrap + DSR | pre-merge | `python/validation/bootstrap.py` + `dsr.py` | 90 s | 3 min |
| 3 | Archetype null | pre-merge | `python/validation/null_tests.py` | 8 min | 20 min |
| 4 | Walk-forward | nightly | `python/validation/walk_forward.py` | 45 min | 2 h |
| 5 | Per-year stability | pre-merge | `python/validation/yearly.py` | 60 s | 2 min |
| 6 | Cost stress | nightly | `python/validation/cost_stress.py` | 25 min | 70 min |
| 7 | Parameter robustness | nightly | `python/validation/robustness.py` | 40 min | 100 min |
| 8 | Multi-instrument | nightly | `python/validation/cross_instrument.py` | 55 min | 140 min |
| 9 | Regime sensitivity | nightly | `python/validation/regime_split.py` | 10 min | 25 min |
| 10 | Held-out 9-mo forward | manual (promote-to-deploy) | `python/validation/forward_holdout.py` | 15 min | 35 min |
| 11 | Live kill-switch | cron 15 min | `python/validation/live_monitor.py` | continuous | continuous |

**Orchestrator:** `python/validation/run_all_gates.py` — runs 1–9 sequentially (stop on first fail), emits JSON pass/fail report, writes `validation_ledger.parquet` row.

---

## Gate 1 — In-Sample Sanity

**Purpose:** reject obvious garbage before spending compute on real statistics.

- **Threshold:** PF ≥ 1.30, trades ≥ 500, max DD ≤ 15%, avg trade duration 15 min – 48 h, Sharpe (daily) ≥ 0.8.
- **Compute:** ~20 s Mac / 40 s Win (pandas pass over 7 yrs of trade log).
- **Inputs:** trade log CSV (`run_id`, `entry_time`, `exit_time`, `pnl`, `symbol`), strategy config JSON.
- **Outputs:** `sanity_<run_id>.json` with every metric and PASS/FAIL flag; prints a 6-row summary.
- **CI event:** pre-commit hook via `.pre-commit-config.yaml`. Every candidate config must pass to even get a PR.
- **Module:** `python/validation/sanity.py` — pure pandas, no broker calls. Unit-tested on known-noise (must reject) and known-edge (must pass) fixtures.

---

## Gate 2 — Bootstrap p-value + Deflated Sharpe

**Purpose:** prove the edge is not an accident of one good streak nor an artifact of the 200-trial p-hacking budget.

- **Threshold:** stationary block bootstrap p < 0.01 (10,000 resamples, block length auto via Politis/White) AND Deflated Sharpe Ratio > 0.95 given global trial-counter N from `validation_ledger.parquet`.
- **Compute:** ~90 s Mac / 3 min Win (10k resamples of ~2k trades/yr × 7 yrs is vectorised numpy).
- **Inputs:** trade-level PnL array, strategy `archetype` tag, current global `N_trials` read from ledger.
- **Outputs:** `bootstrap_<run_id>.json` with `{p_value, CI95_mean, DSR, raw_SR, N_trials}`; CSV append to ledger incrementing N.
- **CI event:** pre-merge — GitHub Actions job `validate-stat-edge` blocks the PR merge.
- **Module:** `python/validation/bootstrap.py` (stationary block bootstrap via `arch.bootstrap`) + `python/validation/dsr.py` (Bailey/López de Prado 2014 formula, skew/kurt aware) + `python/validation/ledger.py` (parquet writer + lock).

---

## Gate 3 — Archetype-Specific Null Test

**Purpose:** match the null to the archetype so we don't make the cfg74 mistake of testing the wrong hypothesis.

- **Threshold:** observed PF ≥ 1.5× the 95th-percentile PF of 500 permutations under the archetype's primary null.
  - **Reversion** → direction-flip (multiply signals × −1; real PF must dominate flipped PF distribution).
  - **Breakout/momentum** → session-shuffle (rotate hour labels).
  - **ML meta-filter** → label permutation (shuffle y, retrain, re-backtest).
  - **Mean-revert band** → price-path bootstrap (IID resample M5 returns, rebuild series).
- **Compute:** ~8 min Mac / 20 min Win (500 resamples × cheap Python re-backtest; parallel across Mac prefixes).
- **Inputs:** strategy class, `archetype` tag, full history CSV, RNG seed.
- **Outputs:** `null_<run_id>.json` with `{null_p95_PF, real_PF, ratio, archetype, n_perms}`; histogram PNG to `findings/null_<run_id>.png`.
- **CI event:** pre-merge — same job as Gate 2 (`validate-stat-edge`).
- **Module:** `python/validation/null_tests.py` with a dispatcher keyed on `strategy.archetype`.

---

## Gate 4 — Walk-Forward Optimization

**Purpose:** prove tuning on IS transfers to OOS — if not, the "optimised" parameters are noise-fit.

- **Threshold:** ≥ 8/11 OOS windows with PF ≥ 1.2, mean OOS PF ≥ 1.3, no OOS window DD > 25%, IS-to-OOS PF correlation ≥ 0.4.
- **Config:** IS=18 months, OOS=6 months, step=6 months, history 2018-01 → 2025-01 (6-month tail held out for Gate 10).
- **Compute:** ~45 min Mac / 2 h Win (11 IS/OOS pairs × 1 MT5-every-tick OOS backtest each, parallel across Mac prefixes with cap=4).
- **Inputs:** strategy config, full history, window definitions.
- **Outputs:** `walkforward_<run_id>.csv` (one row per window), heatmap PNG, summary JSON.
- **CI event:** **nightly cron** — too expensive for per-PR. Runs on merge-to-main + fresh-candidate queue. Failed Gate 4 reverts the candidate's status back to "draft".
- **Module:** `python/validation/walk_forward.py` using claim-queue across Mac prefixes 1–4.

---

## Gate 5 — Per-Year Stability

**Purpose:** kill the M113 pattern where one freak year carries the whole strategy.

- **Threshold:** ≥ 5/7 years (2018–2024) with PF ≥ 1.15, at most 1 year PF < 1.0 with **pre-registered** economic rationale (e.g., 2020 COVID, 2022 war-onset vol) filed in `preregistrations/` before the backtest runs. Worst-year DD ≤ 20%.
- **Compute:** ~60 s Mac / 2 min Win (post-hoc pandas groupby on existing 7-yr trade log).
- **Inputs:** trade log CSV with timestamps, strategy config, any pre-registration note for a borderline year.
- **Outputs:** `yearly_<run_id>.csv` (year, PF, trades, DD, rationale_file), heatmap PNG.
- **CI event:** pre-merge — bundled in `validate-stat-edge` after Gate 2 (no sense running bootstrap if yearly fails).
- **Module:** `python/validation/yearly.py`.

---

## Gate 6 — Cost Stress Test

**Purpose:** reveal strategies whose edge is spread-fitted or slippage-naive.

- **Threshold:** under combined worst case (spread × 2, 50% of stops fill 1 pip worse, commission re-added $7/lot RT, 5-pip news-spike spread for 2 bars around NFP/ECB/FOMC), PF ≥ 1.20 and DD ≤ 1.5× baseline DD.
- **Compute:** ~25 min Mac / 70 min Win (full re-run of MT5 every-tick with cost shim).
- **Inputs:** strategy config, symbol spec (real Vantage Standard STP numbers, pulled live from MT5), economic calendar.
- **Outputs:** `cost_stress_<run_id>.json` with `{baseline_PF, stressed_PF, baseline_DD, stressed_DD, ratio}`.
- **CI event:** nightly cron. Failure drops candidate status.
- **Module:** `python/validation/cost_stress.py` — shim over the MT5 sim that injects spread/slippage/commission perturbations.

---

## Gate 7 — Parameter Robustness (Edge-Cliff Detector)

**Purpose:** reject strategies that sit on a knife-edge — if PF depends on exact parameter values, it's fit to noise.

- **Threshold:** for each tunable `p*`, PF at `p* × {0.8, 0.9, 1.0, 1.1, 1.2}` must not drop > 30%. 100-sample multivariate ±10% jitter: 95th-percentile PF ≥ 1.2.
- **Compute:** ~40 min Mac / 100 min Win (dominated by jitter sweep; univariate sweeps are cheap).
- **Inputs:** strategy config (tunable parameter set declared explicitly), history.
- **Outputs:** `robustness_<run_id>.json` + univariate sweep PNGs in `findings/robustness_<cfg>/`.
- **CI event:** nightly cron.
- **Module:** `python/validation/robustness.py`.

---

## Gate 8 — Multi-Instrument Generalization (THE M113 KILLER)

**Purpose:** prove the edge isn't EUR-specific curve-fit. M113 fade_long wiped 4 non-XAU accounts at 96–99.85% DD — exactly what this gate catches.

### Instruments
**GBPUSD, AUDUSD, USDCHF** — all M5, Vantage Standard STP quotes, same 7-year history window. Deliberately excludes JPY pairs (different tick size / pip convention that would require separate normalisation and hide generalisation failures behind unit math).

### Tier definitions

| Tier | Criteria | Disposition |
|------|----------|-------------|
| **A — Structural** | PF ≥ 1.1 AND DD ≤ 20% on all 3 instruments unchanged | PASS — strategy is structural. Green to promote. |
| **B — EUR-specific with rationale** | PF ≥ 1.0 on ≥ 2/3 AND pre-registered economic rationale filed in `preregistrations/gate8_<run_id>.md` BEFORE Gate 8 runs | PASS with audit flag. |
| **C — Overfit** | Any instrument: DD > 40% OR PF < 0 | FAIL. Strategy rejected. Do not iterate nearby configs — this is the M113 signature. |

### What counts as a Tier B rationale

The rationale file must be committed to the repo with the candidate PR **and** have a commit timestamp earlier than the Gate 8 run. It must name a specific EUR-market mechanism, not a post-hoc narrative. Acceptable examples:

1. **ECB policy schedule:** strategy exploits the ECB rate-decision calendar (1st Thursday of month, 12:45 GMT press conference) which has no direct equivalent for GBP/AUD/CHF pairs.
2. **London/NY overlap micro-structure:** entry conditions keyed to 13–16 GMT liquidity peak where EUR/USD trades ~28% of daily FX volume vs GBP/USD ~9%. The claim must be backed by a BIS triennial survey citation or equivalent.
3. **EUR-funded carry mechanics:** strategy fades mean-reversion on EUR-funded carry-unwind spikes (post-2014 negative rates regime). Specific to EUR's role as funding currency; GBP/AUD/CHF have different carry profiles.
4. **Stop-hunt levels around ECB/Eurex fix:** 14:15 GMT Frankfurt fix creates liquidity voids that fade strategies exploit; no equivalent fixing moment on the other three pairs.

**Rejected as rationale** (post-hoc cope):
- "EUR/USD is more liquid" (so what — should improve all strategies)
- "Different pairs have different volatility" (trivially true, not mechanistic)
- "The algorithm happens to prefer EUR patterns" (not a mechanism)
- Any rationale first committed *after* Gate 8 runs — that is post-hoc by definition.

A Tier B pass flags the candidate for extra scrutiny at Gate 10 and requires explicit user sign-off at deploy time.

- **Compute:** ~55 min Mac / 140 min Win (3 × MT5 every-tick full-history runs, parallel across Mac prefixes 1–3, prefix 4 reserved for headroom).
- **Inputs:** strategy class, config JSON (MUST be unchanged except `symbol`), 3 instrument histories, optional rationale file.
- **Outputs:** `cross_instrument_<run_id>.json` with per-instrument `{PF, DD, trades}` and `tier` field; table summary to stdout.
- **CI event:** nightly cron. Highest-value gate — empirically this and Gate 3 do the most work.
- **Module:** `python/validation/cross_instrument.py`.

---

## Gate 9 — Regime Sensitivity

**Purpose:** prove the strategy works across macro regimes, not just "quiet 2019-like" conditions.

- **Classifiers:**
  - VIX regime: low < 15, normal 15–25, high > 25 (daily close, VIX index from CBOE).
  - DXY regime: trending (|20-day slope| > 1.5σ) vs ranging.
  - News density: high (weeks containing Tier-1 releases — NFP, CPI, FOMC, ECB) vs quiet.
  - EURUSD volatility: ATR(14) daily above/below 90-day median.
- **Threshold:** profitable (PF ≥ 1.1) in ≥ 2/3 VIX regimes; profitable in both trending AND ranging DXY; archetype-appropriate split (reversion wins ranging, must not wipe trending — inverse for breakout).
- **Compute:** ~10 min Mac / 25 min Win (regime assignment is fast; the cost is segmenting trade log and recomputing PF per bucket).
- **Inputs:** trade log, regime lookup parquet (generated once from VIX/DXY/calendar data in `data/regime_lookup.parquet`).
- **Outputs:** `regime_<run_id>.json` + heatmap (regime × PF).
- **CI event:** nightly cron.
- **Module:** `python/validation/regime_split.py` + one-time generator `python/validation/build_regime_lookup.py`.

---

## Gate 10 — Forward Held-Out Test (Deploy Gate)

**Purpose:** the final unbiased verdict. If Gates 1–9 passed but Gate 10 fails, the strategy is dead. No nearby-config retry — that creates leakage.

- **Held-out period:** 2025-07-01 → 2026-04-01 (9 months). Never touched during Gates 1–9 and never in any sweep, parameter search, or ad-hoc backtest. All data access is gated by a filesystem-level read lock enforced by `python/validation/holdout_lock.py` that prevents any process (including exploration scripts) from reading `data/holdout/*.csv` unless the caller is `forward_holdout.py` with a valid `run_id` from the ledger.
- **Threshold:** PF ≥ 1.25, DD ≤ 1.3× worst walk-forward window DD, trade-count within ±30% of extrapolated IS frequency, no single week > 25% of total PnL.
- **Compute:** ~15 min Mac / 35 min Win (9 months MT5 every-tick on one prefix).
- **Inputs:** frozen strategy config (hash-locked), held-out 9-mo history.
- **Outputs:** `holdout_<run_id>.json`, comparison to backtest expectations, single PASS/FAIL.
- **CI event:** **manual promote-to-deploy only.** Trigger: maintainer runs `python python/validation/run_all_gates.py --config <cfg> --forward-holdout`. This command:
  1. Refuses to run unless Gates 1–9 all PASS in the ledger for the exact config hash.
  2. Requires `--yes-i-read-the-hold-out-rules` flag confirming awareness that post-failure iteration on adjacent configs contaminates the held-out set.
  3. Increments a one-use counter in the ledger: each config hash can only trigger Gate 10 **once**. Subsequent runs with the same hash return the cached result.
- **Contamination prevention:**
  - Pre-commit hook `python/validation/hooks/check_no_holdout_reads.py` greps staged Python for any reference to `data/holdout/` outside `forward_holdout.py`.
  - CI runs a daily integrity check on `data/holdout/*.csv` SHA-256 vs locked baseline; any change alerts and blocks Gate 10 until resolved.
  - Ledger records the config hash for each Gate 10 run; re-runs on same hash short-circuit; different hashes must derive from an independent research path, not "we tweaked it after the fail".
- **Module:** `python/validation/forward_holdout.py`.

---

## Gate 11 — Live Kill-Switch

**Purpose:** production-side safety net. Not a CI gate; runs forever post-deploy.

- **Signals (any triggers auto-pause):**
  - Rolling 60-day PF < 1.00
  - Single-day DD > 5% of equity
  - Rolling 30-day DD > 1.5× backtest worst
  - 3 consecutive weeks zero trades (alert-only, regime shift hint)
  - Trade frequency < 50% of backtest expectation on 20-day rolling window (alert — execution or regime)
  - Spread cost as % of PnL > 2× backtest (alert — broker conditions)
- **Compute:** trivial (MT5 account history pull + pandas every 15 min).
- **Inputs:** live MT5 account trade history, baseline backtest statistics JSON from deploy-time.
- **Outputs:** `kill_log.parquet` append + PushNotification to user phone; requires root-cause text entry before any manual un-pause.
- **CI event:** cron every 15 min on the deploy VPS. Never runs in dev CI.
- **Module:** `python/validation/live_monitor.py`.

---

## Orchestrator: `python/validation/run_all_gates.py`

**Contract:**
```
python python/validation/run_all_gates.py \
    --config configs/candidates/<run_id>.json \
    --trade-log runs/<run_id>/trades.parquet \
    [--forward-holdout]   # Gate 10 only
    [--skip-nightly]      # CI mode: only 1,2,3,5
```

**Behaviour:**
1. Parse config, compute config SHA-256 → `run_id`.
2. Load `validation_ledger.parquet`; refuse to re-run completed gates for an identical hash (idempotent).
3. Run Gate 1. Stop on fail.
4. Run Gate 2 (+ DSR, incrementing global trial counter).
5. Run Gates 3, 5 (pre-merge bundle). These are cheap enough to always run in the pre-merge CI job.
6. If `--skip-nightly` omitted: run Gates 4, 6, 7, 8, 9 in dependency-free parallel across Mac prefixes 1–4 with claim-queue (`feedback_claim_queue_pattern`). Auditor agent (separate model) reviews summaries.
7. If `--forward-holdout`: verify 1–9 green for this exact hash, prompt for confirmation flag, invoke Gate 10.
8. Write row to `validation_ledger.parquet`: `{run_id, config_hash, gate_results[], timestamp, N_trials_counter}`.
9. Print coloured summary: green/red per gate, final PASS/FAIL, any TIER B flags.
10. Exit code 0 on full pass, non-zero (gate number) on first fail.

**Hard rules wired into the orchestrator:**
- `workers=1` for any MT5 call (wine-prefix log-race memory).
- `caffeinate -i -s` prepended to all Mac calls (feedback_caffeinate_mac_launches).
- RAM check before launch: refuses nightly bundle if Mac free RAM < 6 GB or swap > 8 GB.
- All Mac prefixes sync'd first via `sync_manifest.json` (feedback_pipeline_auto_sync).
- Auditor agent spawned in parallel on every multi-gate run (feedback_aggressive_agent_use_with_auditor).
- Every gate's output row auto-committed to `claude-memory` + `EuroBigBrain` repos at batch end (feedback_constant_github_push).

---

## CI wiring summary

| Event | Job | Gates run | Blocking |
|-------|-----|-----------|----------|
| pre-commit local hook | `sanity-local` | 1 | Yes (local commit blocked) |
| PR opened/updated | GitHub Actions `validate-stat-edge` | 1, 2, 3, 5 | Yes (merge blocked) |
| merge to `main` | GitHub Actions → triggers nightly queue entry | queues 4, 6, 7, 8, 9 | No (async) |
| nightly 02:00 cron | GitHub-Actions-self-hosted-on-Win + Mac runners | 4, 6, 7, 8, 9 | Reverts candidate status on fail |
| manual `promote-to-deploy` | CLI-gated, requires `--yes` flag | 10 | Yes (deploy blocked) |
| post-deploy 15-min cron | VPS daemon | 11 | Pauses live account on fail |

Daily health check runs on `data/holdout/*.csv` SHA-256 integrity.

---

## The acceptance test for the pipeline itself

Per WG3: before any real EuroBigBrain candidate enters the pipeline, the infrastructure must:
1. PASS the known-good synthetic strategy (trend-follower on a trending synthetic series) on Gates 1–9 (skip 10, no held-out for synthetic).
2. REJECT the known-bad strategy (coin-flip signals on real EUR/USD) no later than Gate 2 or Gate 3.

Both fixtures live at `tests/fixtures/synth_trend.py` and `tests/fixtures/coinflip_eurusd.py`. CI runs them on every push to the `validation/` package and refuses to merge if either acceptance test regresses.

---

## Summary

Eleven gates, each a hard blocker. Five cheap ones run pre-merge (1, 2, 3, 5, plus Gate 1 as pre-commit). Five expensive ones run nightly on the shared Mac+Win fleet (4, 6, 7, 8, 9). One gate (10) is the manual deploy key. One gate (11) is the live-production safety net. Ledger tracks every run to enforce the 200-trial p-hacking budget project-wide. Gate 8's Tier B rationale must be pre-registered, not post-hoc. Gate 10's held-out data is filesystem-locked against accidental reads. If this pipeline had existed one month earlier, cfg74's false-negative reversal and M113's catastrophic cross-instrument wipe would both have been detected on the first night of nightly CI.
