# Lessons from GoldBigBrain — applied to EuroBigBrain

_Source: ~80 memory files in `C:\Users\kahad\.claude\projects\C--Users-kahad\memory\` synthesised 2026-04-21._

_Purpose: encode every painful lesson from XAUUSD research so EuroBigBrain doesn't repeat them. EUR/USD is a different animal (1-pip spread vs gold's 18pt, carry-driven vs inflation-hedge, retail-crowded) — most GoldBigBrain intuitions are NOT portable. The methodology IS._

---

## 1. Anti-patterns to NEVER repeat

### 1.1 Don't trust any Python simulator that hasn't been calibrated to MT5 every-tick

The **mega_screener / old Python sim inflated P&L by 13–20×**. Months of "validated strategies" turned out to be numerical artifacts. Even after rebuilding `mt5_sim.py` with proper state machine, bracket lifecycle, and M1 resolution:

- AR-family strategies match MT5 within ±15% PF (trustable for screening)
- V5-family (tight-SL brackets) still inflates **1.4–1.8× PF** — intra-M1 SL/TP ordering ambiguity
- `mt5_sim` at M1 OHLC median ratio vs MT5 every-tick was **0.195** on the tick-replay gate — catastrophic

**Rule for EuroBigBrain:** never lock a config from Python sim. Sim exists to rank candidates cheaply; MT5 every-tick Model=8 is the only legitimate ground truth. Calibration ratio must be published in the run report, not assumed.

### 1.2 Single-period validation is a lie. Multi-year is mandatory.

The session where this crystallised cost hours of wasted deploy-prep:

- v1.1 (AR VT=0.12 + V5 VT=0.38) was locked on a 3.5-month forward test — "PASS, deploy gate met"
- Full 2020–2026 MT5 every-tick: **AR PF=0.87, DD=92.7% (account wipe), V5 PF=1.09 (noise), combined PF=0.91 (-$396 over 75 months)**
- User response: "Havent you understand anything for the testing so far? What the actual fuck???"

**Hard rule (feedback_multiyear_validation_default):**

- Default horizon = ALL available history, not the post-sweep period
- Single-period results are **candidates**, never **winners**
- Show the year-by-year table **every** time: `2020: PF=X.XX, trades=N ... 2026YTD: PF=X.XX, trades=N`
- Deploy gate: PF ≥ 1.3 across ALL years, ≥6/6 OOS years profitable, ≥4/4 recent quarters profitable, ≥20 trades/mo combined

For EuroBigBrain this is even sharper: EUR/USD has regime shifts (ECB carry-unwinds, post-QE, 2022 parity, SNB pegs in 2015). A strategy that works 2020–2024 but not 2015 is regime-bound.

### 1.3 Lookahead bias can disguise itself as "top features"

Swing detection used `high[i-lookback:i+lookback+1]` — the `i+lookback` half looks forward. Result: `bars_since_swing_high/low` became top-MI features at 0.030/0.024 and inflated LightGBM AUC from 0.50 → 0.6943. **Completely fake signal.**

**Rule:** every rolling window is `[i-N:i+1]` (causal). Every shift is positive. When MI scan shows one feature dramatically outperforming the rest, **audit its computation before celebrating**. Targets can look forward; features cannot.

### 1.4 Retro-filter cascades destroy strategies

GoldDigger with `t1 PF=0.8 + SelfPause` made +$9,095 across 4 periods. Cross-year retro-analysis showed hours `{5, 6, 11, 12, 13, 17}` consistently losing. I added `InpHourBlacklist` — a *perfectly obvious* optimization. Result: Q1 collapsed $8,952 → $462. Total went +$9,095 → **-$9**.

**Why:** trades are non-additive. Each trade's existence depends on prior capital, adaptive state (SelfPause ring buffer, consec_losses, daily_pnl). Removing "losing" hours changes the path so winning hours never get reached.

**Rule:** never static-blacklist from retro stats. If a regime pattern is real, handle it via adaptive gates (SelfPause-style), not static filters. Any static filter must be validated on the full test matrix vs no-filter baseline.

### 1.5 SelfPause masks broken signals — it does not create edge

A strategy that shows "+$X with SelfPause, -$Y without" has NO edge. SelfPause is a circuit breaker, not an edge generator. It reduces trades to 20–30 over 4 years and counts the lucky survivors as signal.

**Rule:** test every strategy at Risk=1.0, SelfPause OFF, first. Only add SelfPause as risk overlay AFTER confirming positive expectancy in the naked strategy.

### 1.6 High PF at low trade count is a small-sample artifact

PF=2.06 on 25 trades means very little; so do grid-sweep winners hand-picked from 40,000 configs. After 84-config MT5 sweeps: `SL=0.3` universally dominated — but the `cfg74 PF=1.41 over 1,674 trades / 6 years` is the number that matters, not the sweep peak.

**Rule:** ≥200 trades before any PF claim is trustworthy. Portfolio-level: ≥20 trades/mo combined (MQL5 signal-store gate).

### 1.7 "Session-shuffle null" tested specificity, not validity (watch the null test setup)

For 4 days we ran permutation null tests that shuffled session-window labels and demanded `real / null_median ≥ 5×`. cfg74, beast, fg, rsisl all "FAILED" at 0.9–1.2×. We declared "all 4 are noise." **Wrong question.**

Session-shuffle asks "is THIS 7h window magical?" (it's not — any 7h window works equally). It does **not** ask "is the strategy positive-expectancy?" cfg74 later passed bootstrap p<0.0001, 7/7 years profitable, direction-flip significant — **real edge, window-invariant.**

**Rule for EuroBigBrain null testing:**
1. Bootstrap mean PnL > 0 at p<0.01 (required — is the strategy actually positive-expectancy?)
2. Year-by-year positive (required — catches regime overfit)
3. Direction-flip PF > flip-null p95 (required — confirms **directional** edge, rules out magnitude-only artifacts)
4. Session-shuffle null is a **substitutability test**, not a validity test. Keep it; don't interpret it as a deploy gate.

### 1.8 Instrument-specific tuning is overfitting dressed up as "calibration"

M113 fade_long with VT=0.10 (calibrated to gold ATR scale) on EURUSD: **zero trades** — ATR thresholds never triggered. On XAGUSD/GBPUSD/USDJPY: all accounts wiped at 96–99.85% DD.

Calibrated retest with per-instrument VTs (0.0008 for EURUSD, 0.0010 for GBPUSD): **4,097 trades, every account wiped, PF 0.12–0.95.** XAUUSD's "mean-reversion edge" was pure gold microstructure. No generalisation.

**Rule for EuroBigBrain:** do not start from `gold params × EURUSD data`. Start from EUR/USD microstructure (carry, London/NY sessions, ECB calendar). Gold's ONNX vol-filter features (wavelet energy, Hurst, sample entropy) may apply, but parameters, session hours, SL/TP multipliers all need fresh derivation.

### 1.9 `workers>1` on MT5 grids corrupts CSVs (wine prefix log race)

AR grid at workers=3: VT=0.10/SL=0.8/TP=5.0 reported PF=1.44, 990 trades. Clean re-run at workers=1: PF=0.90, 2,133 trades. 2× trade-count delta, sign-flip on PF. The Mac wine runner sometimes stitches one config's metrics onto another config's CSV row.

**Rule:** `workers=1` for any deploy-critical grid. Workers>1 only for noise-tolerant exploration. Before locking any config from a multi-worker sweep, re-run single-worker — the delta is your honesty check.

### 1.10 Gold spread was wrong by 2× in every prior conversation

Round table assumed XAUUSD spread = 30–40pt. MT5 actual = **18pt**. The sim used $0.35/oz, actual cost was $0.20/oz — **43% too high**. Every deploy-gate calculation was overstating cost.

**Rule:** never quote spread/commission from memory or agent estimates. Query MT5 / broker specs directly. For EUR/USD on Vantage Standard STP, derive from the live `SymbolInfoInteger(SYMBOL_SPREAD)` during London hours, not from a round-table claim.

### 1.11 `_chart_lib.load_trades` post-hoc Python compounding on MT5 trade logs

Old chart lib had `compound_to_balance=1000, max_lot_multiplier=10` which applied Python compounding on top of trade logs that MT5 had **already** compounded. & Co. showed $96k when real was $102k. Silent double-count.

**Rule:** any chart/analysis helper that reads MT5 output must use the `profit` column as-is. No post-hoc rescaling. Add assertion-tests on known trade counts.

### 1.12 Fixed-lot Python sim understates live DD by 3–4×

rsisl's Mac 182-config sim said "4% DD, +$7,135." Ran at fixed 0.01 lot, no compounding. MT5 with RiskPercent=0.6% compounds naturally (lot 0.06 at $1k → 1.0 cap at $25k). Losing streaks in the $5–15k balance phase eat $800–2,000 in real dollars. Real DD: **13.9%**. Same trades, same dates — real sizing.

**Rule:** never quote fixed-lot Python DD as deploy-relevant. Always MT5 every-tick at target RiskPercent.

---

## 2. Hard validation rules

### 2.1 The EuroBigBrain deploy gate (non-negotiable)

Candidate → deploy-ready requires **all** of:

1. **Multi-year MT5 every-tick Model=8 on full history** (EUR/USD: from whichever earliest year Dukascopy / broker ticks go back). Show year-by-year table.
2. **PF ≥ 1.3 per year**, positive in every year, ≥20 trades/mo combined.
3. **Bootstrap p<0.01** on per-trade PnL (10k resamples with replacement).
4. **Direction-flip null**: real PF > flip-null 95th percentile.
5. **Walk-forward retrain**: train each year, test on next; must pass in ≥5/6 OOS years.
6. **Regime check**: per-regime (bull/bear/sideways/volatile) PF and trade count broken out. If any regime has PF<1.0 on ≥100 trades, drop it from that regime's schedule.
7. **Realistic spread + slippage**: MT5 tester running at live broker spread (pull from MT5, not assumption) and ≥ 2× median spread as slippage. Post-cost PF must still clear 1.15.
8. **Calibration ratio**: if Python sim was used for screening, include sim_PF/MT5_PF ratio per config.

### 2.2 Null-test sequencing (learned the hard way)

Proper order:

1. Bootstrap mean PnL > 0 — is there ANY edge?
2. Year-by-year positive — is the edge stable across regimes?
3. Direction-flip PF > flip-null p95 — is the edge **directional** (not magnitude-only)?
4. Session-shuffle — is the session label meaningful, or substitutable? (diagnostic, not gate)
5. Only after all four pass, move to MT5 realistic-cost verification.

### 2.3 QA-before-scale (feedback_qa_before_assign)

Every new script gets **one** single-input run first, with log grep to verify it does what its name claims. The mi_edge silent XAUUSD-fallback burn cost 40 CPU-minutes and re-sync. Before parallel/multi-machine launch:

- Run one invocation foreground (or short bg with explicit log)
- Grep the log for the specific behavior (e.g. `grep "Loading from" log` to confirm symbol path)
- `md5sum` the script across machines to confirm version parity
- ONLY THEN scale to N parallel workers

### 2.4 Validate before overnight (feedback_validate_before_long_runs)

User burned three times by overnight grids returning NO_RESULT rows (SkipChoppy never fired, `grep -P` locale fail, PowerShell exit codes always 0). For every long run:

- One manual test with exact script/function the grid will use
- Verify output CSV has real PF/Trades/DD, not empty strings
- For boolean flags: run True vs False, verify results differ
- Use `grep -o '[0-9.]*'` not `grep -oP '[\d.]+'` (MSYS2 locale fail)
- PowerShell process counts: use `.Count` compared to "0", not exit codes
- Only after seeing REAL rows from the smoke test, launch full grid

### 2.5 No redundant parallel work

Same script + same args on two machines = 100% overlap. User burned 3 hours on `wall_zoo_phase2` launching defaults on both Win and Mac — both enumerated identical combos. Different `--out` != different work.

Every multi-machine launch must state: **"Win = slice X, Mac = slice Y, no overlap."** If the script lacks a slice flag, add one before launching.

### 2.6 Slot-specialist rule

A strategy that loses money **overall** can still be a slot/regime specialist. Evaluate at the (strategy, slot, regime) tuple level. AR v1.1 globally failed (PF=0.87) — but may have a 2–3h window in trend years where PF≥1.5 that is deployable.

- All sim funnel CSVs include per-(slot, year) rows, never just per-strategy aggregates
- Tuple gate: PF≥1.15 AND ≥10 trades in that tuple AND positive in ≥3/N years
- Portfolio combiner assigns each qualified tuple to its slot in the 24h schedule; multiple strategies can share a slot if their entry conditions don't conflict

### 2.7 No deploy until user explicitly greenlights

Current rule (2026-04-20 13:14): "You're not deployin anything." Frame every winner as "current best candidate", never "ready to deploy." Keep hunting.

---

## 3. Proven patterns worth reusing

### 3.1 ML vol-filter (AUC 0.67) works; ML direction (AUC 0.51) does NOT

For gold, LightGBM on 45 features predicting `max_excursion > 1.5×ATR in next 5 bars` hit AUC 0.6667, permutation p=0.000. Direction prediction stalled at 0.51. **Predicting WHEN to trade is the edge; predicting WHICH DIRECTION is wishful thinking for M5.**

Top features (after ablation): `minutes_since_session`, `wavelet_energy_L0`, `atr_ratio_14_50`, `hour`, `sample_entropy_64`, `hurst_128`. Dead weight: MACD, h4_rsi, h4_trend, bb_width, ema_trend.

**For EuroBigBrain:** build a LightGBM vol-filter → ONNX → MQL5 EA. Don't try to predict direction; predict vol spikes, let the mechanical strategy handle direction. Re-derive features from EUR/USD data — some (wavelets, entropy, hurst) should transfer; `atr_ratio` scaling needs redo.

### 3.2 Walk-forward retrain pattern (6/6 OOS years passed for gold)

| Train | Test | AUC | PF |
|---|---|---|---|
| 2020 | 2021 | 0.732 | 1.08 |
| 2020–21 | 2022 | 0.717 | 1.13 |
| 2020–22 | 2023 | 0.736 | 1.18 |
| 2020–23 | 2024 | 0.732 | 1.16 |
| 2020–24 | 2025 | 0.703 | 1.22 |
| 2020–25 | 2026 | 0.698 | 1.52 |

Expanding-window retrain, 24-bar purge gap between train and test. AUC stable, PF improving with more data. This is the gold standard — literally — for temporal validation. **Replicate it verbatim for EUR/USD.**

### 3.3 Bracket EA architecture (modular core + generic dispatch)

`GBB_Core.mq5` refactor produced identical output to `GBB_Generic` on cfg74 (PF=1.4086 to the cent). Architecture:
- State machine: idle → armed BuyStop/SellStop → filled → managed (BE + trail) → closed
- ATR-scaled entry bracket: `BracketOffset × ATR`, arm for `BracketBars` bars then cancel
- ATR-scaled SL/TP/BE/Trail multipliers
- ONNX vol gate BEFORE strategy dispatch
- Per-strategy entry module (fade_long, momentum, breakout, rsisl, etc.)

**For EuroBigBrain:** clone this architecture. Only the entry module and ONNX model file change. The risk management / order lifecycle is proven.

### 3.4 The XAUUSD "champion" configs (reference points only, don't port parameters)

- **cfg74** (fade_long): VT=0.20, SL=0.3, TP=2.0, sess=13–20, hold=12, BE=0.5, Trail=0.3, FadeLongRSI=35 → PF=1.41, 1,674 trades, +$3,914/6y, 7/7 profitable years.
- **On a Leash** (rsisl): Risk=0.6%, VT=0.20, SL=0.20, TP=2.0, RSI=40 → +6,125%, -13.9% DD, PF=1.76, 441× risk-adjusted (BEST of the 3 London Beasts).
- **Unchained** (fade_long): Risk=1.0%, VT=0.14, SL=0.3, TP=2.0, RSI=35 → +8,525%, -35.4% DD, PF=1.34.
- **& Co.** (fg+rsisl combo): combined PF=1.87, -13.9% DD, needs 2× capital.

**Do NOT port these numbers to EUR/USD.** Port the **structure** (ATR-scaled SL/TP/BE/Trail, ONNX vol gate, session window, hold cap). Every number comes from gold's specific ATR distribution.

### 3.5 Bootstrap + direction-flip validation framework

`python/cfg74_bootstrap_edge.py`: 10,000 resamples with replacement → CI95 on mean PnL → p(mean ≤ 0). Direction-flip: randomize ± sign of each PnL, recompute PF, compare real to flip-null p95. These two tests rescued cfg74 from false rejection. **Use them as the primary validity check for every EuroBigBrain candidate.**

### 3.6 TP=2 / SL≈2 is regime-agnostic; TP=5 fails in bear

26/26 quarters profitable with VT=0.15/TP=2/SL=1.5 across 2020–2026. By regime: BULL PF=1.23, BEAR PF=1.26, SIDE PF=1.22. Meanwhile TP=5 configs: BEAR PF=0.54 (fail). **Small-move capture generalises; directional continuation doesn't.** Start EUR/USD exploration with TP≈2×ATR and SL≈1.5–2×ATR.

---

## 4. Hardware + process rules

### 4.1 Machine speed hierarchy

- **Mac M5 (16 GB, 10 cores):** ~28× faster than Win for MT5 via Wine. Apple Silicon uses 16 KB pages (not 4 KB — `vm_stat × 16384`). Mac=primary cruncher.
- **Windows laptop (16 GB, 8 cores):** MT5 native, 6 instances `C:\MT5-Instances\Instance1..6`. Script: `python/mt5_fast_grid.py --workers 6`.
- **Desktop PC:** GPU for neural nets if used; otherwise supplemental CPU.

### 4.2 MT5 every-tick worker caps (16 GB boxes)

- **Mac Wine prefixes: cap 4** (every-tick Model=8 on multi-month XAUUSD M5). `N_PREFIXES=8` drove swap 394MB → 6,330MB in 4 min; SSH lockout within 2 more min. Dropped to 4 → swap back to 545MB, free recovered to 10.5GB.
- **Win instances: cap 3.** Not 4 — empirically 4 instances drove free RAM 12GB → 0.3GB in 6 min as `metatester64` peaked at 3.3 GB each.
- **Math before launch:** `per_instance_peak_RAM (~3.3 GB) × N + 5 GB OS+overhead ≤ TotalRAM`. On 16 GB: N=3.
- Short (≤1 month) runs tolerate Mac=8; M1 OHLC (not every-tick) tolerates Mac=6.

### 4.3 Claim-queue pattern for multi-worker

Never pre-bind jobs to PIDs. Workers pull from a shared filesystem queue (atomic file-move / file-lock on a `pending/` directory). Handles worker crashes cleanly; any config not consumed stays in the queue.

### 4.4 `caffeinate -i -s` on every long Mac launch

macOS sleeps on lid close or idle timeout and silently kills Wine MT5. Bare `nohup` is NOT enough — it survives shell exit but not system sleep. Every long Mac job:

```bash
ssh -i ~/.ssh/mac_m5 mac "cd ~/GoldBigBrain && caffeinate -i -s nohup python3 -u script.py > log 2>&1 < /dev/null &"
```

Add `nice -n 5` for MT5 grid runs to keep user foreground apps responsive.

### 4.5 Wine audio + Report-path fixes (Mac MT5)

- Disable Wine audio subsystem before launching terminal64: `export WINEDLLOVERRIDES="mmdevapi=d;winepulse.drv=d;winecoreaudio.drv=d"` and `export WINE_NO_AUDIO=1`. Kills the `mmdevapi_private.h:67` assertion spam.
- Use **relative** Report= path in MT5 INI: `Report=beast_mitigation_M4.htm`, NOT `Report=C:\Program Files\MetaTrader 5\...htm` (the absolute path silently fails to write the HTM report).

### 4.6 Constant RAM/swap watchdog

Every monitor tick samples RAM+swap on Win+Mac. Auto-kill worker if: swap > 2× physical, or free < 5% (Win free<0.8GB is a single-tick kill trigger). Mac swap growth >1 GB/min = kill all immediately.

### 4.7 MT5 must launch minimized/hidden

Visible MT5 steals focus and blocks user input. Use persistent minimizer + launch script. Non-negotiable on Win.

### 4.8 Kill conflicts before launch; zombie process hygiene

Before any new MT5 launch: list and kill leftover services/processes. Every monitor tick kill my orphan processes (metatester64 after orchestrator death, dangling Python). **NEVER** touch user's CMD / MT5 / browser.

### 4.9 Pipeline auto-sync to Mac

Code/config sync is bootstrap, not manual. `scp -i ~/.ssh/mac_m5`, `mkdir -p`, verify imports with `python3 -c "import X"`, log `sync_manifest.json`. Manual scp = pipeline incomplete.

### 4.10 GitHub push at every checkpoint

Memory + code + doc auto-pushed at every monitor checkpoint. Memory → `claude-memory`, code → project repo. Never let work sit uncommitted. Use the PAT stored in `reference_github_token.md`.

### 4.11 GPU + CPU run in parallel, never serial

GPUs (Mac MPS, Desktop DirectML) and CPUs are independent resources. Always run neural net training / PyTorch work concurrent with CPU Python sim / MT5 grid. The 22h→35min lesson came from discovering GPU was idle while CPU was saturated. LightGBM / XGBoost get **zero** benefit from CoreML / ANE (only NN models benefit); don't waste time ANE-exporting trees.

### 4.12 Aggressive parallel agents + mandatory auditor

Default to delegating. Launch agents in parallel (single message, multiple Agent calls). For every batch of workers, spawn at least one **auditor** agent in parallel (different model — sonnet auditing opus, or vice versa). Pass `model: "sonnet" | "haiku" | "opus"` explicitly. Never single-threaded agent execution.

### 4.13 Mac power state

Mac is off by default; ping user to wake before scheduling Mac jobs. SSH timeout = informational, not alert. Never assume Mac is online without a successful SSH handshake.

### 4.14 Preflight every grid

`python python/preflight.py` before any grid run. Mandatory. Catches: stale ONNX, missing data files, locale issues, PowerShell exit-code traps, hardcoded dates in log paths.

---

## 5. User preferences / collaboration style

### 5.1 Responses: SHORT, no TLDR

- Status / acknowledgement: 1–3 lines
- Substantive answer: 5–10 lines, hard cap 15
- **No "TL;DR" header** — if a TLDR was needed, response was too long; cut
- No "Status / Next / Goal" 3-section structure unless asked
- Tables only for 3+ items, bullets only for 4+ items, otherwise prose

### 5.2 Humor floor = 8/10 MANDATORY

User has re-tuned this **four times**. Re-read every response; if neutral, rewrite. Floor = 1 quip per response (more if situation deserves — a config blowing up, agents flailing, wine prefixes being wine prefixes, metrics refusing to behave). Dry, observational, occasionally absurd. Punch at the work, the broker's "tick history", the SIM acting like a moody intern. **Never** punch at the user. No emoji spam.

### 5.3 Explain like the user is new to the topic

Every technical term defined inline on first use and re-explained when it reappears. Use analogies. Re-explain config IDs every time (the user has 200+ tasks; expecting them to remember which 4-digit number means what is unfair). Translate metrics: "PF=1.41 — meaning $1.41 in wins for every $1 in losses." Clarity, not baby-talk.

### 5.4 Rules are HARD BLOCKERS, not guidelines

Pre-flight every action against applicable rules. Satisfy IN-BATCH, not deferred. User explicit: "Why we are making this rules if you're violating them constantly?" Before any Write / Edit / Bash-mutate / Agent spawn / background job:

1. Pause. "What rules touch this action?"
2. Name them mentally: preflight, QA-before-assign, autosync, aggressive-agent-with-auditor, multi-year, null-test, workers=1, caffeinate, memory-check-first.
3. Each applicable rule must be satisfied in the same tool-call batch OR the next immediate batch.
4. Background jobs: verify PID alive 3–5s after launch.
5. Multi-machine: state slices explicitly.

### 5.5 Full autonomy on research forks

"Choose yourself. I trust you." Don't present menus. Decide and execute research-path choices. Report what you're doing in ≤2 lines, then run. Still ask on: deploy/live-money, destructive ops, multi-hour priority shifts, genuinely tied calls (run both in parallel instead).

### 5.6 Apply fixes autonomously

Clear fix + defensible default = just apply it. "I want you to do the fixes without the need from me to give OK." Ask only on: strategic direction changes, destructive shared-system ops, multi-hour commits, truly 50/50 calls.

### 5.7 Manager role

User = boss, Claude = manager. Decompose, delegate, audit. Idle CPUs = wasted money. Max parallel agents + auditor always.

### 5.8 Check memory FIRST

Before any debug, fix, or new approach — grep `MEMORY.md`, read matching files. "Reading memory ≠ acting on memory": when memory predicts an outcome, skip the experiment that confirms it. Each skipped memory check has cost ~30 min; asymmetric ratio in our favor.

### 5.9 Web-search when unsure

Factual gaps (broker specs, economic calendar, regime classification, library API) → WebSearch/WebFetch FIRST. Order: memory → web → ask user.

### 5.10 Check task list every 15 min

Every 15 min inspect the task list for updates, new items, staleness. Report changes.

### 5.11 Distinguish `/loop` responses visually

Every `/loop`-triggered response starts with `🔄 **[loop tick]**` header. Keeps loop-auto-fires visually distinct from user messages.

### 5.12 Not a yes-man

Challenge user assumptions openly. Never parrot "you're right." The round-table spread-claim disaster (30–40pt vs actual 18pt) happened partly because no agent pushed back.

### 5.13 Saturate hardware at all times

100% Mac prefixes 1–15 + Win MT5 1–6 + both GPUs at ALL times. Audit every response. This has been stated **six times**; treat as the most-repeated rule in the session.

### 5.14 Max 10s wait on Mac

HARD CEILING. Never wait >10s on Mac. Background long ops with `caffeinate -i -s nohup`.

### 5.15 Remind pending decisions

When blocked on user input, append a `PENDING:` line every response until decided. Win sat idle 30+ min over a silent prompt — don't repeat.

---

## 6. EuroBigBrain-specific implications

These are the places where gold intuition will mislead for EUR/USD.

### 6.1 No ONNX vol model exists yet

Gold's vol model was built from 59 engineered features, LightGBM, AUC=0.67, permutation p=0.000. EUR/USD needs its own, and **the feature distribution will differ**:

- London open / NY open are different focal points (gold has strong 13–20 GMT bias; EUR/USD has 07–16 GMT London + 12–21 NY overlap)
- Carry-trade flows dominate weekly; gold's inflation/risk-off flows are event-driven
- ECB / FOMC / NFP are the event spikes; gold reacts to these too but also geopolitics
- ATR scale: EUR/USD ATR(14,M5) is ~5–15 pips; gold's is $1–3. **Feature scaling must be relative (ratios), not absolute.**
- Hurst/entropy may transfer; wavelets likely transfer; `time_of_day` ablation (65% of signal in gold) will almost certainly be dominant in EUR/USD too but at different peak hours.

**Action:** replicate feature pipeline from `python/goldbigbrain_features.py`, retrain LightGBM on EUR/USD M5, run the same ablation. Do NOT assume gold's top features are EUR/USD's top features.

### 6.2 No validated session window

Gold's 13–20 GMT session was window-invariant (cfg74 bootstrap test showed any 7h window works equally). EUR/USD has different session dynamics — **do the session-shuffle test early** to see if a session label even matters. Likely candidates: London open 07–10 (range expansion), NY/London overlap 13–16 (max liquidity).

### 6.3 Spread/cost profile is radically different

- **XAUUSD:** 18 pt = $0.18/oz median; contract 100 oz, tick value $1/pt, stops level 20 pts
- **EURUSD on Vantage Standard STP:** typical 1 pip = 0.0001 during London/NY, wider at Asia session open. Total cost ≈ 1.1 pip. Stops level and commission both need verification from MT5 `SymbolInfoInteger`.

EUR/USD's narrower spread relative to price changes the edge math entirely. A strategy with TP=2×ATR means TP~20 pips; cost 1 pip = 5% drag. For gold TP=2×ATR means TP~$2, cost $0.20 = 10% drag. **EUR/USD is cheaper to trade but also moves less in absolute terms.**

### 6.4 Retail crowding

EUR/USD is the most-traded retail pair. Means any "obvious" edge (moving averages, Bollinger bounce, round-number fade) is already front-run. Gold also had this problem but not as severely. **Lean harder on the ML vol-filter and unusual session/regime conditioning — the mechanical edge will be thinner.**

### 6.5 DD scales with compounding (3–4× vs fixed-lot sim)

Same rule as gold but worth repeating. Python sim at fixed 0.01 lot showing 4% DD → MT5 at RiskPercent=0.6% showing 13–14% DD. **Always MT5 every-tick at target RiskPercent.** Deploy target DD is whatever the compounding run shows, never the fixed-lot number.

### 6.6 Currency-pair peculiarities

- **Holiday effects / low-liquidity periods** (NFP release, FOMC minutes, Asian holidays) can blow out stops. Consider `EventFilterMinutesBefore/After` inputs like gold had.
- **Roll-over cost** (overnight swap) on open positions. Not relevant for an M5 intraday strategy with `MaxHoldBars=12` (60 min), but verify the strategy closes before 21:00 broker server time to avoid triple-swap Wednesdays.
- **Sunday gap** common in FX. Backtest must use broker's actual weekly data, not Dukascopy's inferred weekend.

### 6.7 Stay in research mode

Same as gold — **nothing deploys until user explicitly says go**. Frame every winner as "current best candidate." Build a portfolio of multiple validated (strategy, slot, regime) tuples before even considering deploy.

### 6.8 Reuse GBB_Core.mq5 architecture verbatim

Change: `Symbol` handle, `ONNX model buffer`, `FadeLongRSI`/entry module per strategy. Keep: bracket lifecycle, ATR-scaled SL/TP/BE/Trail, session gate, vol gate, risk engine, `MaxLotSize`, `DailyLossCapPct`. **The gold code is battle-tested; don't re-invent it.**

---

## 7. Top-10 distilled commandments

**For quick reference when making any EuroBigBrain decision:**

1. **Multi-year MT5 every-tick or it didn't happen.** Single-period = candidate. Show year-by-year every time.
2. **Never trust Python sim without a published MT5 calibration ratio.** Sim inflates 13–20× on old versions; 1.4–1.8× on V5-family even in the new engine.
3. **Bootstrap + year-by-year + direction-flip nulls are the real validity test.** Session-shuffle is a substitutability diagnostic, not a gate.
4. **ML predicts vol (WHEN), not direction (WHICH WAY).** AUC 0.67 vs 0.51 settled this. Build the ONNX vol-filter; let the mechanical strategy handle direction.
5. **Workers=1 for any deploy-critical MT5 grid.** Workers>1 corrupts CSVs via wine prefix log race.
6. **Multi-year MT5 every-tick caps: Mac=4, Win=3 on 16 GB boxes.** Math N × 3.3 GB + 5 GB OS ≤ 16 GB.
7. **No redundant parallel work.** Every multi-machine launch states "Win = slice X, Mac = slice Y." Different `--out` ≠ different work.
8. **Never static-blacklist from retro-filter stats.** Non-additive trade dependencies. +$9k → -$9 taught this.
9. **Lookahead-bias audit every top feature.** Rolling windows `[i-N:i+1]`, shifts positive. Unexpectedly dominant feature = investigate before celebrating.
10. **Research mode only.** Every winner is "current best candidate." No deploy until user explicitly greenlights.

---

_Synthesised from 2026-04-13 through 2026-04-21 memory, covering GoldDigger → GoldBigBrain → London Beasts (Unchained, On a Leash, & Co.) → M113/XAU-overfit discovery. Three MT5-verified XAUUSD variants exist as reference architecture; zero deploy. EuroBigBrain starts from clean EUR/USD microstructure with this rulebook in hand._
