# EBB Tick-Replay Calibration Framework — Design Doc

**Status:** Draft v0.1 (2026-04-21)
**Scope:** Week 1–2 infrastructure deliverable per WG4 architecture mandate
**Gate:** 100-config calibration grid within ±5% PF vs MT5 every-tick before ANY strategy result is trusted
**Hard constraint:** M5 OHLC sim is BANNED in EBB. Tick-replay or nothing.

---

## 0. Why this doc exists

GoldBigBrain's M5 OHLC sim inflated PF by 1.66× median vs MT5 and cost us
months of chasing phantom winners. Phase 2A built a tick-replay engine that
executed 100 configs in 36 s, but Gate 2A.4e returned a catastrophic **0.195
median ratio** — tick fired 5–20× more trades than MT5 because
`mt5_sim_strategies.generate_signals` lacked the per-day "level already
crossed" guard that the MT5 EA had implicitly. The engine state machine was
fine; the signal generator was wrong.

EBB inherits the engine and **must not inherit the bug**. This doc specifies
the architecture, the calibration protocol, the EUR-specific adaptations,
and the abort criteria.

---

## 1. Architecture

### 1.1 Tick source — Dukascopy primary, broker secondary

Three candidates evaluated:

| Source              | Coverage       | Quality              | Cost | Verdict          |
|---------------------|----------------|----------------------|------|------------------|
| MT5 export (Vantage)| 2021-onwards   | Broker-aligned       | Free | **Secondary**    |
| Dukascopy (JForex)  | 2003-onwards   | Reference-grade tick | Free | **PRIMARY**      |
| Broker API (live)   | Rolling window | Real but thin history| Free | Paper-live only  |

**Why Dukascopy primary:** deepest history (we need 2020–2026 minimum for
walk-forward), true bid/ask tick pairs (not synthetic from OHLC), and the
de-facto reference for FX backtesting. Resolution is 0.1 pip which matches
EUR scale.

**Why MT5 export secondary:** we calibrate *to* MT5 every-tick. Using MT5's
own M1-tick-reconstruction as the replay source gives tautologically good
agreement. But its tick depth on EUR is thinner than Dukascopy and its
history before 2021 is broker-provisioning-dependent. Keep it as the
**calibration fidelity check** input, not primary production input.

**Storage:** DuckDB warehouse partitioned `symbol=EURUSD/year=2024/...`,
one parquet per (symbol, year-month). Schema:
`(ts_unix_ns INT64, bid DOUBLE, ask DOUBLE, flags UINT8)`. Flags encode
weekend gap, news-halt, and pre-NY-close session markers so downstream
code never re-derives them.

Estimated size (§5): ~15 GB uncompressed for EUR 6 yrs, ~2.5 GB zstd-parquet.

### 1.2 Replay engine

Re-use Phase 2A `tick_replay.py` (1175 lines, numba-JIT kernel) as
starting point. Carry-over tasks:

1. Port to `python/ebb/tickreplay/engine.py` with adapted config schema.
2. Bake in the **per-day level-crossed guard** (§3) as a first-class
   engine primitive — every `SignalBase` subclass supplies `LevelKey`
   tuples `(symbol, day, side, level_price)` and the engine refuses to
   re-fire on a key whose level has been crossed intraday.
3. Keep the JIT inner loop — 36 s / 100 configs was the only thing that
   worked in Phase 2A.
4. Replace ONNX inference shim with LightGBM ONNX that matches the EBB
   `VolFilter.mqh` bytecode byte-for-byte.
5. Add spread-sampling hook: engine maintains a 1 h rolling median spread
   per symbol; spread-guard rule (reject if spread > N× median) is an
   engine-level primitive, not per-strategy logic.

### 1.3 EBB_Core invocation pattern

Python signal code MIRRORS MT5 EA signal code via a **shared YAML config**.
One signal interface:

```python
class SignalBase:
    def on_tick(self, state: TickState) -> Optional[Signal]: ...
    def on_bar_close(self, bar: M5Bar) -> None: ...
    def level_keys(self, day: date) -> list[LevelKey]: ...
```

Parallel MQL5:
```cpp
class CSignalBase {
    virtual Signal OnTick(const TickState&) = 0;
    virtual void   OnBarClose(const M5Bar&) = 0;
    virtual void   LevelKeys(datetime, LevelKey[]&) = 0;
};
```

Determinism CI (§2.4) diffs the trade tape byte-for-byte between Python
replay and MT5 every-tick on a reference config. A merge is rejected if
output diverges — this forces the Python and MQL5 implementations to be
literally the same logic.

### 1.4 Output trade log

One CSV per run, schema:

```
run_uuid, config_id, symbol, trade_id, entry_ts, entry_px, entry_side,
sl_px, tp_px, exit_ts, exit_px, exit_reason, bars_held, slip_pips,
spread_at_entry, level_key, signal_confidence, onnx_score
```

`exit_reason ∈ {tp, sl, be, trail, eod, kill, gap}`. `gap` flag is
critical — gap-through-stop fills are the single biggest source of
sim/MT5 divergence (§4) and must be auditable.

---

## 2. The 100-config calibration test

### 2.1 Config mix

100 configs stratified across the axes that matter for fidelity:

| Axis                | Levels                         | Count |
|---------------------|--------------------------------|-------|
| Signal family       | Fade, Momentum, Breakout, AR   | 4     |
| Session             | Asia, London, NY, Overlap      | 4     |
| SL (ATR mult)       | 0.2, 0.5, 1.0, 2.0             | 4     |
| TP (ATR mult)       | 0.5, 1.5, 3.0, 5.0             | 4     |
| VolFilter threshold | off, 0.5, 0.7                  | 3     |

Full cross = 768. **Randomly sample 100** with a fixed seed (42) so the
calibration set is reproducible. Reserve a disjoint **holdout 20** with
seed 43 — never shown to the engine during iteration. The holdout is the
final go/no-go gate after the calibration 100 passes.

### 2.2 Ground-truth MT5 run

Each of the 100 configs runs through MT5 every-tick on EURUSD 2023-01-01
to 2025-12-31 (3 yr window, representative of 2020–2026 but half the
compute). Expected wall time on Win+Mac 9-worker pool: ~4 h per grid
(GBB took 50 min for its 100-config grid; EUR dataset is similar size).

Output: `results/calibration/mt5_100_v1.csv` with columns
`config_id, pf, trades, net_pnl, max_dd, sharpe`.

### 2.3 Pass criteria

For each config compute `ratio = pf_tick / pf_mt5`.

| Metric                  | Gate                      |
|-------------------------|---------------------------|
| median ratio            | 0.95 ≤ x ≤ 1.05           |
| p5  ratio               | ≥ 0.85                    |
| p95 ratio               | ≤ 1.15                    |
| Configs within ±5% PF   | ≥ 70 of 100               |
| Configs within ±15% PF  | ≥ 95 of 100               |
| Trade-count delta       | mean |Δ| ≤ 10% per config |

The **trade-count delta** is non-negotiable. Phase 2A passed PF on some
configs accidentally (gain/loss sum ratios cancelled errors) while trade
counts were 5× off. Trade count is the smoking gun for logic divergence.

### 2.4 Determinism gate

Every commit to the engine triggers a CI job that runs **config 001
twice** and byte-diffs the output CSV. Any byte-level difference blocks
merge. This is cheap (~30 s) and catches the entire class of bugs where
a "harmless refactor" silently changes tie-break semantics.

---

## 3. The per-day level-crossed guard (THE gold lesson)

From Phase 2A post-mortem: MT5's `GBB_AsianRange.mq5` had an implicit guard
at line 192:

```mq5
if(buyLevel <= currentAsk || sellLevel >= currentBid) return;
```

Once price had touched the Asian high intraday, the EA stopped trying to
re-place the BuyStop. Python signals had no such guard → 5–20× overfire.

### EBB engine-level solution

**Every signal** must declare its `LevelKey` set per day. The engine
maintains a `crossed_levels: set[LevelKey]` and **refuses** to emit any
signal whose `LevelKey` is in the set. Crossing is detected on every
tick, not every bar close.

Signal interface is non-negotiable on this — no signal can bypass. A
signal that genuinely has no levels (e.g. pure ML score) returns an empty
list, and the engine treats every signal as fresh.

### CI test

`tests/test_level_crossed_guard.py`: synthetic day where price oscillates
10× through a breakout level. AssertEqual(trades_fired, 1). Any signal
implementation that fails this test fails CI.

This is the single most important lesson from gold and it goes into the
engine on day 1, not as a retrofit.

---

## 4. EUR-specific adaptations (vs gold)

### 4.1 Market hours — 24/5 vs XAU 23/5

XAU has a **daily 60-min CME settlement halt** (21:00–22:00 UTC).
EUR runs continuously Sun 22:00 UTC → Fri 22:00 UTC with **no daily
halt**. Implications:

- No intra-day "reset bar" — Asian-range breakout logic must use a
  calendar-day anchor (00:00 UTC) not a session-halt anchor.
- Daily PnL cap (2%) needs to be anchored on explicit session boundaries;
  suggested `daily_reset = 22:00 UTC` (NY close) to align with prop-firm
  daily-loss conventions.
- The weekend gap on EUR is typically 5–15 pips; on XAU it was 50–200
  "pips". EUR gap-through-stop probability is LOW but non-zero and must
  still be modelled.

### 4.2 Spread & slip math

EUR typical spread 0.6–1.5 pips on Vantage STP; XAU was 30–40 "pips"
(where 1 XAU-pip = $0.10 on 1 oz). Slip calculation:

- **Slip = min(spread_at_fill, 1-sided latency scalar)**.
- Latency scalar calibrated from broker round-trip: EUR on Vantage ≈
  0.3 pips at 95th percentile.
- Stop-buy fills on EUR: fill at max(stop_price, next_ask) rather than
  gold's stop_price + spread rule. This matches observed Vantage MT5
  behaviour for EUR on sub-pip moves.

The calibration grid **must include** at least 20 configs with SL ≤ 0.2
ATR because that is where slip dominates and divergence is worst.

### 4.3 NY close gap behaviour

On gold the NY close 22:00 UTC routinely saw 10–30 point rips on
algo-rebalancing. On EUR the NY close is much quieter — the move tends
to be 1–3 pips. BUT: the Sunday open gap is the real event, and EUR
traders often cap weekend exposure explicitly.

Engine must:

- Flag ticks in `last_5_min_before_22:00_Fri` with `flags.WEEKEND_RISK`.
- Signal-level `allow_weekend = false` default causes engine to close
  all open positions at 21:55 Fri and refuse new entries after 21:50 Fri.
- Replay a synthetic Monday-open gap test (±5 pips) as part of CI.

### 4.4 ECB/FOMC event blackouts

XAU reacted to Fed only. EUR reacts to **both** ECB and Fed — roughly
2× the event-risk days. News feed (ForexFactory API) integrated at
engine level, not signal level: ±10 min window around High-impact events
is flagged `flags.NEWS_HALT` and all signals are vetoed.

---

## 5. Compute budget

### Tick dataset size (EUR 6 yr)

- EUR average **~12 ticks/sec** during liquid hours, ~1 tick/sec Asia
  pre-session.
- Active seconds per week: `5 days × 24 hr × 3600 s = 432 000 s`.
- Weekly ticks ≈ `432 000 × 6 = 2.6 M`.
- Annual ticks ≈ `2.6 M × 52 = 135 M`.
- **6 yr EUR ≈ 810 M ticks**. At 20 bytes/tick (ts+bid+ask+flags packed)
  = 16 GB uncompressed, ~2.5 GB zstd-parquet.
- Fits in RAM on Mac (16 GB) at zstd-compressed, streams from parquet
  on Win (16 GB) with DuckDB lazy scan.

### Replay speed target

- Phase 2A hit **36 s / 100 configs** on gold M5 (~2.2 yr, ~200 M ticks)
  with numba JIT.
- EUR 810 M ticks ≈ 4× data. Target: **≤ 150 s / 100 configs single
  machine**, ≤ 40 s with 4 workers.
- Ticks/sec target: **5 M ticks/sec/worker** (Phase 2A measured 5.5 M).

### Full grid turnaround

- Calibration (100 configs, 3 yr): < 2 min Mac 4-worker.
- Full research grid (e.g. 10 000 configs, 6 yr): ~4 h on Win + Mac
  combined. Acceptable for overnight run.
- **Key win vs MT5-only:** 100-config MT5 every-tick grid is ~4 h.
  Tick-replay is **~2 min**. That's the **120× speedup** that makes fast
  iteration possible. If we can't hit it, the framework has no reason to
  exist.

### Data ingestion

One-time cost: ~6 h to download Dukascopy EUR ticks 2020–2026 via
`dukascopy-node` at 10 requests/sec (polite), plus ~1 h DuckDB ingest.
Total: one evening.

---

## 6. Failure mode — what happens if calibration gate fails

**Decision: we do NOT fall back to MT5-only.** MT5-only means 4 h per
100 configs = no iteration loop. The whole point of tick-replay is the
120× speedup.

### Staged response

| Outcome                                         | Action                                 |
|-------------------------------------------------|----------------------------------------|
| Gate passes (≥70 within ±5%)                    | Proceed to strategy work.              |
| Marginal (50–69 within ±5%, trade delta OK)     | Debug top-5 outliers trade-by-trade.   |
| Trade-count delta > 20% on ≥ 10 configs         | STOP. Signal-generator logic bug.      |
| Trade count OK, PF dispersion wide (<50 w/ ±5%) | Fill-semantics bug — audit slip model. |
| Median off by > 10%                             | Systemic bias — recalibrate spread.    |
| Disagreement on deterministic CI (byte diff)    | Block merge. Fix engine-level bug.     |

### If trade count is OK but PF dispersion is stubbornly wide:

Inspect per-trade `entry_px`, `exit_px`, `slip_pips` diffs. Common cause
on gold was stop-fill semantics (MT5 fills at `stop_price + current_spread`
at the moment of breach; some replay engines fill at `stop_price` flat).
Fix at engine level, re-run.

### Abort condition — the hard escalation

If after **2 full weeks of iteration** the gate still can't hit
≥ 70 within ±5%, escalate:

1. Publish `findings/tick_replay_abort.md` with per-config residual
   analysis.
2. Consider **MT5 Strategy Tester subprocess harness** as replacement:
   run MT5 ST in batch via CLI, parallelised over 10 prefixes
   (GBB hit this workflow at ~50 min / 100 configs). Slower than
   tick-replay but deterministic.
3. DO NOT extend indefinitely. The gold attempt took 2 weeks and failed;
   budget 2 weeks here, then escalate or pivot.

---

## 7. Deliverables & ownership

| # | Artifact                                               | Week |
|---|--------------------------------------------------------|------|
| 1 | `python/ebb/tickreplay/engine.py` (carry-over + guard) | W1   |
| 2 | DuckDB EUR tick warehouse 2020–2026                    | W1   |
| 3 | 100-config calibration grid CLI                        | W1   |
| 4 | MT5 ground-truth 100-config CSV                        | W1   |
| 5 | Gate script + CI job                                   | W2   |
| 6 | Determinism CI (byte-diff)                             | W2   |
| 7 | Level-crossed guard unit test                          | W2   |
| 8 | This doc, frozen v1.0                                  | W2   |

A strategy is merged to `main` only if (a) replay engine gate is green
and (b) strategy-level gates (null, bootstrap, WF, MC-DD, multi-inst)
are green.

---

**End of design doc.** Next: build it. If it doesn't hit the gate in two
weeks, we escalate per §6.
