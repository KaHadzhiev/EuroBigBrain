# Kill Switch + Quarter-Kelly Vol-Target Sizing — EuroBigBrain Design

**Status:** implemented (WG4 mandate, 2026-04-21). Research mode only — live deploy remains blocked per user's standing rule 2026-04-20 ("current best candidate, never ready"). Code lives at `mql5/EBB/KillSwitch.mqh`, `mql5/EBB/Sizing.mqh`, and `python/validation/kill_switch_simulator.py`.

## 1. Why dual-layer safety before a single live trade

GoldBigBrain nearly deployed strategies that showed 4% Python-sim drawdowns and 13.9% live drawdowns (rsisl "On a Leash"; LESSONS 1.12). Unchained ran +8525% but 35.4% DD — not a scenario you'd hand-calibrate risk around once blood is on the chart. The M113 fade_long probe (April 2026) wiped 96–99.85% of equity on non-XAU instruments when the XAU-tuned vol thresholds let trades through. A pre-deploy circuit breaker that caps *both* intraday pain and peak-to-trough damage is non-negotiable before any EuroBigBrain EA sees real money.

The mandate (WG4 architecture, Section 7) is dual-layer because each layer handles a different failure mode:

| Layer | Trigger | Response | Failure mode covered |
| --- | --- | --- | --- |
| A: daily loss cap | today_PnL ≤ -2% of start-of-day equity | block new entries until next session | flash losing streak, news spike, bug that opens too-large positions |
| B: account DD cap | equity ≤ peak × 0.92 (8% from ATH) | halt ALL trading, close open positions, latch for manual reset | regime shift, edge decay, ML model stale — slow bleed |

Layer A is a same-session circuit. Layer B is a career-preserving latch.

## 2. KillSwitch.mqh — implementation

**Persistence:** every state change writes to MT5 `GlobalVariable` keys (`EBB_KS_sod_equity`, `EBB_KS_peak_equity`, `EBB_KS_today_pnl`, `EBB_KS_acct_halt`, etc). If the EA restarts mid-day, `KS_Init` restores peak + SoD + halt flag. A latched Layer B halt survives terminal reboots — the operator must explicitly call `KS_ResetAccountHalt()` (or edit the GV) to resume trading. This is deliberate: the purpose is to force the human to look at the dashboard before restarting.

**Ordering in `KS_AllowNewEntry`:** override → Layer B → Layer A → consec-brake. Layer B precedes A because B is latching and should short-circuit regardless of the intraday state. The manual override flag (`InpKillSwitchOverride`) is for unit tests and synthetic drills only; in production it is `false` and any commit that flips it to `true` should fail CI.

**`KS_CloseAllForMagic`** fires positions via `TRADE_ACTION_DEAL` at the current bid/ask (not via the position's SL). This eliminates the risk that a broken SL or a gapped market leaves positions open while the EA thinks it's halted.

**Heartbeat:** `KS_Heartbeat` runs every tick to (a) update peak equity if we've made a new high, (b) roll over the day. It does NOT evaluate the gates — that's `KS_AllowNewEntry`'s job, called by Core right before signal placement.

## 3. Sizing.mqh — quarter-Kelly × vol-target

Formula (from WG4 Section 6):

```
kelly_frac  = max(0, (W·A − L·B) / A)
raw_pct     = kelly_frac · 0.25 · 100
risk_pct    = min(base_risk_pct, raw_pct) · vol_scalar      // vol ∈ [0.5, 2.0]
risk_pct    = min(risk_pct, base_risk_pct)                   // hard cap
lot         = (equity · risk_pct / 100) / (SL_ticks · tick_value)
```

### 3.1 Baked-in assumptions

Kelly needs four numbers: `win_count`, `loss_count`, `avg_win_R`, `avg_loss_R`. The sizer does NOT hard-code these — they come from a rolling trade-log tracker updated weekly. However the *baseline expected behaviour* relies on an implicit prior:

- **cfg74 heir profile** (gold's best-validated edge, analogous target for EURUSD): ~45% win-rate, avg-win ≈ 3R, avg-loss ≈ 1R. Kelly fraction ≈ `(0.45·3 − 0.55·1) / 3` = `0.267`. Quarter of that = `0.067`. As a raw percent: `6.7%`. Clamped against `base_risk_pct = 0.5%`, the sizer lands at `0.5% × vol_scalar` — i.e. the hard cap dominates.
- **weak-edge profile** (win-rate 0.40, avg-win 1.5R, avg-loss 1R): `(0.6·1.5 − 0.4·1) / 1.5` = `0.267`, same quarter-Kelly. Still capped.
- **no-edge profile** (win-rate 0.45, avg-win 1R, avg-loss 1R): `(0.45·1 − 0.55·1) / 1` = `-0.10` → clamped to 0, fallback 0.3% fixed.
- **breakeven profile** (win-rate 0.50, 1R/1R): kelly_frac = 0 → fallback.

So in practice, for any positive-expectancy strategy with ≥200 trades, the sizer simply enforces 0.5% × vol-scalar. Kelly here functions less as a "find the optimal bet size" engine and more as an **automatic risk killer** — when edge evaporates, it drops sizing to 0 and the fallback takes over at a conservative 0.3%. This is the deliberate design: Kelly is a *veto*, not a target, matching LESSONS 3.1 (ML as veto, not primary signal).

### 3.2 Why quarter-Kelly specifically

Full-Kelly requires exact knowledge of `W`, `A`, `B`. Our estimates have wide CIs (bootstrap p<0.01 says the edge *is*, not *how big*). Half-Kelly (common on prop desks) still exceeds 0.5% for cfg74-profile edges. Quarter-Kelly times a 0.5% base is the right size so that a 2× overestimate of Kelly still lands inside the cap. See Thorp 1997 for the Kelly-CI argument.

### 3.3 Vol-target overlay

`realized_vol` is measured on a rolling 20-day window of symbol returns; `target_vol` is a config value (e.g. 0.0075 for EURUSD, because EURUSD's historical σ is ~0.6%/day and a target a bit above preserves bet frequency). `vol_scalar = target_vol / realized_vol`, clipped to `[0.5, 2.0]`. Purpose: keep $-risk-per-trade constant across regimes instead of letting high-vol weeks produce 3× the dollar loss on the same setup.

If volatility measurement fails (data gap, NaN), `Sizing_VolScalar` returns 1.0 (neutral) and we fall through to fixed 0.3%. **Never scale up on measurement failure** — error-safe by construction.

## 4. Interaction between the two gates

**Do Layers A and B ever conflict?** Layer A blocks new entries on an intraday basis; Layer B is a latching halt. There is no conflict because:

- If Layer A trips first (e.g. a -2% day after 9 straight green days so DD from peak is only 2%), Layer B stays inactive. Tomorrow's SoD rollover re-allows trading. Good.
- If Layer B trips (cumulative 8% from peak), it supersedes A regardless of intraday state. Layer A's "tomorrow it resets" semantics are irrelevant — Layer B is latched.
- **Edge case:** Layer B *precedes* Layer A in the predicate ordering. This means on a day where the final losing trade pushes both "today -3%" and "peak-to-trough 8%", the account halt is reported and the daily trip is suppressed. The Print log distinguishes "LAYER B TRIP" vs "LAYER A TRIP" for clarity.

## 5. kill_switch_simulator.py — validation

Replays a trade log (or synthetic fixture) through a Python mirror of `KS_AllowNewEntry`. Test fixtures:

- `synthetic_2pct_daily` — 6 losers of 0.5% each; Layer A must trip on trade 5.
- `synthetic_8pct_account` — 80 small losers over 20 days, each day < 2%; Layer B must latch around day 5–6.
- `m113_like_wipe` — 200 losers cumulating to -96%; kill switch must save ≥85% of starting equity.
- `unchained_like_35pct` — up 50% then bleed 35%; Layer B must latch near the 8% DD mark.
- `manual_override` — all blocks bypassed; sanity check.

All five pass. `m113_like_wipe` finishes at **$9,191** instead of ~$400 — the simulator confirms that if we had been running the WG4 kill switch during the M113 probe, we would have lost ~8% instead of 96%.

## 6. Deploy gates (future — reminder of the standing rule)

No EBB candidate goes live without (per LESSONS 2.1 and WG4 Section 3):

1. Kill-switch unit test must synthesise a 2% intraday loss and verify halt.
2. Kill-switch drill on demo account before paper-live.
3. `p99 DD ≤ 20%` on MC-DD harness.
4. User explicit go — research mode until then.

## 7. Sources

- WG4 architecture (`findings/wg4_architecture_improvements.md`), sections 6–7.
- LESSONS_FROM_GOLDBIGBRAIN, sections 1.12, 2.1, 3.1, 4.7.
- GBB `mql5/GBB/TradeMgmt.mqh` — position sizing reference (fixed-% model, now superseded by Kelly).
