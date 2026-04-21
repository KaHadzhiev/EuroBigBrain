# EBB_Core.mq5 — Symbol-Agnostic Refactor Design

**Author:** EBB architecture pass
**Date:** 2026-04-21
**Source files reviewed:**
- `GoldBigBrain/mql5/GBB_Core.mq5` (334 lines)
- `GoldBigBrain/mql5/GBB/{Types,StrategyParams,State,Indicators,Features,TradeMgmt}.mqh`
- `GoldBigBrain/mql5/GBB/Strategies/*.mqh` (15 strategy modules)
- `EuroBigBrain/findings/wg4_architecture_improvements.md`
- `EuroBigBrain/findings/LESSONS_FROM_GOLDBIGBRAIN.md`

**Goal:** produce one `EBB_Core.mq5` binary that takes `InpSymbol` as input and runs identically on EURUSD, GBPUSD, USDJPY, XAGUSD M5 with zero code changes per symbol. Compliant with the M113-killer rule (WG3): any strategy that wipes ≥3/4 of the secondaries is rejected at CI.

---

## 1. XAU-Specific Assumptions in Current GBB_Core

The good news: **~75% of GBB already routes through `_Symbol`, `SymbolInfoDouble`, `SymbolInfoInteger`**. The cost/risk/normalize plumbing is clean — `TradeMgmt.mqh:40-54` reads `SYMBOL_TRADE_TICK_VALUE`, `SYMBOL_TRADE_TICK_SIZE`, `SYMBOL_VOLUME_STEP`, `SYMBOL_VOLUME_MIN` at runtime; `TradeMgmt.mqh:62,97,168` use `SYMBOL_DIGITS`; all bar access uses `_Symbol`. The ATR-relative SL/TP/BE/Trail multipliers in inputs (`GBB_Core.mq5:34-51`) are already scale-free.

The bad news is clustered in five concrete spots:

### 1.1 Round-number features hardcoded to $50/$100 grid
**File:** `GBB/Features.mqh:114-117`
```
double mod50  = MathMod(c, 50.0);   g_features[41] = ...MathMin(mod50, 50.0-mod50)/50.0...
double mod100 = MathMod(c, 100.0);  g_features[42] = ...MathMin(mod100,100.0-mod100)/100.0...
```
$50/$100 psychological levels make sense at XAUUSD ~$2,000. On EURUSD (~1.08), `MathMod(1.08, 50.0) = 1.08` always → feature is a constant, dead input that training will discover but adds noise.

### 1.2 Magic-number default branded 2026-04-20
**File:** `GBB_Core.mq5:42` — `input int MagicNumber = 20260420;`
Not XAU-specific per se, but every symbol instance needs its own magic range (WG4 kill-switch rule: "magic-number isolation, every EA instance owns a magic range"). Sharing this default across EUR/GBP/JPY instances lets them step on each other's positions.

### 1.3 ONNX model resource is compile-time-bound to XAU vol-model
**File:** `GBB_Core.mq5:7` — `#resource "vol_model_6yr.onnx" as uchar VolModelData[]`
**File:** `GBB_Core.mq5:165` — `OnnxCreateFromBuffer(VolModelData, ONNX_DEFAULT)`
Baked at compile. Feature distributions (ATR scale, round-number mod, hour-of-day peaks) all differ across symbols — a gold-trained 59-feature vol_model will output garbage probabilities on EURUSD. LESSONS §6.1: "feature distribution will differ."

### 1.4 MaxLotSize hardcoded to 0.10 (gold contract sizing)
**File:** `GBB_Core.mq5:43` — `input double MaxLotSize = 0.10;`
0.10 lots XAUUSD = 10 oz notional ~$20k; 0.10 lots EURUSD = 10k notional; 0.10 lots USDJPY = 10k USD notional. The dollar-risk-per-pip differs by ~100×. Not broken (you can just reconfigure), but the *default* is XAU-calibrated. Needs to auto-derive from `SYMBOL_VOLUME_MAX` or a risk-driven cap.

### 1.5 Trend-filter ATR(14) hardcoded, reasonable but locks the timeframe-14 combo
**File:** `GBB_Core.mq5:156` — `g_hTrendATR = iATR(_Symbol, TrendTimeframe, 14);`
Fine on XAU at H4. For JPY pairs the H4 ATR can be numerically dominant over price-distance in ways that invert the trend filter semantics. Not broken; flagged for audit.

### 1.6 M5 timeframe hardcoded everywhere
**Files:** `GBB_Core.mq5:196,219,220`, `GBB/Indicators.mqh:85`, `GBB/Features.mqh:102`
Every `iTime`, `iHigh`, `iLow`, `iClose`, `CopyRates` pins `PERIOD_M5`. Not an XAU assumption, but worth parameterising so EBB can probe M15/H1 edge on slower majors if M5 turns out dead (LESSONS §6.4: retail crowding on EURUSD).

### 1.7 Session hours default 7-20 UTC (XAU "London + NY")
**File:** `GBB_Core.mq5:40-41`
Not hardcoded, but the default 7-20 is the cfg74-validated gold window. Per LESSONS §6.2 "do the session-shuffle test early" on EURUSD. This is a *config* issue, not code — flag only.

### 1.8 Stop-level / min-distance never queried
**Nowhere in GBB_Core.** `TradeMgmt.mqh:56-89` (`PlaceBracket`, `PlaceSingle`) never checks `SYMBOL_TRADE_STOPS_LEVEL`. XAU's 20pt stops level rarely matters at SL_ATR_Mult=1.0; on USDJPY with tight SL=0.2×ATR and broker stops-level = 30 points, the order will reject silently. This is a latent bug that will surface on migration.

### 1.9 Spread-guard absent
**Nowhere in GBB_Core.** WG4 §7 kill-switch: "reject signals when spread > 2× 1h median." Not in GBB. A symbol-agnostic EA absolutely needs this — Asian-session EURUSD spread can spike 3–4×, and the XAU-trained VolThreshold won't protect against it.

### 1.10 Hardcoded feature count = 59
**File:** `GBB/Types.mqh:26` — `#define NUM_FEATURES 59`
Locks all symbols to the same ONNX input shape. If the EURUSD model drops dead features (§1.1) and adds carry/DXY/event-proximity features, this define must move from header to per-symbol config.

---

## 2. How Each Becomes Symbol-Agnostic

### 2.1 Round-number grid → symbol-aware tick scale
Replace `Features.mqh:114-117` with a price-relative grid:
```mql5
double price_scale = SymbolInfoDouble(_Symbol, SYMBOL_BID);
// round-number increment: 5% and 10% of price-per-1000-points
double grid_major = price_scale * 0.01;   // 1% psychological
double grid_minor = price_scale * 0.005;
```
Or cleaner: compute as multiples of `_Point × 10^N` where N comes from digits. For EURUSD (5-digit), `grid_minor = _Point*500 = 0.005`; for XAUUSD (2-digit), `grid_minor = _Point*500 = 5.0`. Scales naturally.

### 2.2 Magic range auto-derived
`MagicNumber = InpMagicBase + SymbolHash(InpSymbol)` where `SymbolHash` is a deterministic 16-bit hash of the symbol string. Guarantees non-collision without per-symbol config edit. EA refuses to start if `PositionsTotal()` contains a magic in its range that's not its own (kill-switch magic-isolation rule).

### 2.3 ONNX by symbol-suffixed filename (runtime load, not compile-time resource)
Drop `#resource` at line 7. Replace `OnnxCreateFromBuffer(VolModelData, ...)` at line 165 with file-based load:
```mql5
string modelPath = StringFormat("EBB\\models\\vol_model_%s.onnx", InpSymbol);
g_hVolModel = OnnxCreate(modelPath, ONNX_DEFAULT);
```
Model files live in `MQL5\Files\EBB\models\` and ship with the EA via installer. Missing file → EA init fails with explicit error (not silent fallback — LESSONS §2.3: "mi_edge silent XAUUSD-fallback burn cost 40 CPU-minutes"). Model metadata JSON alongside (`vol_model_EURUSD.json` with `num_features`, `feature_hash`, `git_sha`) — EA asserts hash matches before running.

### 2.4 MaxLotSize auto-cap by risk budget
Replace fixed `MaxLotSize = 0.10` with two inputs:
```mql5
input double MaxRiskUsdPerTrade = 50.0;   // hard-dollar cap
input double MaxLotSizeAbsolute = 0.0;    // 0 = auto-derive
```
In `CalcLotSize` (TradeMgmt.mqh:37), compute `lot_cap_from_risk = MaxRiskUsdPerTrade / (slTicks * tickValue)` and cap lot at `min(MaxLotSizeAbsolute || huge, lot_cap_from_risk, SYMBOL_VOLUME_MAX)`. Never hand the EA a symbol without a dollar-risk sanity cap.

### 2.5 Trend-filter parameterised ATR period
Add `input int TrendAtrPeriod = 14;`. Pass into `iATR` at line 156. Default 14 stays; user can rescan on JPY.

### 2.6 Timeframe parameterised
```mql5
input ENUM_TIMEFRAMES PrimaryTimeframe = PERIOD_M5;
```
Replace every `PERIOD_M5` literal in `GBB_Core.mq5`, `Indicators.mqh`, `Features.mqh` with `PrimaryTimeframe`. One global search-replace. Pre-flight assertion: `PERIOD_M1 ≤ PrimaryTimeframe ≤ PERIOD_H1`.

### 2.7 Stops-level guard before every order
In `TradeMgmt.mqh` `PlaceBracket`/`PlaceSingle`, before submitting:
```mql5
double stopsLvl = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
double minDist = MathMax(stopsLvl, SymbolInfoDouble(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) * _Point);
if(MathAbs(level - currentPrice) < minDist) return;  // skip rather than reject
if(sl_pts < minDist) sl_pts = minDist;
if(tp_pts < minDist) tp_pts = minDist;
```

### 2.8 Spread guard in `Session.mqh` (new module per WG4 §1)
```mql5
int spreadPts = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
if(spreadPts > g_spread_median_1h * 2) return;   // skip bar
```
`g_spread_median_1h` updated rolling via a 12-sample ring on every new M5 bar.

### 2.9 NUM_FEATURES becomes runtime-configurable
Move `#define NUM_FEATURES 59` out of `Types.mqh`. Replace with `int g_num_features;` populated in `OnInit` from the ONNX model metadata JSON. `Features.mqh::ComputeFeatures` reads the symbol's feature spec from a YAML bundled with the model (feature-name → index) rather than hardcoding indices 0-58. Per-symbol models can have different feature counts.

---

## 3. EBB_Core.mq5 Skeleton

```mql5
//+------------------------------------------------------------------+
//| EBB_Core.mq5 — symbol-agnostic orchestrator EA                   |
//+------------------------------------------------------------------+
#property copyright "EuroBigBrain"
#property version   "1.00"
#property strict

#include "EBB/Types.mqh"
#include "EBB/Config.mqh"          // loads YAML config by InpSymbol
#include "EBB/RiskCore.mqh"        // NON-OVERRIDABLE sizing
#include "EBB/KillSwitch.mqh"      // NON-OVERRIDABLE halt logic
#include "EBB/PositionMgr.mqh"     // BE / trailing / time-stop
#include "EBB/Session.mqh"         // hour gate + news blackout + spread guard
#include "EBB/VolFilter.mqh"       // ONNX inference wrapper (runtime-loaded)
#include "EBB/Indicators.mqh"      // ATR / EMA / RSI — already symbol-agnostic
#include "EBB/Features.mqh"        // feature-registry-driven
#include "EBB/Telemetry.mqh"       // CSV + journal structured logging
#include "EBB/Signals/Base.mqh"
#include "EBB/Signals/Fade.mqh"
#include "EBB/Signals/Momentum.mqh"
#include "EBB/Signals/Breakout.mqh"
#include "EBB/Signals/MlEntry.mqh"

//--- Core inputs (symbol + meta only; all strategy params in config YAML)
input string  InpSymbol          = "EURUSD";
input string  InpConfigName      = "ebb_cfg_v0";    // loads configs/strategies/<name>.yaml
input ENUM_TIMEFRAMES PrimaryTimeframe = PERIOD_M5;
input int     InpMagicBase       = 26100000;
input double  InpRiskPercent     = 0.5;             // hard cap per WG4 §6
input double  InpMaxRiskUsdPerTrade = 50.0;
input double  InpDailyLossCapPct = 2.0;             // WG4 kill-switch
input double  InpAccountDDCapPct = 8.0;
input bool    InpEnableKillSwitch   = true;
input bool    InpEnableSpreadGuard  = true;
input double  InpSpreadGuardMult    = 2.0;

string g_symbol;           // InpSymbol, validated at init
int    g_magic;            // InpMagicBase + SymbolHash(g_symbol)
int    g_num_features;     // from model metadata
long   g_hVolModel = INVALID_HANDLE;
SignalBase* g_signal = NULL;

int OnInit()
{
   // 1. Validate symbol — assert tradeable, request bars subscribed
   if(!SymbolSelect(InpSymbol, true)) return INIT_FAILED;
   g_symbol = InpSymbol;

   // 2. Load config YAML → populate g_tparams, g_sparams
   if(!ConfigLoad(InpConfigName, g_symbol)) return INIT_FAILED;

   // 3. Derive magic from symbol hash (kill-switch isolation)
   g_magic = InpMagicBase + SymbolHash(g_symbol);
   g_trade.SetExpertMagicNumber(g_magic);

   // 4. Load symbol-specific ONNX model + metadata
   string modelPath = StringFormat("EBB\\models\\vol_model_%s.onnx", g_symbol);
   g_hVolModel = OnnxCreate(modelPath, ONNX_DEFAULT);
   if(g_hVolModel == INVALID_HANDLE)
   { PrintFormat("Missing ONNX model: %s", modelPath); return INIT_FAILED; }
   if(!VolFilterValidateMetadata(g_symbol, g_num_features)) return INIT_FAILED;

   // 5. Instantiate the selected signal (from config)
   g_signal = SignalFactory(g_config.signal_name);
   if(g_signal == NULL) return INIT_FAILED;

   // 6. Session / KillSwitch / RiskCore init
   SessionInit(g_config.session_start, g_config.session_end);
   KillSwitchInit(InpDailyLossCapPct, InpAccountDDCapPct);
   RiskCoreInit(InpRiskPercent, InpMaxRiskUsdPerTrade);

   // 7. Stops-level + spread-median warmup
   PreflightSymbolSpecs();
   return INIT_SUCCEEDED;
}

void OnTick()
{
   if(!NewBar(PrimaryTimeframe)) return;

   // --- NON-OVERRIDABLE gates, in order ---
   if(!KillSwitchCheck())      return;    // daily loss / account DD / heartbeat
   if(!SessionCheck())         return;    // hour window + news blackout
   if(!SpreadGuardCheck())     return;    // spread > 2× rolling median → skip
   if(HasOpenPosition(g_magic)){ PositionManage(); return; }
   if(BracketExpiryTick())     return;    // pending bracket countdown
   if(!LoadBars(g_symbol, PrimaryTimeframe)) return;

   // --- ML vol gate ---
   if(VolThresholdEnabled())
   {
      if(!ComputeFeatures(g_symbol)) return;
      float p = VolFilterRun(g_hVolModel);
      if(p < g_config.vol_threshold) return;
   }

   // --- Trend filter (optional, per-config) ---
   if(!TrendFilterCheck()) return;

   // --- Delegate to signal ---
   SignalResult sig = g_signal.GetSignal();   // {direction, sl_atr_mult, tp_atr_mult, level, atr}
   if(sig.direction == DIR_NONE) return;

   // --- NON-OVERRIDABLE risk sizing + placement ---
   double lot = RiskCoreCalcLot(sig.sl_atr_mult * sig.atr);
   if(lot <= 0) return;
   PlaceBracketOrSingle(sig, lot);            // stops-level-aware
}
```

**Key invariants enforced by the skeleton:**
- The signal returns `{direction, sl_atr_mult, tp_atr_mult, level, atr}` **only**. It cannot size the position (RiskCore owns that), cannot bypass daily loss (KillSwitch owns that), cannot touch spread guard. Mirrors WG4 §1 non-overridable core.
- Symbol name flows from `InpSymbol` into every downstream call. No `_Symbol` literals inside signals (strategies read `g_symbol`).
- ONNX model, feature count, feature spec, and config YAML are all symbol-suffixed — swap `InpSymbol = "GBPUSD"` and everything re-binds at `OnInit`.

---

## 4. Multi-Instrument Null-Test Gate (the M113-killer rule)

**Problem recap (WG3, LESSONS §1.8):** M113 fade_long calibrated on XAU looked brilliant. On XAGUSD/EURUSD/GBPUSD/USDJPY it wiped 3/4 accounts at 96–99.85% DD. Edge was XAU-microstructure-specific, zero generalisation. Discovered at month 5 of research — we want that discovery at week 2 of any new strategy.

**Gate spec (CI-enforced, lives in `python/ebb/validation/multi_instrument.py`):**

1. **Inputs:** candidate strategy config + 4 symbols `{EURUSD (primary), GBPUSD, USDJPY, XAGUSD}`. Same YAML; only `symbol` field swaps. Tick-replay engine runs each with symbol-specific: `_Point`, `tickValue`, ATR scale, session hours per config.
2. **Per-symbol run:** full history (≥ 3y, prefer 6y), MT5 every-tick Model=8 for the primary; tick-replay for secondaries (cheaper, still calibrated ≤ ±5% PF per WG4 §2).
3. **Per-symbol metrics:** PF, trade count, max DD %, year-by-year PF table.
4. **Rejection conditions (ANY triggers REJECT):**
   - ≥ 2 of 4 symbols show **PF < 0.9** on the same config → strategy is XAU-artifact, reject.
   - ≥ 1 of 4 shows **DD > 40%** (account-wipe threshold from M113 experience) → reject regardless of PF elsewhere.
   - Primary PF < 1.3 → reject (can't pass even where it's supposed to work).
5. **Conditional pass ("primary-specialist"):**
   - Primary PF ≥ 1.3, **3 of 4** secondaries show `PF ∈ [0.9, 1.2]` (break-even-ish, no wipeouts) → label `candidate_primary_only`. Not rejected but flagged for manual review and pinned to primary symbol in deployment config.
6. **Full pass ("portable"):**
   - Primary PF ≥ 1.3, ≥ 2 of 3 secondaries PF ≥ 1.2, zero wipeouts → label `candidate_portable`. Allowed to progress to walk-forward gate.
7. **CI report:** `results/<run>/multi_inst_verdict.md` with the 4-symbol table, verdict, rationale. Commit refuses to merge without this file (determinism CI rule, WG4 §3).

**Why this kills M113:** M113 at calibrated VTs produced 4/4 wipeouts (zero trades on gold-params-EURUSD, wipe on calibrated-EURUSD). Gate triggers at step 4a AND 4b simultaneously. Reject at week 2, not month 5.

**Why it doesn't kill legitimate primary-specialists:** EUR/USD can have carry-flow edge that USDJPY doesn't share (JPY inverts carry). Conditional pass at step 5 preserves those strategies with correct labelling. They deploy only on their primary symbol, by config pinning.

---

## 5. Migration Plan — Verbatim vs Rewrite

### Reuse verbatim (≈ 60% of GBB)
- `GBB/Indicators.mqh` entire file — already symbol-agnostic (uses `_Symbol`, `_Point`, buffers). Copy to `EBB/Indicators.mqh` as-is.
- `GBB/TradeMgmt.mqh:37-54` `CalcLotSize` — fully SymbolInfo-driven. Add one line for `MaxRiskUsdPerTrade` cap.
- `GBB/TradeMgmt.mqh:123-156` `HasOpenPosition / SelectMyPosition / CancelBracket` — symbol-agnostic via `PositionGetString(POSITION_SYMBOL) == _Symbol`.
- `GBB/TradeMgmt.mqh:158-236` `ManageOpenTrade` (BE + trailing + time-stop) — all ATR-multiplier-driven, symbol-invariant.
- `GBB/State.mqh` buffers — rename globals but structure stays.
- All 15 strategy `.mqh` modules — they already read from `g_sparams` and call `PlaceSingle/PlaceBracket`. **Port signal logic verbatim**, only wrap each in the new `SignalBase` interface returning `{direction, atr, level, sl_atr_mult, tp_atr_mult}` instead of calling `PlaceSingle` directly.

### Light rewrite (≈ 25%)
- `GBB_Core.mq5` OnInit/OnTick (334 lines) → `EBB_Core.mq5` (~220 lines) per §3 skeleton. Most logic survives; just re-ordered into gates + signal dispatch.
- `GBB/Features.mqh` — 57 of 59 features are price-ratio-normalised, safe. Lines 114-117 (mod50/mod100) → price-relative grid per §2.1. Line 26 `NUM_FEATURES` → runtime. Feature indices read from symbol-specific `features.yaml`.
- `GBB/TradeMgmt.mqh:56-121` `PlaceBracket/PlaceSingle` — add stops-level guard (§2.7), otherwise unchanged.

### Full rewrite (≈ 15%)
- `VolModel` ONNX resource loading → runtime file-based load (§2.3) + metadata validation. New file `EBB/VolFilter.mqh`.
- `EBB/KillSwitch.mqh` — new. Daily loss cap, account DD cap, heartbeat watchdog, consecutive-loss brake. Not in GBB at this rigour.
- `EBB/Session.mqh` — new. Spread guard, news blackout, session hour gating (GBB has hour-gating inline; break out).
- `EBB/Config.mqh` — new. YAML-loader (MQL5 has no native YAML — either CSV or a parser wrapper over `FileReadString`). Parses `configs/strategies/<name>.yaml` and populates `g_tparams`/`g_sparams`.
- `EBB/Telemetry.mqh` — new. Structured CSV logging, one row per decision (bar, signal, filter-state, order-state). Enables determinism CI.
- `EBB/Signals/Base.mqh` — new. Abstract interface `class SignalBase { virtual SignalResult GetSignal() = 0; };`. All 15 ported signals inherit.
- Magic-number `SymbolHash` helper — trivial, 10 lines.

### Week-by-week rewrite execution
**W1:** skeleton + Indicators + TradeMgmt port + SymbolHash + stops-level guard + runtime ONNX load. Smoke-test on EURUSD with null-signal (just heartbeats). Get the event-loop green before touching any signal.

**W1 late:** port FadeLong + MomentumLong + MlEntry through the new `SignalBase` interface. Verify byte-identical output vs GBB when run on XAUUSD with same config (regression gate — if EBB doesn't match GBB on XAU, refactor is broken).

**W2:** KillSwitch + Session + spread guard + Telemetry. Unit test kill-switch with synthetic 2% loss scenario (WG4 merge gate).

**W2 late:** multi-instrument null-test harness (§4). Must be runnable before any W3 strategy experiment.

**W3:** port remaining 12 signals through `SignalBase`. Run EBB on XAUUSD cfg74 as regression — PF=1.4086 to the cent or the port is buggy (GBB_Core achieved this vs GBB_Generic, we should too).

---

## Appendix: File:line cheat-sheet for migration

| Concern | GBB location | EBB target |
|---|---|---|
| ONNX resource | `GBB_Core.mq5:7` | Drop, runtime load in `EBB/VolFilter.mqh` |
| Magic hardcoded | `GBB_Core.mq5:42` | `InpMagicBase + SymbolHash` |
| MaxLotSize=0.10 | `GBB_Core.mq5:43` | `MaxRiskUsdPerTrade` + auto-cap |
| Round-number mod50/100 | `GBB/Features.mqh:114-117` | Price-relative grid |
| NUM_FEATURES=59 | `GBB/Types.mqh:26` | Runtime from model metadata |
| M5 literal | `GBB_Core.mq5:196`, `Indicators.mqh:85`, `Features.mqh:102` | `PrimaryTimeframe` input |
| Stops-level unguarded | `GBB/TradeMgmt.mqh:56-121` | Add `SYMBOL_TRADE_STOPS_LEVEL` check |
| No spread guard | absent | `EBB/Session.mqh::SpreadGuardCheck` |
| No kill switch | absent (daily-loss partial at `GBB_Core.mq5:256`) | `EBB/KillSwitch.mqh` |
| Session hardcoded 7-20 | `GBB_Core.mq5:40-41` | Config YAML (not code) |

**End of design doc.**
