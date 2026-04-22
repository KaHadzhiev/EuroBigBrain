# EBB_TripleBarrier — Deploy Readme

**EA:** `EBB_TripleBarrier.mq5`
**Magic:** `26042201` (EBB-2026-04-22 v1)
**Strategy spec:** `project_ebb_DEPLOY_CANDIDATE_h10` (h=10, SL=0.7×ATR, TP=2.0×ATR, P>0.44)

## 1. Compile

MetaEditor CLI (Windows Bash):

```bash
MSYS_NO_PATHCONV=1 \
  "C:/MT5-Instances/Instance1/MetaEditor64.exe" \
  /compile:"C:\\Users\\kahad\\IdeaProjects\\EuroBigBrain\\mql5\\EBB_TripleBarrier.mq5" \
  /log:"C:\\Users\\kahad\\IdeaProjects\\EuroBigBrain\\mql5\\EBB_TripleBarrier.log"
cat "C:/Users/kahad/IdeaProjects/EuroBigBrain/mql5/EBB_TripleBarrier.log"
```

Or just run the batch:

```
C:\Users\kahad\IdeaProjects\EuroBigBrain\mql5\EBB\TripleBarrierCompile.bat
```

Look for `0 errors, 0 warnings` in the log. The ONNX file must be next to the `.mq5` (it is — `eur_tb_h10.onnx` already sits in the same folder) — it is embedded via `#resource` at compile time, so it is baked into the compiled `.ex5`.

## 2. Deploy to MT5

After compilation succeeds you get `EBB_TripleBarrier.ex5` next to the `.mq5`.

Copy it into each MT5 instance's Experts folder:

```
C:\MT5-Instances\Instance1\MQL5\Experts\EBB_TripleBarrier.ex5
```

(Repeat for `Instance2` … `Instance6` as needed. The ONNX model is embedded inside the `.ex5`, so nothing else needs to be copied. If you ever re-load the raw `.onnx` file at runtime instead of via resource, place it at `MQL5\Files\eur_tb_h10.onnx`.)

## 3. Required symbols (MarketWatch subscription)

The EA computes a DXY proxy and cross-asset returns. These symbols **must be subscribed** in MarketWatch on the broker profile you test against:

- `EURUSD` — the trading symbol (chart)
- `USDJPY`, `USDCHF`, `GBPUSD` — for DXY proxy (`log(USDJPY) + log(USDCHF) − log(GBPUSD)`) and for EURGBP synthetic (`EURUSD / GBPUSD`)
- `XAUUSD` — for `xau_ret_5`

`OnInit` will hard-fail (INIT_FAILED) if any of these is absent unless you set the input `RequireCrossSymbols = false`. With `false` the missing cross-asset features fall back to 0.0 (expect degraded performance).

For live Vantage: all four are standard STP symbols — just right-click MarketWatch → "Show All". For Strategy Tester: the tester auto-synchronizes referenced symbols, but you MUST enable the option "Use every tick for all symbols" in the tester settings (see below) or the cross-asset CopyClose calls return empty.

## 4. Strategy Tester — recommended settings

| Setting | Value |
|---|---|
| Expert | `EBB_TripleBarrier` |
| Symbol | `EURUSD` |
| Period | `M5` |
| Modeling | **Every tick based on real ticks** (critical — close-only will misrepresent SL/TP fills) |
| Spread | `Real` (or fixed `7` for Vantage Standard STP 0.7 pip) |
| Deposit | `10000 USD` (or whatever you prefer; 0.6% risk floats with balance) |
| Leverage | `1:500` (match Vantage) |
| Use Date | `true` |
| Date from | `2025-01-01` (model training window started here) |
| Date to | `2026-04-22` (today) |
| Forward test | optional — cut at 2026-02-01 for 2.5mo forward |
| Optimization | OFF for the first run |
| Multi-symbol sync | Make sure USDJPY, USDCHF, GBPUSD, XAUUSD history is downloaded *before* launching the tester. MT5 will refuse or return empty series otherwise. |

### Suggested command line (run from the MT5 instance root)

```bash
"C:\MT5-Instances\Instance1\terminal64.exe" \
  /config:"C:\Users\kahad\IdeaProjects\EuroBigBrain\mql5\EBB_TripleBarrier_test.ini"
```

Create a `.ini` (UTF-16LE + BOM per `reference_mt5_config_flag`) with a `[Tester]` section pointing at `Expert=EBB_TripleBarrier` and the period `2025.01.01–2026.04.22`.

## 5. Input parameters

| Name | Default | Meaning |
|---|---|---|
| `ProbThreshold` | `0.44` | P(up) cutoff for a BUY signal |
| `SL_ATR_Mult` | `0.7` | SL distance = this × ATR(14) |
| `TP_ATR_Mult` | `2.0` | TP distance = this × ATR(14) |
| `MaxHoldBars` | `10` | Close at market after this many M5 bars if no SL/TP hit (≈ 50 min) |
| `RiskPercent` | `0.6` | % of balance risked per trade |
| `MaxLotSize` | `0.10` | Hard cap on lot size after risk-sizing |
| `MaxTradesPerDay` | `20` | Daily trade count limit |
| `DailyLossCapPct` | `5.0` | Stop opening new trades once −5% booked today |
| `MagicNumber` | `26042201` | Position tag (per memory format: YYMMDD##) |
| `DebugEveryNTicks` | `100` | Print probability + state every N OnTick calls |
| `RequireCrossSymbols` | `true` | INIT_FAILED if any cross symbol absent |

## 6. Feature parity caveats (vs. training script)

The python training pipeline (`build_eurusd_features.py`) uses pandas. MQL5 can't exactly replicate every pandas behavior — here are the known mismatches, ranked by expected impact:

1. **EWM convention**: pandas `Series.ewm(span=N).mean()` defaults to `adjust=True` (weighted-average form). MQL5 uses the classic exponential recursion (equivalent to `adjust=False`). The first ~3×span bars will differ slightly; steady-state converges to the same value. Affects `ema20_zdist`, `ema50_zdist`, `h1_ema50_slope`. Impact: small (~1–2% of feature value in steady state). Same as GBB EA convention.
2. **H1 resampling**: python uses `pd.resample('1H').last()` which snaps to wall-clock hour boundaries. EA samples every 12th M5 bar going back, which only approximates the same thing if the chart timezone aligns with the broker's hour boundary. Broker server time drift → up to 1-bar slip in `h1_ema50_slope`. Impact: low (feature is just sign of slope).
3. **Session grouping for `bar_of_sess_mom`**: python groups by `(hour_string + '_' + date_string)`, so "session open" is the open of the first bar in the *current clock hour on the current date*. EA walks back through the in-memory buffer stopping when hour/date differs. Same logic — but if there's any gap in the bar buffer (weekend, outage) it may produce a slightly different anchor. Impact: low to medium on the first bar after a gap.
4. **ATR(14) vs pandas rolling mean**: python uses plain `.rolling(14).mean()` of true range; EA uses Wilder's smoothing (standard MT5 convention, and what `iATR` returns). The ratio of these two is close to 1.0 in steady state but not identical. **This is a known delta** — could shift `atr14_norm` by 2-5% in high-vol bars. If OOS performance disappoints, swap to a simple-mean ATR to match training exactly.
5. **Timezone**: python uses the raw timestamp index from the CSV (broker/MT5 export timezone). The EA uses `buf_time[idx]` which is broker server time. These should match if the training CSV came from the same broker; if not, hour/session features will be offset by (chart_tz − training_tz). **Check `EURUSD_M5_full.csv` timezone before deploying.**
6. **vol_of_vol** uses rolling-20 over the ATR series. The ATR series is recomputed from scratch for each lookback bar (O(40×period) per bar), which is correct but costly. Expect ~1 ms per feature-compute call — acceptable at M5.
7. **EURGBP synthetic**: we build `EURUSD / GBPUSD` using the closes aligned to the EURUSD bar times. Python does `reindex(ffill)`, which is essentially the same — but if GBPUSD bar times don't align exactly with EURUSD (different quote server), we may drop bars. Fallback is zero, which is not great. Monitor for runs of `eurgbp_ret_5 = 0` in logs.
8. **`near_ny_fix` boundary**: python uses `((h==15)&(m>=55))|((h==16)&(m<=5))` — EA matches exactly.
9. **`dow`**: python uses pandas `.dayofweek` (Monday=0). EA maps MT5's `day_of_week` (Sunday=0) with `(dow_mt + 6) % 7`. Matches.
10. **Feature 16 `h1_ema50_slope`**: python applies `np.sign` (returns 1/0/−1 float). EA returns float 1/0/−1. Matches.

**Expected impact on OOS PF**: −10% to −15% of the in-sample PF=2.76 due to compound feature noise + broker spread realism. A live PF of 1.6–1.8 would be excellent. Below 1.3 would mean a real parity bug to hunt down.

## 7. Debugging checklist if first run is bad

- Pull the Experts log for the tester run and grep for `EBB_TB debug` — verify probabilities are spread across [0,1], not stuck near 0 or 0.5.
- If prob is always ~0.5: features are garbage (likely cross-symbol data missing or column order wrong).
- If prob varies but no trades: threshold too high or daily caps tripping. Drop `ProbThreshold` to 0.40 for a sanity check.
- If many trades but all losers: SL/TP sign flipped or ATR computation off — check bar direction of the first few BUY entries against the chart.
- Compare a single bar's feature vector against the python parquet (`data/eurusd_features.parquet`) for the same timestamp; differences > 5% per feature mean something is off.

## 8. Files

| File | Purpose |
|---|---|
| `EBB_TripleBarrier.mq5` | EA source |
| `eur_tb_h10.onnx` | Trained LightGBM (685 KB), embedded via `#resource` |
| `EBB_TripleBarrier_README.md` | This document |
| `EBB/TripleBarrierCompile.bat` | Compile helper |

## 9. Next TODOs (after first MT5 run)

1. Every-tick MT5 real-tick run 2025-01-01 → 2026-04-22 with Vantage spread.
2. Null-permutation test on shuffled feature matrix (target ratio ≥ 5×).
3. Bootstrap equity DD (Monte Carlo 1000 resamples).
4. Threshold sweep 0.40 → 0.50 step 0.01 for PF/tr-per-month frontier.
5. Walk-forward yearly when ≥ 3yr data available.
