# EuroBigBrain Data Pipeline — Design

**Status:** initial spec, 2026-04-21
**Scope:** EUR/USD primary + GBP/USD, USD/JPY, XAG/USD for multi-instrument null gate (WG3 Gate 8).
**Author:** data layer / validation plumbing.

---

## 1. Why this layer exists

Every downstream module — MI scan, bootstrap nulls, walk-forward retrain, cost-stress sim, ONNX feature builder — reads the same M5 OHLCV. If those bars are wrong (silent gaps, broker Saturday bars, DST jumps mis-handled, duplicate timestamps from MT5's re-download quirks) every finding above it is suspect. GoldBigBrain's `wg3_validation_pipeline.md` puts this under "build infrastructure FIRST". The two scripts in `python/data/` are that infrastructure.

> MT5 every-tick backtests don't need this — MT5 loads its own internal cache. Everything Python-side (sim, nulls, features, regime splits) does.

## 2. Components

### `fetch_history.py`
Pulls M5 OHLCV from the local MT5 terminal via `MetaTrader5.copy_rates_range`, chunked in 180-day windows (the MT5 Python lib has an undocumented ~100k-row ceiling per call — chunking is defensive).

- Default symbols: `EURUSD, GBPUSD, USDJPY, XAGUSD`.
- Default span: `2020-01-03 .. 2026-04-10` (matches GoldBigBrain's XAU M5 corpus so validation horizons line up).
- Per-symbol output: `data/{SYMBOL}_M5_{y0}_{y1}.parquet` with columns `time, open, high, low, close, tick_volume, spread, real_volume`.
- Handles `symbol_select` if the pair isn't in Market Watch, de-dupes across chunk boundaries, snappy-compresses parquet.

### `quality_audit.py`
`audit_dataset(parquet_path) -> dict` runs five deterministic checks and emits two CSVs next to the parquet:
- `{symbol}_audit_findings.csv` — one row per anomaly (time, kind, delta_min).
- `{symbol}_audit_summary.csv` — one-row headline (bars, completeness, counts by kind).

Checks:
1. **Bar count vs expected** — `~288 M5 bars/day × trading_days (5/7 of span)`. Flags completeness below 90%.
2. **Inter-bar gap classification** — a non-5 min delta is either a legitimate weekend (47–65 h), a DST spring-forward (55 min, logged but OK), a backward/zero delta (DST fall-back / broker resend), or a real intraday gap (>10 min).
3. **OHLC sanity** — `H >= max(O,C,L)` and `L <= min(O,C,H)`. Single violation fails the audit.
4. **Duplicate timestamps** — broker feed glitches + DST-fall repeat bars.
5. **Zero-volume bars** — `tick_volume == 0` often means no real fills in that M5; informational for forex where thin liquidity at Sunday open is normal, load-bearing for commodities like XAGUSD.
6. **Saturday bars** — should never exist on a healthy FX feed; if present the broker is mixing aggregated weekend data.

Non-zero exit on `completeness < 0.90` or any OHLC violation, so CI and orchestrators can `&&`-chain.

### CLI
```
python fetch_history.py                                      # full fetch
python fetch_history.py --symbols EURUSD --start 2020-01-03
python quality_audit.py data/EURUSD_M5_2020_2026.parquet
python quality_audit.py "data/*.parquet"                     # multi-symbol
```

## 3. Expected shape (EUR/USD baseline)

FX runs 24×5. Sunday ~22:00 UTC open to Friday ~22:00 UTC close. ~277 M5 bars/day × ~1,620 trading days over 6 years ≈ **~450k M5 bars** for EUR/USD. Deviation > 5% triggers investigation — usually missing weeks or DST boundary misses.

## 4. Known forex-vs-gold pitfalls the auditor catches

- **DST asymmetry.** EUR/USD sits across Europe + US DST. Twice a year Europe and the US shift on different Sundays, producing a 3-week window with irregular hour offsets. Gold trades on one feed; forex sees both — hence two DST spring events per year, each legitimate.
- **Asia open gap.** Sunday ~21:00 UTC the first few M5 bars often have zero tick volume; gold rarely does. `zero_vol_bars` is expected to be non-trivial on EURUSD/GBPUSD, higher still on XAGUSD (London-only).
- **News spikes ≠ gaps.** A 5 min bar with huge range around NFP is *not* a gap — don't mistake range for missing data. The gap checker looks at inter-bar time deltas, not bar ranges. Correct by construction.

## 5. Integration contract for downstream modules

Any sim, null, walk-forward, or feature module MUST:
1. Call `audit_dataset(path)` on load; abort if `completeness < 0.95` or `ohlc_violations > 0`.
2. Sort by `time`, drop duplicates before use (the auditor reports, it doesn't mutate — mutation is the consumer's call).
3. Re-run audit whenever `fetch_history.py` refreshes the parquet; overwrite summary CSV atomically.

## 6. What this layer intentionally does *not* do

- No resampling to M1/M15/H1 — downstream scripts own their own resampling so the audit stays a pure check of the canonical M5 source.
- No broker-spread override — we report the MT5-reported `spread` column as-is; cost stress (`validation/cost_stress.py`) is where spread perturbation happens.
- No tick replay — that's `phase2a_tick_replay` on GoldBigBrain, to be ported separately when EUR/USD calibration work starts.
