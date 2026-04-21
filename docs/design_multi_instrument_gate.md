# Design — Multi-Instrument Null Gate (WG3 Gate 8, the M113 Killer)

**Implementation:** `python/validation/multi_instrument_null.py` (405 lines)
**Tests:** `python/validation/tests/test_multi_instrument_null.py` (classifier + M113 replay)
**Author:** EuroBigBrain validation pipeline
**Status:** Skeleton complete; needs live MT5 smoke test before first candidate run

---

## Why this gate exists

2026-04-21 — M113 (XAU "structural mean-reversion" winner, PF=1.41, 7/7
years green) wiped 96–99.85% on XAGUSD/GBPUSD/USDJPY and took **zero
trades** on EURUSD. The edge was XAU-microstructure overfit, not
structural. Gate 8 runs every EBB candidate unchanged on EUR + 3 tests
via MT5 every-tick Model=8; failure is a first-class reject — no further
gates, no iteration.

## Instruments chosen

| Role | Symbol | Rationale |
|---|---|---|
| Base | EURUSD | Home turf — strategy tuned here |
| Test | GBPUSD | Same asset class (FX major), different liquidity profile and ECB vs BoE event calendar — pure structural-generalization probe |
| Test | USDJPY | Different tick size and different correlated-flow regime (carry pair, BOJ stability) — catches strategies that depend on EUR-ish mean-reversion dynamics |
| Test | XAGUSD | Non-FX, metal — stretches the ATR scale by ~3× and exposes any residual gold-microstructure assumption |

Set deliberately mirrors the M113 failure set — if an EBB candidate
generalizes where M113 collapsed, that is meaningful evidence. WG3's
original spec named AUDUSD/USDCHF; JPY+XAG was chosen for highest
diagnostic value against the exact scenario we are trying to catch.

## 3-tier classification

Implemented in `classify()`:

- **Tier A (ACCEPT):** Base passes AND all 3 tests pass (PF≥1.0, DD≤40%,
  nonzero trades). Strategy is structural.
- **Tier B (CONDITIONAL):** 2/3 tests pass, nothing wiped. Requires a
  pre-registered economic rationale for partial generalization (ECB
  calendar, London/NY overlap specificity, EUR-funded carry mechanics).
  No post-hoc "oh that's why it works only on EUR".
- **Tier C (REJECT):** Any test instrument DDs > 40%, OR 2+ test
  instruments fail PF≥1.0. This is the M113 fingerprint.

Zero-trades on a test instrument is folded into the fail condition — the
M113 EURUSD result (no trades triggered at all) must count as a fail,
not as a neutral unknown. Retuning VT per instrument to force trades is
explicitly flagged in warnings as "admission of overfit" per the
2026-04-21 kill of Option 3.

## ATR-relative-scaling handling

Params are ATR-ratios (`SL_ATR_Mult`, `TP_ATR_Mult`, `BracketOffset`,
`VolThreshold`) so they auto-scale with instrument ATR — that's the
design intent of GBB_Core. Three caveats:

1. **Gold-ONNX VolThreshold is not transferable.** A VT from a gold-trained
   ONNX model filters on vol-features whose distribution is
   instrument-specific (wavelet energies, entropy, atr-ratios all shift).
   The code emits a warning when `strategy_config["vol_model_source"] ==
   "gold_onnx"` — results reflect gated-by-gold behaviour, not a native
   vol gate.
2. **Session hours are not ATR-relative.** Zero-trade outcomes on a test
   instrument are counted as fails (M113 EUR=0-trades signature), not
   neutral unknowns.
3. **Spread/slippage per-instrument.** Gate 6 (cost stress) owns that;
   Gate 8 uses broker-raw spread.

## Runtime envelope

MT5 every-tick Model=8 on M5 2020-01-03 → 2026-04-10 is ~20–40 min per
instrument. 4 sequential ≈ 80–160 min. Parallelising across 3 Win
instances (hardware playbook) brings wall-clock to ~30–50 min per
candidate.

## Invocation

```
python python/validation/multi_instrument_null.py \
    --config configs/<strategy>.json \
    --base EURUSD --from 2020.01.03 --to 2026.04.10
```

Exit codes: 0 = A/B accept, 1 = C reject, 2 = runtime error. Dry-run
mode (`--dry-run`) builds a fabricated passing verdict and is used by
CI and as a smoke test before scheduling expensive MT5 runs.
