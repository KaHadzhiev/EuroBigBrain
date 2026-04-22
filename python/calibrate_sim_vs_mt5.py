#!/usr/bin/env python3
"""Compare Python sim PF vs MT5 every-tick PF per ProbThreshold; learn isotonic correction.

Inputs:
  results/sim_eurusd_6yr_threshold_sweep.csv          (from sim_eurusd_6yr.py)
  runs/mt5_thr_sweep.json                             (from run_mt5_thr_sweep.py;
                                                       expects per-thr PF/trades/DD)

Pipeline (BEAT-GBB strategy):
  1. Per-threshold delta = (sim_pf - mt5_pf) / mt5_pf      [target: |delta| < 5%]
  2. Industry multi-metric gate (Spearman rank-corr + KS PnL + Sharpe band + expectancy band)
     -- ported from GBB calibration_metrics_v2.py
  3. Fit isotonic regressor on (sim_pf -> mt5_pf) so future sim runs can be auto-corrected
  4. (Optional) Fit GBR on (sim_pf, sim_trades) -> mt5_pf for richer correction

Outputs:
  results/calibrate_sim_vs_mt5.csv          (paired thresholds + delta)
  results/calibrate_metrics.json            (multi-metric verdict)
  results/eurusd_isotonic_calibrators.joblib (sim_iso + sim_gbr models)

How this beats GBB (target <5% gap vs GBB's 11%):
  (a) Isotonic regression learns a monotone sim_pf -> mt5_pf mapping. Even if our raw
      sim is biased, the fitted f() removes systematic skew. (See GBB isotonic_calibration.py)
  (b) Vantage spread MARKUP modeled explicitly in sim (sim_eurusd_6yr.py uses 1.0 pip vs
      Dukascopy ECN ~0.5 pip). GBB only modeled session-based spread, not broker markup.
  (c) Slippage tuned to EURUSD (0.2 pips per fill) — much tighter than GBB's XAUUSD (1.5 pts).
"""
import argparse
import sys
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIM_CSV = REPO / "results" / "sim_eurusd_6yr_threshold_sweep.csv"
MT5_JSON = REPO / "runs" / "mt5_thr_sweep.json"
PAIRED_CSV = REPO / "results" / "calibrate_sim_vs_mt5.csv"
METRICS_JSON = REPO / "results" / "calibrate_metrics.json"
ISO_OUT = REPO / "results" / "eurusd_isotonic_calibrators.joblib"

# Calibration gates (EBB target: BEAT GBB's 11% gap with 5%)
GATE_PF_DELTA = 0.05            # per-threshold abs(sim_pf - mt5_pf)/mt5_pf <= 5%
GATE_SPEARMAN_MIN = 0.80
GATE_KS_MAX = 0.15
GATE_SHARPE_BAND = 0.10
GATE_BAND_FRAC_MIN = 0.70


def _safe(v) -> float:
    try:
        if v in (None, "", "?"):
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_sim() -> list[dict]:
    import pandas as pd
    if not SIM_CSV.exists():
        print(f"FATAL: {SIM_CSV} not found - run sim_eurusd_6yr.py first", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(SIM_CSV)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "threshold": float(r["threshold"]),
            "sim_pf": float(r["pf"]),
            "sim_trades": int(r["trades"]),
            "sim_pnl_pips": float(r["pnl_pips"]),
            "sim_dd_pips": float(r["max_dd_pips"]),
        })
    return rows


def load_mt5() -> list[dict]:
    if not MT5_JSON.exists():
        print(f"FATAL: {MT5_JSON} not found - run run_mt5_thr_sweep.py first", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(MT5_JSON.read_text(encoding="utf-8"))
    rows = []
    for r in raw:
        thr = float(r.get("thr", float("nan")))
        pf  = _safe(r.get("pf"))
        tr  = _safe(r.get("trades"))
        np_ = _safe(str(r.get("net_profit", "0")).replace(" ", "").replace("\xa0", ""))
        dd  = _safe(str(r.get("dd_equity", "0")).split()[0].replace(" ", "").replace("\xa0", ""))
        rows.append({
            "threshold": thr, "mt5_pf": pf, "mt5_trades": int(tr) if math.isfinite(tr) else 0,
            "mt5_net_profit": np_, "mt5_dd_equity": dd,
        })
    return rows


def pseudo_sharpe(pf: float, trades: int) -> float:
    if trades <= 0 or not math.isfinite(pf) or pf <= 0:
        return float("nan")
    return ((pf - 1.0) / (pf + 1.0)) * math.sqrt(trades)


def main() -> int:
    sim = load_sim()
    mt5 = load_mt5()

    # Pair by threshold (round to 2 decimals for matching)
    sim_by = {round(r["threshold"], 2): r for r in sim}
    mt5_by = {round(r["threshold"], 2): r for r in mt5}
    common = sorted(set(sim_by) & set(mt5_by))
    print(f"sim thresholds: {sorted(sim_by)}")
    print(f"mt5 thresholds: {sorted(mt5_by)}")
    print(f"common: {common}")

    if not common:
        print("FATAL: no common thresholds between sim and mt5", file=sys.stderr)
        return 1

    paired = []
    print(f"\n{'thr':>6} {'sim_pf':>7} {'mt5_pf':>7} {'delta':>7} {'sim_tr':>7} {'mt5_tr':>7} "
          f"{'verdict':>8}")
    for thr in common:
        s = sim_by[thr]; m = mt5_by[thr]
        if not (math.isfinite(s["sim_pf"]) and math.isfinite(m["mt5_pf"]) and m["mt5_pf"] > 0):
            continue
        delta = (s["sim_pf"] - m["mt5_pf"]) / m["mt5_pf"]
        verdict = "PASS" if abs(delta) <= GATE_PF_DELTA else "FAIL"
        paired.append({
            "threshold": thr,
            "sim_pf": s["sim_pf"], "mt5_pf": m["mt5_pf"],
            "delta_rel": round(delta, 4),
            "sim_trades": s["sim_trades"], "mt5_trades": m["mt5_trades"],
            "sim_pnl_pips": s["sim_pnl_pips"], "mt5_net_profit": m["mt5_net_profit"],
            "verdict": verdict,
        })
        print(f"{thr:>6.2f} {s['sim_pf']:>7.3f} {m['mt5_pf']:>7.3f} {delta:>+7.2%} "
              f"{s['sim_trades']:>7} {m['mt5_trades']:>7} {verdict:>8}")

    # Save pairing
    PAIRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(PAIRED_CSV, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)
    print(f"\nSaved {PAIRED_CSV}")

    # Multi-metric verdict (Spearman/KS/Sharpe band)
    metrics = {"thresholds_paired": len(paired), "gates": {
        "pf_delta_max": GATE_PF_DELTA, "spearman_min": GATE_SPEARMAN_MIN,
        "ks_max": GATE_KS_MAX, "sharpe_band": GATE_SHARPE_BAND,
        "band_frac_min": GATE_BAND_FRAC_MIN,
    }}
    if len(paired) >= 5:
        try:
            from scipy import stats as _scistats
            sim_pf = [p["sim_pf"] for p in paired]
            mt5_pf = [p["mt5_pf"] for p in paired]
            sim_sh = [pseudo_sharpe(p["sim_pf"], p["sim_trades"]) for p in paired]
            mt5_sh = [pseudo_sharpe(p["mt5_pf"], p["mt5_trades"]) for p in paired]
            rho, p_rho = _scistats.spearmanr(sim_pf, mt5_pf)
            ks_stat, ks_p = _scistats.ks_2samp(sim_pf, mt5_pf)
            inside = sum(1 for s, m in zip(sim_sh, mt5_sh)
                         if math.isfinite(s) and math.isfinite(m) and m != 0
                         and abs(s - m) / abs(m) <= GATE_SHARPE_BAND)
            used = sum(1 for s, m in zip(sim_sh, mt5_sh)
                       if math.isfinite(s) and math.isfinite(m) and m != 0)
            sharpe_frac = (inside / used) if used else float("nan")
            metrics["spearman"] = {"rho": float(rho), "p": float(p_rho),
                                   "pass": bool(rho >= GATE_SPEARMAN_MIN)}
            metrics["ks_pf"] = {"stat": float(ks_stat), "p": float(ks_p),
                                "pass": bool(ks_stat <= GATE_KS_MAX)}
            metrics["sharpe_band"] = {"frac": sharpe_frac, "inside": inside, "used": used,
                                      "pass": bool(math.isfinite(sharpe_frac) and
                                                   sharpe_frac >= GATE_BAND_FRAC_MIN)}
        except ImportError:
            print("[warn] scipy not available -- skipping multi-metric gates", file=sys.stderr)
    else:
        print(f"[warn] only {len(paired)} paired rows -- skipping multi-metric gates")

    # Fit isotonic correction (sim_pf -> mt5_pf)
    iso_saved = False
    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        from sklearn.ensemble import GradientBoostingRegressor
        import joblib
        if len(paired) >= 5:
            sim_pf = np.array([p["sim_pf"] for p in paired])
            mt5_pf = np.array([p["mt5_pf"] for p in paired])
            sim_tr = np.array([p["sim_trades"] for p in paired])
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
            iso.fit(sim_pf, mt5_pf)
            iso_pred = iso.predict(sim_pf)
            iso_mae = float(np.mean(np.abs(iso_pred - mt5_pf)))
            metrics["isotonic"] = {"mae": iso_mae, "n": int(len(paired))}
            print(f"\nIsotonic fit: MAE = {iso_mae:.4f} on {len(paired)} pairs")
            bundle = {"sim_iso": iso, "n_train": int(len(paired))}
            if len(paired) >= 8:
                X = np.column_stack([sim_pf, sim_tr])
                gbr = GradientBoostingRegressor(n_estimators=80, max_depth=2,
                                                learning_rate=0.05, subsample=0.8,
                                                random_state=42)
                gbr.fit(X, mt5_pf)
                gbr_pred = gbr.predict(X)
                gbr_mae = float(np.mean(np.abs(gbr_pred - mt5_pf)))
                bundle["sim_gbr"] = gbr
                metrics["gbr"] = {"mae": gbr_mae, "n": int(len(paired))}
                print(f"GBR fit: MAE = {gbr_mae:.4f} (uses sim_pf + sim_trades)")
            ISO_OUT.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, ISO_OUT)
            print(f"Saved calibrators -> {ISO_OUT}")
            iso_saved = True
    except ImportError as e:
        print(f"[warn] sklearn/joblib missing -- isotonic skipped: {e}", file=sys.stderr)

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved {METRICS_JSON}")

    # Final verdict line
    fails = [p for p in paired if p["verdict"] == "FAIL"]
    print("\n" + "=" * 60)
    print(f"FINAL: {len(paired)-len(fails)}/{len(paired)} thresholds within +/-{GATE_PF_DELTA*100:.0f}% PF")
    if not fails:
        print("CALIBRATION PASS -- sim within target gap of MT5 every-tick.")
    else:
        worst = max(fails, key=lambda p: abs(p["delta_rel"]))
        print(f"CALIBRATION FAIL on {len(fails)} thresholds. Worst: thr={worst['threshold']} "
              f"delta={worst['delta_rel']:+.1%}")
        print("Root-cause hints:")
        print("  - delta > 0  (sim too optimistic) -> increase VANTAGE_SPREAD_PIPS or SLIPPAGE_PIPS")
        print("  - delta < 0  (sim too pessimistic) -> check fill timing (entry on next-bar OPEN?)")
        print("  - low trade count delta -> check feature/probability alignment between python+MT5")
    if iso_saved:
        print(f"Isotonic calibrator saved -- can be used to auto-correct future sim PFs.")
    return 0 if not fails else 2


def smoke_test() -> int:
    """Fit isotonic on 5 fake (sim_pf, mt5_pf) pairs; verify monotone output."""
    print("[smoke] calibrate_sim_vs_mt5")
    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
    except ImportError as e:
        print(f"[smoke] FAIL: missing dep {e}")
        return 1
    # Sim consistently overshoots MT5 (typical EBB pattern). 5 thresholds.
    sim_pf = np.array([1.20, 1.40, 1.60, 1.80, 2.00])
    mt5_pf = np.array([1.05, 1.18, 1.30, 1.42, 1.55])  # monotone-up
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(sim_pf, mt5_pf)
    pred = iso.predict(sim_pf)
    monotone = bool(np.all(np.diff(pred) >= -1e-9))
    mae = float(np.mean(np.abs(pred - mt5_pf)))
    mae_ok = mae < 0.05  # tight on perfectly monotone synthetic data
    # Verify gates module-constants are EUR-correct (NOT inherited from gold's looser gates)
    gates_ok = (GATE_PF_DELTA == 0.05 and GATE_SPEARMAN_MIN == 0.80
                and GATE_KS_MAX == 0.15)
    if monotone and mae_ok and gates_ok:
        print(f"[smoke] PASS: isotonic monotone, MAE={mae:.4f}, gates=EUR-tight")
        return 0
    print(f"[smoke] FAIL: monotone={monotone} mae={mae:.4f} gates={gates_ok}")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", help="run smoke test and exit")
    args = ap.parse_args()
    if args.smoke_only:
        sys.exit(smoke_test())
    sys.exit(main())
