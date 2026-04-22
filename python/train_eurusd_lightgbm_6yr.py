#!/usr/bin/env python3
"""Train LightGBM on 6yr EURUSD features with Triple-Barrier TRUE-hilo labels.

Train window: 2020-01 .. 2024-01  (4yr)
Test window:  2024-01 .. 2026-04  (~2.3yr OOS)

Triple-Barrier params (matches deployed EBB_TripleBarrier.mq5):
  h = 10 bars  (~50 min on M5)
  SL = 0.7 x ATR14
  TP = 2.0 x ATR14

Output:
  models/eur_tb_h10_6yr.onnx              (ONNX, zipmap=False -- CRITICAL for MQL5 EA)
  models/eur_tb_h10_6yr_feature_list.txt  (one feature name per line, in ONNX input order)
  runs/train_eurusd_lightgbm_6yr.json     (per-fold AUC + OOF predictions metadata)

Also dumps OOF probabilities to data/eurusd_oof_probs_6yr.parquet for sim_eurusd_6yr.py.
"""
import argparse
import os
import sys
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
FEATURES_PATH = REPO / "data" / "eurusd_features_6yr.parquet"
M5_PARQUET = REPO / "data" / "eurusd_m5_2020_2026.parquet"
MODELS_DIR = REPO / "models"
OUT_ONNX = MODELS_DIR / "eur_tb_h10_6yr.onnx"
OUT_FEATLIST = MODELS_DIR / "eur_tb_h10_6yr_feature_list.txt"
OUT_OOF = REPO / "data" / "eurusd_oof_probs_6yr.parquet"
OUT_RUN = REPO / "runs" / "train_eurusd_lightgbm_6yr.json"

H_BARS, SL_MULT, TP_MULT = 10, 0.7, 2.0
TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-01-01"
TEST_END    = "2026-04-13"

PARAMS = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 31, "learning_rate": 0.03,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
    "min_child_samples": 200, "lambda_l1": 0.1, "lambda_l2": 0.1,
    "verbose": -1, "num_threads": 4,
}
NUM_ROUNDS = 300


def tb_label_TRUE(close: np.ndarray, atr_arr: np.ndarray,
                  th: np.ndarray, tl: np.ndarray,
                  h: int, sl_m: float, tp_m: float) -> np.ndarray:
    """Triple-Barrier label using TRUE bar high/low. 1 if TP hit first, 0 otherwise."""
    n = len(close)
    label = np.zeros(n, dtype=np.int8)
    for i in range(n - h):
        tp_px = close[i] + tp_m * atr_arr[i]
        sl_px = close[i] - sl_m * atr_arr[i]
        for j in range(1, h + 1):
            if tl[i + j] <= sl_px:
                label[i] = 0
                break
            if th[i + j] >= tp_px:
                label[i] = 1
                break
    return label


def main() -> int:
    t0 = time.time()
    if not FEATURES_PATH.exists():
        print(f"FATAL: {FEATURES_PATH} not found - run build_eurusd_features_6yr.py first", file=sys.stderr)
        return 1
    if not M5_PARQUET.exists():
        print(f"FATAL: {M5_PARQUET} not found - run dukascopy_to_m5_bars.py first", file=sys.stderr)
        return 1

    print(f"Loading features: {FEATURES_PATH}")
    f = pd.read_parquet(FEATURES_PATH)
    print(f"  {len(f):,} rows x {len(f.columns)} cols  span={f.index.min()} -> {f.index.max()}")

    print(f"Loading raw M5 for TRUE high/low: {M5_PARQUET}")
    raw = pd.read_parquet(M5_PARQUET)
    raw["time"] = pd.to_datetime(raw["time"])
    raw = raw.set_index("time").sort_index()

    common = f.index.intersection(raw.index)
    f = f.loc[common]
    raw = raw.loc[common]

    close   = f["close"].values
    atr_arr = (f["atr14_norm"].values * close)
    th      = raw["high"].values
    tl      = raw["low"].values

    feature_cols = [c for c in f.columns if not c.startswith("target_") and c != "close"]
    print(f"  feature count: {len(feature_cols)}")

    print("Computing Triple-Barrier labels (TRUE hi/lo)...")
    y = tb_label_TRUE(close, atr_arr, th, tl, H_BARS, SL_MULT, TP_MULT)
    y = pd.Series(y, index=f.index)

    # Mask: drop rows where any feature NaN, ATR NaN, or label window incomplete
    fwd_hi = pd.Series(th, index=f.index).rolling(H_BARS).max().shift(-H_BARS)
    fwd_lo = pd.Series(tl, index=f.index).rolling(H_BARS).min().shift(-H_BARS)
    mask = (f[feature_cols].notna().all(axis=1)
            & pd.Series(atr_arr, index=f.index).notna()
            & fwd_hi.notna() & fwd_lo.notna())
    Xs = f.loc[mask, feature_cols]
    ys = y[mask]
    print(f"  usable rows: {len(Xs):,}, label balance (P=1): {ys.mean():.3f}")

    # Time-based split
    train_mask = (Xs.index >= TRAIN_START) & (Xs.index < TRAIN_END)
    test_mask  = (Xs.index >= TRAIN_END)   & (Xs.index <= TEST_END)
    Xtr, ytr = Xs.loc[train_mask], ys.loc[train_mask]
    Xte, yte = Xs.loc[test_mask],  ys.loc[test_mask]
    print(f"  TRAIN rows: {len(Xtr):,}  TEST rows: {len(Xte):,}")

    print(f"\nTraining LightGBM ({NUM_ROUNDS} rounds)...")
    d_tr = lgb.Dataset(Xtr, label=ytr)
    model = lgb.train(PARAMS, d_tr, num_boost_round=NUM_ROUNDS,
                      callbacks=[lgb.log_evaluation(0)])

    pte = model.predict(Xte)
    auc_test = float(roc_auc_score(yte, pte)) if yte.nunique() > 1 else float("nan")
    ptr = model.predict(Xtr)
    auc_train = float(roc_auc_score(ytr, ptr)) if ytr.nunique() > 1 else float("nan")
    print(f"  TRAIN AUC = {auc_train:.4f}")
    print(f"  TEST  AUC = {auc_test:.4f}  (gate: >= 0.55)")

    # Save OOF probs for sim
    OUT_OOF.parent.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({"time": Xte.index, "prob": pte}).reset_index(drop=True)
    oof_df.to_parquet(OUT_OOF, compression="snappy", index=False)
    print(f"Saved OOF probs: {OUT_OOF} ({len(oof_df):,} rows)")

    # Feature list (ONNX input order = feature_cols order)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FEATLIST.write_text("\n".join(feature_cols), encoding="utf-8")
    print(f"Saved feature list: {OUT_FEATLIST}")

    # ONNX export — zipmap=False is CRITICAL for MQL5 (per project_ebb_DEPLOY_CANDIDATE_h10)
    print("Exporting ONNX (zipmap=False)...")
    try:
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        initial_types = [("input", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=12,
            zipmap=False,   # CRITICAL: produces plain float tensor [batch, 2 classes]
        )
        with open(OUT_ONNX, "wb") as fh:
            fh.write(onnx_model.SerializeToString())
        sz = os.path.getsize(OUT_ONNX)
        print(f"Saved ONNX: {OUT_ONNX} ({sz} bytes)")
    except Exception as e:
        print(f"ONNX export FAILED: {e}", file=sys.stderr)
        return 2

    OUT_RUN.parent.mkdir(parents=True, exist_ok=True)
    OUT_RUN.write_text(json.dumps({
        "config": {"h": H_BARS, "sl": SL_MULT, "tp": TP_MULT,
                   "train_start": TRAIN_START, "train_end": TRAIN_END, "test_end": TEST_END,
                   "num_rounds": NUM_ROUNDS, "params": PARAMS},
        "data": {"train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
                 "label_balance_train": float(ytr.mean()),
                 "label_balance_test": float(yte.mean())},
        "metrics": {"auc_train": auc_train, "auc_test": auc_test},
        "feature_count": len(feature_cols),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"Saved run summary: {OUT_RUN}")
    print(f"\nElapsed: {time.time() - t0:.1f}s")
    return 0


def smoke_test() -> int:
    """Train LightGBM on 1000 random rows; verify model trains, predicts, and ONNX export with zipmap=False."""
    print("[smoke] train_eurusd_lightgbm_6yr")
    rng = np.random.default_rng(123)
    n, k = 1000, 8
    X = pd.DataFrame(rng.normal(0, 1, size=(n, k)),
                     columns=[f"f{i}" for i in range(k)])
    y = ((X["f0"] + 0.5 * X["f1"] + rng.normal(0, 0.5, n)) > 0).astype(int)
    d_tr = lgb.Dataset(X, label=y)
    model = lgb.train({"objective": "binary", "metric": "auc",
                       "num_leaves": 15, "learning_rate": 0.1,
                       "verbose": -1, "num_threads": 2},
                      d_tr, num_boost_round=20,
                      callbacks=[lgb.log_evaluation(0)])
    preds = model.predict(X)
    pred_ok = preds.shape == (n,) and (preds >= 0).all() and (preds <= 1).all()
    auc = roc_auc_score(y, preds)
    auc_ok = auc > 0.6
    # ONNX export with zipmap=False (CRITICAL for MQL5 EA)
    try:
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke.onnx"
            onnx_model = convert_lightgbm(
                model,
                initial_types=[("input", FloatTensorType([None, k]))],
                target_opset=12, zipmap=False)
            out.write_bytes(onnx_model.SerializeToString())
            onnx_ok = out.stat().st_size > 100
            # Verify zipmap=False produces tensor output (not sequence-of-maps)
            zipmap_ok = b"ZipMap" not in out.read_bytes()
    except Exception as e:
        print(f"[smoke] FAIL: ONNX export error: {e}")
        return 1
    if pred_ok and auc_ok and onnx_ok and zipmap_ok:
        print(f"[smoke] PASS: train_auc={auc:.3f} preds OK, ONNX zipmap-free")
        return 0
    print(f"[smoke] FAIL: pred={pred_ok} auc={auc_ok}({auc:.3f}) onnx={onnx_ok} zipmap_free={zipmap_ok}")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true", help="run smoke test and exit")
    args = ap.parse_args()
    if args.smoke_only:
        sys.exit(smoke_test())
    sys.exit(main())
