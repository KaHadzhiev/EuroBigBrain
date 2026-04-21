#!/usr/bin/env python3
"""EBB Gate 8 — Multi-Instrument Null Test (M113 killer).

Runs candidate strategy UNCHANGED on EURUSD base + 3 tests via MT5
every-tick Model=8, parses HTMs, classifies:
  Tier A (ACCEPT)     : PF>=1.0 on >=3/4 of {EUR, GBP, JPY, XAG}, no wipe
  Tier B (CONDITIONAL): PF>=1.0 on 2/3 tests, no wipe (needs rationale)
  Tier C (REJECT)     : 2+ test fails OR any DD>40% (M113 signature)

Refs: project_m113_xauusd_overfit.md, findings/wg3_validation_pipeline.md

ATR-relative-scaling caveats (baked into warnings in the verdict):
 * SL/TP/Bracket/VolThreshold are ATR-ratios — they auto-scale across
   instruments via ATR itself. That's the whole point of relative params.
 * BUT a VolThreshold derived from a GOLD-trained ONNX vol model is
   MEANINGLESS on EUR features — different feature distribution. EBB's
   vol gate must be per-instrument or pooled. Flag if source=gold_onnx.
 * Session hours are NOT ATR-relative. If zero trades fire on a test
   instrument, that is the M113 signature (EURUSD=0 trades @ VT=0.10
   gold-scaled) — treated as a FAIL, not a neutral result.
 * Per-instrument spread differs; Gate 6 (cost stress) owns that.

Output dir: runs/multi_instrument_null/<cfg_id>_<ts>/
  verdict.json  results.csv  ini/  reports/  logs/  run.log
"""
from __future__ import annotations
import argparse, csv, json, logging, re, shutil, subprocess
import sys, time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MT5_INSTANCE = Path("C:/MT5-Instances/Instance1")
EXPERT_NAME = "GoldBigBrain\\GBB_Core"  # same core EA; symbol switches via ini
OUT_ROOT = REPO_ROOT / "runs" / "multi_instrument_null"

# Default test instruments per WG3 Gate 8 spec. GBPUSD + USDJPY + XAGUSD
# mirror the M113 failure set exactly (EURUSD is the base). AUDUSD/USDCHF
# were WG3's originals; we follow the user's explicit overrides.
DEFAULT_TEST_INSTRUMENTS = ["GBPUSD", "USDJPY", "XAGUSD"]

# 3-tier thresholds (WG3 Gate 8)
PF_PASS_THRESHOLD = 1.0
WIPE_DD_PCT = 40.0   # Tier C auto-trigger if any DD > this

# MT5 tester settle / timeout
LAUNCH_SETTLE_SEC = 3.0
MT5_TIMEOUT_SEC = 3600  # 1h cap per instrument run


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class InstrumentResult:
    symbol: str
    pf: Optional[float] = None
    dd_pct: Optional[float] = None
    net_profit: Optional[float] = None
    trades: Optional[int] = None
    error: Optional[str] = None
    report_path: Optional[str] = None

    @property
    def passed(self) -> bool:
        return (self.pf is not None and self.dd_pct is not None
                and self.pf >= PF_PASS_THRESHOLD and self.dd_pct <= WIPE_DD_PCT
                and self.trades != 0)
    @property
    def wiped(self) -> bool:
        return self.dd_pct is not None and self.dd_pct > WIPE_DD_PCT
    @property
    def zero_trades(self) -> bool:
        return self.trades == 0


@dataclass
class Verdict:
    tier: str                # "A" | "B" | "C" | "ERROR"
    classification: str      # ACCEPT / CONDITIONAL / REJECT / ERROR
    reason: str
    results: List[InstrumentResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    run_id: str = ""
    config_id: str = ""

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# MT5 ini / launcher
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.10f}".rstrip("0")
        return s + "0" if s.endswith(".") else s
    return str(v)


def build_mt5_ini(params: Dict, symbol: str, ini_path: Path, report_name: str,
                  from_date: str, to_date: str, deposit: int = 1000,
                  leverage: int = 500) -> None:
    """Write MT5 tester ini (UTF-16LE BOM required by MT5)."""
    lines = ["[Tester]", f"Expert={EXPERT_NAME}", f"Symbol={symbol}",
             "Period=M5", "Model=8", "Optimization=0",
             f"FromDate={from_date}", f"ToDate={to_date}", "ForwardMode=0",
             f"Deposit={deposit}", "Currency=USD", f"Leverage={leverage}",
             "ExecutionMode=0", "Visual=0", "ShutdownTerminal=1",
             "ReplaceReport=1", f"Report={report_name}", "", "[TesterInputs]"]
    for k, v in sorted(params.items()):
        fv = _fmt(v)
        lines.append(f"{k}={fv}||{fv}||0||{fv}||N")
    ini_path.parent.mkdir(parents=True, exist_ok=True)
    ini_path.write_bytes(b"\xff\xfe" + ("\n".join(lines) + "\n").encode("utf-16-le"))


def launch_mt5_tester(instance_path: Path, ini_path: Path, log_path: Path,
                      timeout_sec: int = MT5_TIMEOUT_SEC) -> int:
    """Launch hidden MT5, wait for self-shutdown. Returns exit code."""
    terminal = instance_path / "terminal64.exe"
    if not terminal.exists():
        raise FileNotFoundError(f"MT5 terminal not found: {terminal}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ps_cmd = (f'$p = Start-Process -FilePath "{terminal}" '
              f'-ArgumentList "/config:{ini_path}" -WindowStyle Hidden '
              f'-RedirectStandardOutput "{log_path}" '
              f'-RedirectStandardError "{log_path}.err" -PassThru; '
              f'Wait-Process -Id $p.Id -Timeout {timeout_sec}; exit $p.ExitCode')
    return subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                          capture_output=True, text=True).returncode


# ---------------------------------------------------------------------------
# HTM parser (MT5 tester report)
# ---------------------------------------------------------------------------
_NUM_SANITIZE = re.compile(r"[\s ]")


def _to_float(raw: str) -> Optional[float]:
    s = _NUM_SANITIZE.sub("", raw).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_mt5_htm(report_path: Path) -> InstrumentResult:
    """Extract PF/trades/DD%/NetProfit from MT5 tester HTM."""
    symbol = report_path.stem
    if not report_path.exists():
        return InstrumentResult(symbol=symbol, error=f"No report: {report_path}")

    text = report_path.read_text(encoding="utf-8", errors="replace")

    def grab(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, re.S)
        return m.group(1) if m else None

    pf_raw = grab(r"Profit Factor.*?<td[^>]*>([\d.,\s ]+)")
    trades_raw = grab(r"Total Trades.*?<td[^>]*>(\d+)")
    pnl_raw = grab(r"Total Net Profit.*?<td[^>]*>([-\d.,\s ]+)")
    # DD as % is typically in "Equity Drawdown Maximal" cell with (xx.xx%) suffix
    dd_raw = grab(r"(?:Equity Drawdown Maximal|Balance Drawdown Maximal)"
                  r".*?<td[^>]*>[^(]*\(([\d.,]+)%\)")

    return InstrumentResult(
        symbol=symbol,
        pf=_to_float(pf_raw) if pf_raw else None,
        trades=int(trades_raw) if trades_raw else None,
        net_profit=_to_float(pnl_raw) if pnl_raw else None,
        dd_pct=_to_float(dd_raw) if dd_raw else None,
        report_path=str(report_path),
    )


# ---------------------------------------------------------------------------
# 3-tier classifier
# ---------------------------------------------------------------------------
def classify(base_result: InstrumentResult,
             test_results: List[InstrumentResult]) -> Verdict:
    """WG3 Gate 8 tiers. Zero-trades on a test = FAIL (M113 signature)."""
    warnings: List[str] = []
    all_results = [base_result] + test_results

    if any(r.error for r in all_results):
        errs = "; ".join(f"{r.symbol}: {r.error}" for r in all_results if r.error)
        return Verdict(tier="ERROR", classification="ERROR",
                       reason=f"Parse/launch errors: {errs}",
                       results=all_results, warnings=warnings)

    passes_all = sum(1 for r in all_results if r.passed)  # includes base
    test_passes = sum(1 for r in test_results if r.passed)
    test_fails = len(test_results) - test_passes
    wiped = [r.symbol for r in test_results if r.wiped]

    for r in test_results:
        if r.zero_trades:
            warnings.append(
                f"{r.symbol}: zero trades — ATR threshold likely gold-scaled "
                "(M113 signature). Retuning VT per instrument = admission of overfit.")
        if r.wiped:
            warnings.append(
                f"{r.symbol}: DD {r.dd_pct:.1f}% > {WIPE_DD_PCT}% — wipe territory.")

    # Tier C (M113 signature): wipe on any test, or 2+ test fails
    if wiped or test_fails >= 2:
        bits = []
        if wiped:
            bits.append(f"wiped on {','.join(wiped)} (DD>{WIPE_DD_PCT}%)")
        if test_fails >= 2:
            fs = [r.symbol for r in test_results if not r.passed]
            bits.append(f"{test_fails}/{len(test_results)} tests failed PF>="
                        f"{PF_PASS_THRESHOLD} ({','.join(fs)})")
        return Verdict(tier="C", classification="REJECT",
                       reason="M113 signature: " + "; ".join(bits),
                       results=all_results, warnings=warnings)
    # Tier A: base + ALL tests pass (the "structural" bar per WG3)
    if base_result.passed and test_passes == len(test_results):
        return Verdict(tier="A", classification="ACCEPT",
                       reason=f"{passes_all}/4 instruments pass PF>="
                              f"{PF_PASS_THRESHOLD}, no wipes. Structural edge.",
                       results=all_results, warnings=warnings)
    # Tier B: at least 2/3 tests pass and nothing wiped
    if test_passes >= 2:
        return Verdict(tier="B", classification="CONDITIONAL",
                       reason=f"{test_passes}/{len(test_results)} tests pass "
                              "— requires documented economic rationale.",
                       results=all_results, warnings=warnings)
    return Verdict(tier="C", classification="REJECT",
                   reason=f"Only {test_passes}/{len(test_results)} tests passed.",
                   results=all_results, warnings=warnings)


# ---------------------------------------------------------------------------
# CSV log writer
# ---------------------------------------------------------------------------
def write_results_csv(verdict: Verdict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["run_id", "config_id", "symbol", "pf", "dd_pct", "net_profit",
            "trades", "passed", "wiped", "zero_trades", "tier",
            "classification", "error"]
    def nn(v): return "" if v is None else v
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in verdict.results:
            w.writerow([verdict.run_id, verdict.config_id, r.symbol,
                        nn(r.pf), nn(r.dd_pct), nn(r.net_profit), nn(r.trades),
                        r.passed, r.wiped, r.zero_trades,
                        verdict.tier, verdict.classification, r.error or ""])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_multi_instrument_validation(
    strategy_config: Dict,
    base_instrument: str = "EURUSD",
    test_instruments: Optional[List[str]] = None,
    from_date: str = "2020.01.03",
    to_date: str = "2026.04.10",
    mt5_instance: Path = DEFAULT_MT5_INSTANCE,
    out_root: Path = OUT_ROOT,
    config_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Verdict:
    """Run the 4-instrument null test sequentially and return a Verdict."""
    if test_instruments is None:
        test_instruments = list(DEFAULT_TEST_INSTRUMENTS)
    logger = logger or logging.getLogger("multi_instrument_null")
    params = dict(strategy_config.get("params", strategy_config))
    cfg_id = config_id or strategy_config.get("id") or "cfg_" + datetime.now().strftime("%H%M%S")
    run_id = f"{cfg_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    for sub in ("ini", "reports", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    all_symbols = [base_instrument] + [s for s in test_instruments if s != base_instrument]
    logger.info(f"[{run_id}] instruments={all_symbols}  period={from_date}->{to_date}")

    results: List[InstrumentResult] = []
    for idx, symbol in enumerate(all_symbols):
        report_name = f"{symbol}_{run_id}.htm"
        ini_path = run_dir / "ini" / f"{symbol}.ini"
        log_path = run_dir / "logs" / f"{symbol}.log"
        sym_params = dict(params)
        sym_params["MagicNumber"] = int(sym_params.get("MagicNumber", 20260422000)) + idx
        build_mt5_ini(sym_params, symbol, ini_path, report_name, from_date, to_date)
        logger.info(f"[{symbol}] launching MT5 every-tick...")
        t0 = time.time()
        try:
            rc = launch_mt5_tester(mt5_instance, ini_path, log_path)
            logger.info(f"[{symbol}] MT5 rc={rc} in {(time.time()-t0)/60:.1f}min")
        except Exception as e:
            results.append(InstrumentResult(symbol=symbol, error=f"launch_failed: {e}"))
            continue
        # MT5 writes HTM relative to instance dir when Report=<name>
        report_src = mt5_instance / report_name
        report_dst = run_dir / "reports" / f"{symbol}.htm"
        if report_src.exists():
            shutil.copy2(report_src, report_dst)
            r = parse_mt5_htm(report_dst)
            r.symbol = symbol
            results.append(r)
            logger.info(f"[{symbol}] PF={r.pf} DD={r.dd_pct}% trades={r.trades}")
        else:
            results.append(InstrumentResult(symbol=symbol, error=f"report missing: {report_src}"))
            logger.warning(f"[{symbol}] no report at {report_src}")
        time.sleep(LAUNCH_SETTLE_SEC)

    base_result = next((r for r in results if r.symbol == base_instrument), None)
    test_results = [r for r in results if r.symbol != base_instrument]
    if base_result is None:
        verdict = Verdict(tier="ERROR", classification="ERROR",
                          reason="Base instrument run missing",
                          results=results, run_id=run_id, config_id=cfg_id)
    else:
        verdict = classify(base_result, test_results)
        verdict.run_id = run_id
        verdict.config_id = cfg_id

    # Gold-ONNX VolThreshold caveat (see module docstring section 1)
    if params.get("VolThreshold") is not None and \
       strategy_config.get("vol_model_source") == "gold_onnx":
        verdict.warnings.append(
            "VolThreshold came from gold-trained ONNX; vol features on non-XAU "
            "have different distribution. Results reflect a gated-by-gold-model "
            "gate, not an EUR-native vol gate. Retrain vol model per-instrument "
            "or pooled before interpreting as structural edge.")

    (run_dir / "verdict.json").write_text(json.dumps(verdict.to_dict(), indent=2, default=str))
    write_results_csv(verdict, run_dir / "results.csv")
    logger.info(f"[{run_id}] VERDICT: Tier {verdict.tier}/{verdict.classification} — {verdict.reason}")
    for w in verdict.warnings:
        logger.warning(f"[{run_id}] {w}")
    return verdict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _setup_logging(logfile: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("multi_instrument_null")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="EBB multi-instrument null test (M113 killer, WG3 Gate 8)")
    ap.add_argument("--config", required=True, help="strategy config JSON")
    ap.add_argument("--base", default="EURUSD")
    ap.add_argument("--tests", default=",".join(DEFAULT_TEST_INSTRUMENTS))
    ap.add_argument("--from", dest="from_date", default="2020.01.03")
    ap.add_argument("--to", dest="to_date", default="2026.04.10")
    ap.add_argument("--mt5-instance", default=str(DEFAULT_MT5_INSTANCE))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip MT5, fabricate a passing verdict (smoke test)")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    test_instruments = [s.strip() for s in args.tests.split(",") if s.strip()]
    out_root = Path(args.out)
    run_id = f"{cfg.get('id', cfg_path.stem)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = _setup_logging(out_root / run_id / "run.log")

    if args.dry_run:
        logger.info("[DRY-RUN] skipping MT5, producing fake PASS verdict")
        fake = [InstrumentResult(s, pf=1.2, dd_pct=10, net_profit=100, trades=500)
                for s in [args.base] + test_instruments]
        v = classify(fake[0], fake[1:])
        v.run_id, v.config_id = run_id, cfg.get("id", cfg_path.stem)
        print(json.dumps(v.to_dict(), indent=2, default=str))
        return 0

    verdict = run_multi_instrument_validation(
        strategy_config=cfg, base_instrument=args.base,
        test_instruments=test_instruments,
        from_date=args.from_date, to_date=args.to_date,
        mt5_instance=Path(args.mt5_instance), out_root=out_root,
        config_id=cfg.get("id", cfg_path.stem), logger=logger)
    # Exit: 0=A/B (accept/conditional), 1=C (reject), 2=error
    return {"A": 0, "B": 0, "C": 1}.get(verdict.tier, 2)


if __name__ == "__main__":
    sys.exit(main())
