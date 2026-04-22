"""Harvest & parse EuroBigBrain MT5 wave reports from Mac prefixes.

Fetches ebb_w{1,2}_*.htm from ~/mt5_prefix_*/drive_c/Program Files/MetaTrader 5/
over SSH, parses MT5 tester HTMLs (UTF-16LE), writes a unified CSV.

Usage: python parse_ebb_results.py [--ssh-only-fetch|--parse-only] [--filter-pf-min N]
"""
from __future__ import annotations
import argparse, csv, os, re, subprocess, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SSH_KEY = os.path.expanduser("~/.ssh/mac_m5")
SSH_USER_HOST = "kalinhadzhievm5@192.168.100.68"
REMOTE_GLOB = "~/mt5_prefix_*/drive_c/Program\\ Files/MetaTrader\\ 5/ebb_w*.htm"
LOCAL_DIR = Path(r"C:\Users\kahad\IdeaProjects\EuroBigBrain\runs\harvested")
CSV_OUT = LOCAL_DIR / "wave_results.csv"


def _ssh(cmd: str, timeout: int = 60) -> str:
    full = ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10", SSH_USER_HOST, cmd]
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        sys.stderr.write(f"[ssh err] {r.stderr}\n")
    return r.stdout


def list_remote_htms() -> List[str]:
    out = _ssh(f"ls {REMOTE_GLOB} 2>/dev/null")
    return [line.strip() for line in out.splitlines() if line.strip()]


def fetch_all() -> List[Path]:
    """Pull each htm via `cat` over SSH (binary stdout -> local file).

    scp choked on the embedded space in "Program Files" regardless of how we
    quoted it; cat-over-ssh sidesteps the double-shell-escape problem.
    """
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    remote_paths = list_remote_htms()
    print(f"[fetch] {len(remote_paths)} remote HTMs found")
    fetched: List[Path] = []
    for rp in remote_paths:
        m = re.search(r"/mt5_prefix_(\d+)/", rp)
        tag = f"p{m.group(1)}_" if m else ""
        local = LOCAL_DIR / (tag + os.path.basename(rp))
        cmd = ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=10", SSH_USER_HOST, f"cat '{rp}'"]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode == 0 and r.stdout:
            local.write_bytes(r.stdout)
            fetched.append(local)
        else:
            err = r.stderr.decode("utf-8", "ignore").strip()
            sys.stderr.write(f"[fetch fail] {rp}: {err}\n")
    print(f"[fetch] downloaded {len(fetched)} HTMs to {LOCAL_DIR}")
    return fetched


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-16-le", "utf-16", "utf-8"):
        try:
            t = raw.decode(enc)
            if t and t[0] == "﻿":
                t = t[1:]
            return t
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def _val_after(text: str, label: str) -> Optional[str]:
    """Return bolded value in first <td><b>...</b></td> after `label`."""
    i = text.find(label)
    if i < 0:
        return None
    m = re.search(r"<b>([^<]+)</b>", text[i: i + 600])
    return m.group(1).strip() if m else None


def _to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    # MT5 uses Unicode NBSP (U+00A0) as thousands separator
    s = s.replace(" ", "").replace(" ", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _pct_in_parens(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"\(([-+]?\d+(?:\.\d+)?)\s*%\)", s)
    return float(m.group(1)) if m else None


def _count_in_value(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(\d+)", s.replace(" ", "").replace(" ", ""))
    return int(m.group(1)) if m else None


LABEL_RE = re.compile(r"ebb_(w\d)_(.+?)_\d{8}_\d{6}\.htm$", re.I)


def parse_label(fname: str) -> Dict[str, Any]:
    """Extract wave + session + VT hints from the filename stem."""
    m = LABEL_RE.search(fname)
    wave = m.group(1).lower() if m else ""
    label = m.group(2) if m else Path(fname).stem
    out: Dict[str, Any] = {"wave": wave, "label": label}
    sm = re.search(r"S(\d{2})(\d{2})", label)
    if sm:
        out["session_start"] = int(sm.group(1))
        out["session_end"] = int(sm.group(2))
    vm = re.search(r"VT(\d{3})", label)
    if vm:
        out["vt"] = int(vm.group(1)) / 100.0
    return out


def parse_htm(path: Path) -> Dict[str, Any]:
    text = _read_text(path).replace("\r", "").replace("\n", " ")
    rec: Dict[str, Any] = {"source_path": str(path)}
    rec.update(parse_label(path.name))
    rec["symbol"] = _val_after(text, "Symbol:")
    rec["period"] = _val_after(text, "Period:")

    # Collect all <b>key=value</b> pairs in the Inputs block.
    inputs: Dict[str, str] = {}
    i0 = text.find("Inputs:")
    if i0 >= 0:
        scan = text[i0: i0 + 60000]
        stop = scan.find("Company:")
        if stop > 0:
            scan = scan[:stop]
        for key, val in re.findall(r"<b>([A-Za-z_][A-Za-z0-9_]*)=([^<]+)</b>", scan):
            inputs[key] = val.strip()
    if rec.get("vt") is None:
        rec["vt"] = _to_float(inputs.get("VolThreshold"))
    rec["sl_atr"] = _to_float(inputs.get("SL_ATR_Mult"))
    rec["tp_atr"] = _to_float(inputs.get("TP_ATR_Mult"))
    if rec.get("session_start") is None:
        rec["session_start"] = _to_float(inputs.get("SessionStart"))
    if rec.get("session_end") is None:
        rec["session_end"] = _to_float(inputs.get("SessionEnd"))

    rec["total_net_profit"] = _to_float(_val_after(text, "Total Net Profit:"))
    rec["profit_factor"] = _to_float(_val_after(text, "Profit Factor:"))
    rec["total_trades"] = _count_in_value(_val_after(text, "Total Trades:"))
    rec["max_dd_pct"] = _pct_in_parens(_val_after(text, "Balance Drawdown Maximal:"))
    rec["sharpe"] = _to_float(_val_after(text, "Sharpe Ratio:"))
    rec["recovery_factor"] = _to_float(_val_after(text, "Recovery Factor:"))
    rec["win_rate"] = _pct_in_parens(_val_after(text, "Profit Trades (% of total):"))
    rec["largest_profit"] = _to_float(_val_after(text, "Largest profit trade:"))
    rec["largest_loss"] = _to_float(_val_after(text, "Largest loss trade:"))
    rec["avg_hold_time"] = _val_after(text, "Average position holding time:")
    return rec


CSV_COLS = ["wave", "label", "symbol", "session_start", "session_end",
            "vt", "sl_atr", "tp_atr", "total_net_profit", "profit_factor",
            "total_trades", "max_dd_pct", "sharpe", "recovery_factor",
            "win_rate", "source_path"]


def write_csv(records: List[Dict[str, Any]]) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    print(f"[csv] wrote {len(records)} rows -> {CSV_OUT}")


def parse_dir() -> List[Dict[str, Any]]:
    htms = sorted(set(list(LOCAL_DIR.glob("*ebb_w*.htm"))))
    parsed, failed = [], []
    for p in htms:
        try:
            rec = parse_htm(p)
            (parsed if rec.get("profit_factor") is not None
                     or rec.get("total_trades") is not None else failed).append(rec)
        except Exception as e:
            sys.stderr.write(f"[parse fail] {p.name}: {e}\n")
            failed.append(p)
    print(f"[parse] {len(parsed)} ok / {len(failed)} failed of {len(htms)} HTMs")
    return parsed


def top5(records: List[Dict[str, Any]], min_trades: int = 100) -> None:
    elig = [r for r in records
            if (r.get("total_trades") or 0) > min_trades
            and r.get("profit_factor") is not None]
    elig.sort(key=lambda r: r["profit_factor"], reverse=True)
    top = elig[:5]
    print(f"\nTop 5 by Profit Factor (trades > {min_trades}):")
    hdr = f"{'wave':<4} {'label':<28} {'PF':>6} {'trades':>7} {'net':>12} {'DD%':>7} {'WR%':>6}"
    print(hdr); print("-" * len(hdr))
    for r in top:
        print(f"{r.get('wave',''):<4} {str(r.get('label',''))[:28]:<28} "
              f"{r.get('profit_factor') or 0:>6.2f} "
              f"{r.get('total_trades') or 0:>7d} "
              f"{r.get('total_net_profit') or 0:>12.2f} "
              f"{r.get('max_dd_pct') or 0:>6.2f}% "
              f"{r.get('win_rate') or 0:>5.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh-only-fetch", action="store_true")
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--filter-pf-min", type=float, default=0.0)
    args = ap.parse_args()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if not args.parse_only:
        fetch_all()
    if args.ssh_only_fetch:
        return 0
    records = parse_dir()
    if args.filter_pf_min > 0:
        records = [r for r in records
                   if (r.get("profit_factor") or 0) >= args.filter_pf_min]
        print(f"[filter] kept {len(records)} rows with PF >= {args.filter_pf_min}")
    write_csv(records)
    top5(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
