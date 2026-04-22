#!/usr/bin/env python3
"""Parse all Wave 3 MT5 HTM reports and rank by PF."""
import json, re, sys
from pathlib import Path

MANIFEST = Path("/Users/kalinhadzhiev/eurobigbrain/wave3/wave3_manifest_20260422_003456.json")

def parse_htm(path):
    p = Path(path)
    if not p.exists():
        return None
    raw = p.read_bytes()
    for enc in ("utf-16", "utf-8", "cp1252"):
        try:
            txt = raw.decode(enc, errors="ignore")
            if "Profit Factor" in txt or "Total Net Profit" in txt:
                break
        except Exception:
            continue
    plain = re.sub(r"<[^>]+>", "|", txt)
    plain = re.sub(r"\s+", " ", plain)
    def grab(label):
        m = re.search(re.escape(label) + r":?\s*\|+\s*([\-0-9.,]+)", plain)
        return m.group(1).replace(",", "") if m else "?"
    return {
        "net":   grab("Total Net Profit"),
        "pf":    grab("Profit Factor"),
        "trades":grab("Total Trades"),
        "dd":    grab("Equity Drawdown Maximal"),
        "wr":    grab("Profit Trades (% of total)"),
    }

manifest = json.loads(MANIFEST.read_text())
rows = []
for j in manifest:
    r = parse_htm(j["report_abs_path"])
    if r is None:
        rows.append((j["label"], "MISS", "MISS", "MISS", "MISS", "MISS"))
    else:
        rows.append((j["label"], r["pf"], r["trades"], r["net"], r["dd"], r["wr"]))

def to_f(s):
    try: return float(s)
    except: return -999

rows.sort(key=lambda r: -to_f(r[1]))
hdr = "{:<25} {:>7} {:>7} {:>10} {:>10} {:>10}".format("label", "PF", "trades", "net", "DD", "WR")
print(hdr)
print("-" * len(hdr))
for r in rows:
    print("{:<25} {:>7} {:>7} {:>10} {:>10} {:>10}".format(*r))

# Top filtered: PF>=1.3 AND trades>=100
print("\n--- SURVIVORS (PF>=1.3 AND trades>=100) ---")
survivors = [r for r in rows if to_f(r[1]) >= 1.3 and to_f(r[2]) >= 100]
if not survivors:
    print("NONE — same EUR-fails verdict as Wave 1+2.")
else:
    for r in survivors:
        print("{:<25} PF={} trades={} net=${} DD={}% WR={}%".format(*r))
