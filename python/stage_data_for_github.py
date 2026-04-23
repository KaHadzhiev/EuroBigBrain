#!/usr/bin/env python3
"""Stage all EBB + GBB data into MT-Bots-trading-data tree, gzip in parallel.

Skips the huge merged tick parquets (eurusd_ticks_2020_2026.parquet,
gbpusd_ticks_2020_2026.parquet) — they are rebuildable from the raw monthly
Dukascopy files via python/merge_dukascopy_*.py.
"""
from __future__ import annotations
import gzip
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

EBB_DATA = Path("C:/Users/kahad/IdeaProjects/EuroBigBrain/data")
GBB_DATA = Path("C:/Users/kahad/IdeaProjects/GoldBigBrain/data")
STAGE    = Path("C:/Users/kahad/IdeaProjects/MT-Bots-trading-data")

# Files we DON'T upload (rebuildable from raw)
SKIP_NAMES = {
    "eurusd_ticks_2020_2026.parquet",
    "gbpusd_ticks_2020_2026.parquet",
}
# File extensions worth gzipping (skip already-compressed-ish things)
GZ_EXTS = {".parquet", ".csv", ".json", ".yaml", ".yml", ".txt"}


def plan() -> list[tuple[Path, Path]]:
    """Yield (src, dst) tuples. dst is the .gz output path."""
    jobs: list[tuple[Path, Path]] = []

    # ---- EBB ----
    # Raw Dukascopy monthlies → eurobigbrain/dukascopy/<symbol>/<file>.gz
    for f in sorted((EBB_DATA / "dukascopy").glob("*_ticks_*.parquet")):
        sym = f.name.split("_ticks_")[0]
        out = STAGE / "eurobigbrain" / "dukascopy" / sym / (f.name + ".gz")
        jobs.append((f, out))
    # Built top-level parquets → eurobigbrain/built/<file>.gz
    for f in sorted(EBB_DATA.glob("*.parquet")):
        if f.name in SKIP_NAMES:
            continue
        out = STAGE / "eurobigbrain" / "built" / (f.name + ".gz")
        jobs.append((f, out))
    # Misc CSVs at top level → eurobigbrain/built/
    for f in sorted(EBB_DATA.glob("*.csv")):
        out = STAGE / "eurobigbrain" / "built" / (f.name + ".gz")
        jobs.append((f, out))

    # ---- GBB ----
    # Raw Dukascopy → goldbigbrain/dukascopy/xauusd/
    for f in sorted((GBB_DATA / "dukascopy").glob("*_ticks_*.parquet")):
        sym = f.name.split("_ticks_")[0]
        out = STAGE / "goldbigbrain" / "dukascopy" / sym / (f.name + ".gz")
        jobs.append((f, out))
    # Top-level parquets
    for f in sorted(GBB_DATA.glob("*.parquet")):
        out = STAGE / "goldbigbrain" / "built" / (f.name + ".gz")
        jobs.append((f, out))
    # Top-level CSVs (M1/M5/M15/H1/ticks/specs)
    for f in sorted(GBB_DATA.glob("*.csv")):
        out = STAGE / "goldbigbrain" / "built" / (f.name + ".gz")
        jobs.append((f, out))
    # YAML / JSON / env
    for ext in ("*.yaml", "*.json", "*.env"):
        for f in sorted(GBB_DATA.glob(ext)):
            out = STAGE / "goldbigbrain" / "misc" / (f.name + ".gz")
            jobs.append((f, out))

    return jobs


def gzip_one(args: tuple[Path, Path]) -> tuple[str, int, int, float]:
    src, dst = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return (str(src), src.stat().st_size, dst.stat().st_size, 0.0)
    t0 = time.time()
    with src.open("rb") as fin, gzip.open(dst, "wb", compresslevel=9) as fout:
        shutil.copyfileobj(fin, fout, length=4 * 1024 * 1024)
    return (str(src), src.stat().st_size, dst.stat().st_size, time.time() - t0)


def main() -> int:
    jobs = plan()
    total_in = sum(s.stat().st_size for s, _ in jobs)
    print(f"[stage] {len(jobs)} files, {total_in/1e9:.2f} GB raw -> staging")
    t0 = time.time()
    done_in = 0
    done_out = 0
    with mp.Pool(processes=6) as pool:
        for i, (path, sz_in, sz_out, dt) in enumerate(pool.imap_unordered(gzip_one, jobs), 1):
            done_in += sz_in
            done_out += sz_out
            if i % 25 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                pct = done_in / total_in * 100
                eta = elapsed * (total_in - done_in) / max(done_in, 1)
                print(f"  [{i:3d}/{len(jobs)}] {pct:5.1f}%  in={done_in/1e9:.2f}GB out={done_out/1e9:.2f}GB  saved={(done_in-done_out)/done_in*100:.1f}%  eta={eta:.0f}s")
    print(f"\n[done] {len(jobs)} files in {time.time()-t0:.1f}s")
    print(f"  raw  : {done_in/1e9:.2f} GB")
    print(f"  gzip : {done_out/1e9:.2f} GB  ({(done_in-done_out)/done_in*100:.1f}% saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
