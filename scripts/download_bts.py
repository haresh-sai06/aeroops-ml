"""Download BTS "Reporting Carrier On-Time Performance" monthly files (public, US DOT).

Example: python scripts/download_data.py --year 2024 --months 1 2 3
Each month is ~25-30 MB zipped / ~250 MB CSV. Files land in data/raw/ (git-ignored).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests

URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"


def download(year: int, month: int, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"ontime_{year}_{month:02d}.csv"
    if target.exists():
        print(f"exists: {target}")
        return target
    zpath = raw_dir / f"ontime_{year}_{month:02d}.zip"
    with requests.get(URL.format(year=year, month=month), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    zpath.unlink()
    print(f"downloaded: {target}")
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--months", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--raw-dir", default="data/raw")
    a = ap.parse_args()
    for m in a.months:
        download(a.year, m, Path(a.raw_dir))
