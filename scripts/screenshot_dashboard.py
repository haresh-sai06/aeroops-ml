"""Render dashboard screenshots headlessly (used for the README).

streamlit run dashboard/app.py --server.headless true --server.port 8501 &
python scripts/screenshot_dashboard.py --url http://localhost:8501 --out docs/img
Set CHROME_PATH to use a system Chrome instead of `playwright install chromium`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

TABS = {"Network KPIs": "dashboard_kpis.png", "Airports & causes": "dashboard_airports.png", "Models": "dashboard_models.png"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default="docs/img")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=os.getenv("CHROME_PATH") or None, headless=True)
        page = b.new_page(viewport={"width": 1400, "height": 1700})
        page.goto(a.url, wait_until="networkidle")
        page.wait_for_selector("text=AeroOps", timeout=60000)
        for tab, fname in TABS.items():
            page.get_by_role("tab", name=tab).click()
            page.wait_for_timeout(6000)
            page.screenshot(path=str(Path(a.out) / fname), full_page=True)
            print("saved", fname)
        b.close()
