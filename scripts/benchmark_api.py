"""Load-test the running API and report latency percentiles + throughput.

Start the server first, e.g.:  uvicorn aeroops.serving.api:app --port 8000 --workers 1
Then:  python scripts/benchmark_api.py --url http://127.0.0.1:8000 --requests 2000 --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

CARRIERS, AIRPORTS = ["WN", "AA", "DL", "UA", "B6", "AS"], ["ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "LGA", "SEA"]
SENSORS = ["s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
BASE = {"s_2": 642.5, "s_3": 1589.7, "s_4": 1408.9, "s_7": 553.4, "s_8": 2388.1, "s_9": 9065.2, "s_11": 47.5,
        "s_12": 521.4, "s_13": 2388.1, "s_14": 8143.8, "s_15": 8.44, "s_17": 393.0, "s_20": 38.8, "s_21": 23.3}


def delay_payload():
    o, d = random.sample(AIRPORTS, 2)
    return {"carrier": random.choice(CARRIERS), "origin": o, "dest": d, "flight_date": "2024-04-02",
            "dep_hour": random.randint(5, 22), "distance": random.randint(200, 2500),
            "prev_leg_arr_delay": random.choice([None, -5, 0, 20, 75]), "leg_of_day": random.randint(1, 5)}


def rul_payload(n_cycles=30):
    return {"unit_id": 1, "cycles": [{"cycle": i + 1, "sensors": {s: BASE[s] * (1 + random.gauss(0, 1e-3)) for s in SENSORS}}
                                     for i in range(n_cycles)]}


async def run(url, endpoint, n, conc):
    lat = []
    sem = asyncio.Semaphore(conc)
    make = delay_payload if endpoint == "delay" else rul_payload
    async with httpx.AsyncClient(base_url=url, timeout=30) as client:
        for _ in range(20):  # warm-up
            await client.post(f"/predict/{endpoint}", json=make())

        async def one():
            async with sem:
                body = make()
                t = time.perf_counter()
                r = await client.post(f"/predict/{endpoint}", json=body)
                lat.append((time.perf_counter() - t) * 1000)
                r.raise_for_status()

        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(n)])
        wall = time.perf_counter() - t0
    q = statistics.quantiles(lat, n=100)
    return {"endpoint": f"/predict/{endpoint}", "requests": n, "concurrency": conc,
            "p50_ms": round(q[49], 1), "p95_ms": round(q[94], 1), "p99_ms": round(q[98], 1),
            "throughput_rps": round(n / wall, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--out", default="reports/api_benchmark.json")
    a = ap.parse_args()
    random.seed(0)
    res = [asyncio.run(run(a.url, ep, a.requests, c)) for ep in ("delay", "rul") for c in a.concurrency]
    for r in res:
        print(r)
    Path(a.out).write_text(json.dumps(res, indent=2))
