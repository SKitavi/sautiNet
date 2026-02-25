#!/usr/bin/env python3
"""
SentiKenya Inference Latency Benchmark
========================================
Sends N requests to POST /predict and reports latency statistics.

Metrics: average, median (p50), p95, p99, min, max.
Target: average < 200ms per post.

Usage:
    python scripts/benchmark_latency.py                     # 100 requests to localhost
    python scripts/benchmark_latency.py --url http://localhost:8000 --n 200
    python scripts/benchmark_latency.py --concurrent 5      # parallel requests
"""

import argparse
import asyncio
import json
import logging
import statistics
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("benchmark")

SAMPLE_TEXTS = [
    "The government needs to address the rising cost of living immediately",
    "Serikali inafanya kazi nzuri katika ujenzi wa barabara",
    "Manze hii economy ni tight sana bana wasee",
    "Corruption continues to drain public resources",
    "Vijana wanahitaji kazi na fursa za kujiajiri",
    "Hii mambo ya tax ni noma bana inaumiza wasee",
    "Kenya's tech ecosystem is growing faster than ever before",
    "Hali ya uchumi ni mbaya sana kwa wananchi wa kawaida",
    "Youth tunahitaji chance ya kufanya hustle",
    "M-Pesa has revolutionized how we do business in Kenya",
    "Rushwa ni adui mkubwa wa maendeleo katika nchi yetu",
    "Cheki vile Safaricom wamebadilisha game ya tech Kenya",
    "Healthcare workers deserve better pay and working conditions",
    "Ushuru umezidi na unaumiza wafanyabiashara wadogo",
    "Siste pesa imeisha bana hii life ni struggle tu",
]


async def benchmark(url: str, n: int, concurrent: int):
    """Run N inference requests and collect latency stats."""
    endpoint = f"{url.rstrip('/')}/api/v1/predict"
    latencies = []
    errors = 0

    print(f"\n{'═'*60}")
    print(f"  SentiKenya Latency Benchmark")
    print(f"  Endpoint: {endpoint}")
    print(f"  Requests: {n}  |  Concurrency: {concurrent}")
    print(f"{'═'*60}\n")

    semaphore = asyncio.Semaphore(concurrent)

    async def single_request(client: httpx.AsyncClient, idx: int):
        nonlocal errors
        text = SAMPLE_TEXTS[idx % len(SAMPLE_TEXTS)]
        async with semaphore:
            try:
                t0 = time.perf_counter()
                resp = await client.post(endpoint, json={"text": text}, timeout=10.0)
                latency_ms = (time.perf_counter() - t0) * 1000

                if resp.status_code == 200:
                    latencies.append(latency_ms)
                else:
                    errors += 1
                    logger.warning(f"Request {idx}: HTTP {resp.status_code}")
            except Exception as e:
                errors += 1
                logger.warning(f"Request {idx}: {e}")

    async with httpx.AsyncClient() as client:
        # Warm-up
        logger.info("Warming up (3 requests)...")
        for i in range(3):
            await single_request(client, i)
        latencies.clear()

        # Benchmark
        logger.info(f"Running {n} benchmark requests...")
        t_start = time.perf_counter()

        tasks = [single_request(client, i) for i in range(n)]
        await asyncio.gather(*tasks)

        total_time = time.perf_counter() - t_start

    if not latencies:
        print("  ❌ No successful requests. Is the server running?")
        return

    # Compute statistics
    latencies.sort()
    avg = statistics.mean(latencies)
    median = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    throughput = len(latencies) / total_time

    print(f"  Results ({len(latencies)} successful, {errors} errors):")
    print(f"  {'─'*44}")
    print(f"  Average:    {avg:>8.1f} ms  {'✅' if avg < 200 else '⚠️'} (target: <200ms)")
    print(f"  Median:     {median:>8.1f} ms")
    print(f"  P95:        {p95:>8.1f} ms")
    print(f"  P99:        {p99:>8.1f} ms")
    print(f"  Min:        {min(latencies):>8.1f} ms")
    print(f"  Max:        {max(latencies):>8.1f} ms")
    print(f"  Throughput: {throughput:>8.1f} req/s")
    print(f"  Total time: {total_time:>8.1f} s")
    print()

    target_met = avg < 200
    print(f"  Target (<200ms avg): {'✅ PASS' if target_met else '❌ FAIL'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="SentiKenya Latency Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--n", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrent requests")
    args = parser.parse_args()

    asyncio.run(benchmark(args.url, args.n, args.concurrent))


if __name__ == "__main__":
    main()
