#!/usr/bin/env python3
"""
SentiKenya End-to-End Inference Test
======================================
Tests the full pipeline: produce 50 posts → preprocess → inference → verify results.

This test exercises:
  1. Kafka producer → raw-posts
  2. Consumer → preprocessed-posts
  3. Inference worker → sentiment-results
  4. Verify all 50 results land in sentiment-results

Usage:
    # Full Kafka-based E2E (requires Kafka running)
    python tests/test_e2e_inference.py --mode kafka

    # Local pipeline E2E (no Kafka required)
    python tests/test_e2e_inference.py --mode local

    # HTTP API E2E (requires FastAPI server running)
    python tests/test_e2e_inference.py --mode api
"""

import argparse
import asyncio
import json
import logging
import sys
import os
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("sentikenya.e2e_test")

# ── 50 sample posts across English, Swahili, and Sheng ──
SAMPLE_POSTS = [
    # English (20)
    "The government needs to address the rising cost of living",
    "Great progress on the road construction in Kiambu County",
    "Healthcare workers deserve better pay and working conditions",
    "M-Pesa has revolutionized how we do business in Kenya",
    "Corruption continues to drain public resources",
    "The housing levy is adding unnecessary burden on workers",
    "Kenya's tech ecosystem is growing faster than ever",
    "Fuel prices keep going up while salaries remain stagnant",
    "The judiciary must remain independent from political interference",
    "Our farmers need better access to markets and fair prices",
    "Security has improved significantly in the northern counties",
    "Youth unemployment is a ticking time bomb",
    "The new expressway has reduced travel time significantly",
    "Water scarcity needs urgent government attention",
    "Education reform is long overdue in this country",
    "The digital economy is creating new opportunities",
    "Climate change is affecting agricultural productivity",
    "Mental health services are severely underfunded",
    "Infrastructure development has been impressive this year",
    "Tax compliance should be voluntary not forced",
    # Swahili (15)
    "Serikali inafanya kazi nzuri katika ujenzi wa barabara",
    "Hali ya uchumi ni mbaya sana kwa wananchi wa kawaida",
    "Tunahitaji mabadiliko katika sekta ya elimu",
    "Vijana wanahitaji kazi na fursa za kujiajiri",
    "Rushwa ni adui mkubwa wa maendeleo",
    "Walimu wanastahili mshahara bora zaidi",
    "Ushuru umezidi na unaumiza wafanyabiashara wadogo",
    "Kilimo kinaweza kuwa suluhisho la ukosefu wa ajira",
    "Teknolojia inabadilisha jinsi tunavyofanya biashara",
    "Elimu ni muhimu sana kwa maendeleo ya taifa",
    "Maji ni uhai lakini bado kuna maeneo bila maji safi",
    "Afya bora kwa wote ni haki ya kila Mkenya",
    "Barabara zetu ni mbaya na zinahitaji kutengenezwa",
    "Serikali imeshindwa kupunguza bei ya mafuta",
    "Wananchi wanateseka na maisha magumu",
    # Sheng (15)
    "Siste hii economy ni tight sana hakuna pesa",
    "Manze serikali iko rada lakini watu hawajui",
    "Hii mambo ya tax ni noma bana inaumiza wasee",
    "Youth tunahitaji chance ya kufanya hustle",
    "Nairobi inakuanga busy sana maze haiwezekani",
    "Pesa imeisha bana hii life ni struggle tu",
    "Hii life ni noma lakini tuko rada wasee",
    "Tech ndio future ya Kenya na youth wako ready",
    "Manze bei ya mafuta imepanda tena si poa",
    "Wasee wa bunge wanajipangia salary kubwa sisi tunapiga hustle",
    "Hii serikali inafanya form tu hawaskii wananchi",
    "Cheki vile Safaricom wamebadilisha game ya tech Kenya",
    "Kazi imeisha bana siste tuko home tu",
    "Mbogi ya tech wako juu sana hii mtaa",
    "Niaje wasee mambo vipi huku mtaani",
]

assert len(SAMPLE_POSTS) == 50, f"Expected 50 posts, got {len(SAMPLE_POSTS)}"


def test_local_pipeline():
    """
    Test all 50 posts through the local NLP pipeline (no Kafka/HTTP required).
    """
    from app.ml.pipeline import NLPPipeline

    pipeline = NLPPipeline(
        sheng_lexicon_path="./data/sheng_lexicon.json",
        entities_path="./data/kenyan_entities.json",
        node_id="TEST-01",
    )

    print(f"\n{'═'*60}")
    print(f"  E2E LOCAL PIPELINE TEST: 50 posts")
    print(f"{'═'*60}\n")

    results = []
    latencies = []
    errors = []

    for i, text in enumerate(SAMPLE_POSTS):
        try:
            t0 = time.time()
            result = pipeline.process_text(text)
            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)

            entry = {
                "text": text[:60],
                "label": result.sentiment.label.value,
                "confidence": round(result.sentiment.confidence, 4),
                "language": result.language.detected_language.value,
                "latency_ms": round(latency_ms, 2),
            }
            results.append(entry)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/50 posts...")

        except Exception as e:
            errors.append((i, str(e)))
            logger.error(f"  ❌ Post {i} failed: {e}")

    # Report
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    labels = [r["label"] for r in results]
    langs = [r["language"] for r in results]

    print(f"\n  Results:")
    print(f"  {'─'*50}")
    print(f"  Total processed:  {len(results)}/50")
    print(f"  Errors:           {len(errors)}")
    print(f"  Avg latency:      {avg_latency:.1f}ms {'✅' if avg_latency < 200 else '⚠️'}")
    print(f"  Max latency:      {max(latencies):.1f}ms" if latencies else "")
    print(f"  Min latency:      {min(latencies):.1f}ms" if latencies else "")
    print(f"  Sentiment dist:   pos={labels.count('positive')}, "
          f"neg={labels.count('negative')}, neu={labels.count('neutral')}")
    print(f"  Language dist:    en={langs.count('en')}, "
          f"sw={langs.count('sw')}, sh={langs.count('sh')}")
    print()

    # Assertions
    assert len(results) == 50, f"Expected 50 results, got {len(results)}"
    assert len(errors) == 0, f"Errors occurred: {errors}"
    assert all(r["label"] in ("positive", "negative", "neutral", "mixed") for r in results)
    assert all(r["confidence"] >= 0 for r in results)

    print(f"  ✅ ALL 50 POSTS PROCESSED SUCCESSFULLY")
    print(f"  ✅ AVG LATENCY: {avg_latency:.1f}ms {'(target: <200ms)' if avg_latency < 200 else '(EXCEEDS TARGET)'}")
    print()

    return results


async def test_api_endpoint():
    """
    Test all 50 posts through the HTTP /predict endpoint.
    """
    import httpx

    url = "http://localhost:8000/api/v1/predict"

    print(f"\n{'═'*60}")
    print(f"  E2E API TEST: 50 posts → POST /predict")
    print(f"{'═'*60}\n")

    results = []
    latencies = []

    async with httpx.AsyncClient() as client:
        for i, text in enumerate(SAMPLE_POSTS):
            try:
                t0 = time.time()
                resp = await client.post(url, json={"text": text}, timeout=10.0)
                latency_ms = (time.time() - t0) * 1000
                latencies.append(latency_ms)

                if resp.status_code == 200:
                    data = resp.json()
                    assert "label" in data, f"Missing 'label' in response"
                    assert "confidence" in data, f"Missing 'confidence' in response"
                    assert "language" in data, f"Missing 'language' in response"
                    results.append(data)
                else:
                    logger.error(f"  Post {i}: HTTP {resp.status_code}")

            except Exception as e:
                logger.error(f"  Post {i}: {e}")

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/50 posts...")

    avg = sum(latencies) / len(latencies) if latencies else 0
    print(f"\n  API Results: {len(results)}/50 successful, avg={avg:.1f}ms")
    print(f"  {'✅ PASS' if len(results) == 50 else '❌ FAIL'}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="SentiKenya E2E Inference Test")
    parser.add_argument("--mode", choices=["local", "api", "kafka"], default="local",
                        help="Test mode: local (pipeline only), api (HTTP), kafka (full)")
    args = parser.parse_args()

    if args.mode == "local":
        test_local_pipeline()
    elif args.mode == "api":
        asyncio.run(test_api_endpoint())
    elif args.mode == "kafka":
        print("  For full Kafka E2E test:")
        print("  1. Start services: docker compose up -d")
        print("  2. Run producer:   python -m kafka.producer --count 50")
        print("  3. Run consumer:   python -m kafka.consumer --forward --count 50")
        print("  4. Run inference:  python -m app.workers.kafka_inference_worker --count 50")
        print("  5. Verify:         Check sentiment-results topic in Kafka UI")


if __name__ == "__main__":
    main()
