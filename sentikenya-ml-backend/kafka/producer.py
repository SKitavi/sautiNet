#!/usr/bin/env python3
"""
SentiKenya Kafka Producer
==========================
Standalone producer script that simulates streaming Kenyan social media posts
into the ``raw-posts`` Kafka topic.

Usage:
    python -m kafka.producer                        # defaults: localhost:9092, 1.5s interval
    python -m kafka.producer --broker kafka:29092 --interval 0.5 --count 100
    python -m kafka.producer --continuous            # run indefinitely
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

from aiokafka import AIOKafkaProducer
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("sentikenya.producer")

# ── Kafka Topics ──
TOPIC_RAW = "raw-posts"

# ── Simulated corpus ──

TEXTS = {
    "en": [
        "The government needs to address the rising cost of living immediately",
        "Great progress on the road construction in Kiambu County",
        "Healthcare workers deserve better pay and working conditions",
        "The new education curriculum is confusing parents and teachers alike",
        "M-Pesa has revolutionized how we do business in Kenya",
        "Corruption continues to drain public resources meant for development",
        "The housing levy is adding unnecessary burden on workers",
        "Kenya's tech ecosystem is growing faster than ever before",
        "Fuel prices keep going up while salaries remain stagnant",
        "The judiciary must remain independent from political interference",
        "Our farmers need better access to markets and fair prices",
        "Security has improved significantly in the northern counties",
        "Youth unemployment is a ticking time bomb we must address",
        "The new expressway has reduced travel time significantly",
        "Water scarcity in arid counties needs urgent government attention",
    ],
    "sw": [
        "Serikali inafanya kazi nzuri katika ujenzi wa barabara",
        "Hali ya uchumi ni mbaya sana kwa wananchi wa kawaida",
        "Tunahitaji mabadiliko katika sekta ya elimu nchini Kenya",
        "Vijana wanahitaji kazi na fursa za kujiajiri",
        "Elimu ni muhimu sana kwa maendeleo ya taifa letu",
        "Maji ni uhai lakini bado kuna maeneo bila maji safi",
        "Barabara zetu ni mbaya na zinahitaji kutengenezwa",
        "Afya bora kwa wote ni haki ya kila Mkenya",
        "Ushuru umezidi na unaumiza wafanyabiashara wadogo",
        "Serikali imeshindwa kupunguza bei ya mafuta",
        "Rushwa ni adui mkubwa wa maendeleo katika nchi yetu",
        "Walimu wetu wanastahili mshahara bora zaidi",
        "Kilimo kinaweza kuwa suluhisho la ukosefu wa ajira",
        "Teknolojia inabadilisha jinsi tunavyofanya biashara Kenya",
    ],
    "sh": [
        "Siste hii economy ni tight sana hakuna pesa",
        "Manze serikali iko rada lakini watu hawajui",
        "Hii mambo ya tax ni noma bana inaumiza wasee",
        "Youth tunahitaji chance ya kufanya hustle",
        "Nairobi inakuanga busy sana maze haiwezekani",
        "Pesa imeisha bana hii life ni struggle tu",
        "Hii life ni noma lakini tuko rada wasee",
        "Tech ndio future ya Kenya na youth wako ready",
        "Manze bei ya mafuta imepanda tena si poa",
        "Wasee wa bunge wanajipangia salary kubwa na sisi tunapiga hustle",
        "Hii serikali inafanya form tu hawaskii wananchi",
        "Cheki vile Safaricom wamebadilisha game ya tech Kenya",
    ],
}

COUNTIES = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu",
    "Machakos", "Kakamega", "Kilifi", "Meru", "Nyeri",
    "Uasin Gishu", "Bungoma", "Garissa", "Turkana", "Nandi",
]

PLATFORMS = ["twitter", "facebook", "reddit", "tiktok", "telegram"]

HASHTAG_SETS = [
    ["KenyaDecides", "Elections2027"], ["TaxReform", "KenyaFinance"],
    ["Healthcare4All", "UHC"], ["DigitalKenya", "TechKE"],
    ["YouthEmpowerment", "Ajira"], ["RoadConstruction", "Infrastructure"],
    ["WaterIsLife", "MajiNiUhai"], ["EducationReform", "CBC"],
    ["FuelPrices", "CostOfLiving"], ["AntiCorruption", "EACC"],
]


def generate_post(seq: int) -> dict:
    """Generate a single simulated Kenyan social media post."""
    lang = random.choices(["en", "sw", "sh"], weights=[0.45, 0.35, 0.20])[0]
    text = random.choice(TEXTS[lang])

    if random.random() < 0.3:
        tags = random.choice(HASHTAG_SETS)
        text += " " + " ".join(f"#{t}" for t in tags)
        hashtags = tags
    else:
        hashtags = []

    return {
        "post_id": f"prod_{seq}_{int(time.time())}",
        "platform": random.choice(PLATFORMS),
        "text": text,
        "author_handle": f"@user_{random.randint(1000, 9999)}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "county": random.choice(COUNTIES),
        "engagement": {
            "likes": random.randint(0, 5000),
            "retweets": random.randint(0, 2000),
            "replies": random.randint(0, 500),
            "shares": random.randint(0, 1000),
        },
        "hashtags": hashtags,
        "mentions": [],
    }


async def run_producer(
    broker: str,
    interval: float,
    count: int | None,
    continuous: bool,
):
    """Produce messages to the raw-posts Kafka topic."""
    producer = AIOKafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    logger.info(f"Connecting to Kafka broker at {broker} ...")
    await producer.start()
    logger.info(f"Producer connected. Publishing to topic: {TOPIC_RAW}")

    seq = 0
    try:
        while True:
            post = generate_post(seq)
            await producer.send_and_wait(TOPIC_RAW, post)
            logger.info(
                f"[{seq:>5}] Produced → {TOPIC_RAW}  |  "
                f"{post['platform']:<10} | {post['county']:<12} | "
                f"{post['text'][:60]}..."
            )
            seq += 1

            if not continuous and count and seq >= count:
                logger.info(f"Finished producing {count} messages.")
                break

            jitter = random.uniform(0.5, 1.5) * interval
            await asyncio.sleep(jitter)

    except KeyboardInterrupt:
        logger.info("Producer interrupted by user.")
    finally:
        await producer.stop()
        logger.info(f"Producer stopped. Total messages produced: {seq}")


def main():
    parser = argparse.ArgumentParser(description="SentiKenya Kafka Producer")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka bootstrap server")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between messages")
    parser.add_argument("--count", type=int, default=50, help="Number of messages to produce")
    parser.add_argument("--continuous", action="store_true", help="Run indefinitely")
    args = parser.parse_args()

    asyncio.run(run_producer(args.broker, args.interval, args.count, args.continuous))


if __name__ == "__main__":
    main()
