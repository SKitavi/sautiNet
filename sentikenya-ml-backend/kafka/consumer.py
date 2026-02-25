#!/usr/bin/env python3
"""
SentiKenya Kafka Consumer
==========================
Standalone consumer script that reads messages from the ``raw-posts`` topic,
logs each message, and optionally forwards cleaned text to ``preprocessed-posts``.

Usage:
    python -m kafka.consumer                              # consume and log
    python -m kafka.consumer --broker kafka:29092          # custom broker
    python -m kafka.consumer --forward                     # also forward to preprocessed-posts
    python -m kafka.consumer --group test-group --count 20 # limited run
"""

import argparse
import asyncio
import json
import logging
import sys
import re
from datetime import datetime

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("sentikenya.consumer")

# ── Kafka Topics ──
TOPIC_RAW = "raw-posts"
TOPIC_PREPROCESSED = "preprocessed-posts"


def basic_preprocess(text: str) -> str:
    """Lightweight text cleaning before forwarding to the preprocessed topic."""
    text = re.sub(r"https?://\S+", "", text)       # strip URLs
    text = re.sub(r"@\w+", "", text)                # strip mentions
    text = re.sub(r"\s+", " ", text).strip()        # normalise whitespace
    return text


async def run_consumer(
    broker: str,
    group_id: str,
    forward: bool,
    count: int | None,
):
    """Consume messages from raw-posts, log them, optionally forward."""
    consumer = AIOKafkaConsumer(
        TOPIC_RAW,
        bootstrap_servers=broker,
        group_id=group_id,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    producer = None
    if forward:
        producer = AIOKafkaProducer(
            bootstrap_servers=broker,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )

    logger.info(f"Connecting consumer to {broker}, group={group_id} ...")
    await consumer.start()
    if producer:
        await producer.start()
        logger.info(f"Forwarding enabled → {TOPIC_PREPROCESSED}")

    logger.info(f"Consumer listening on topic: {TOPIC_RAW}")

    processed = 0
    try:
        async for message in consumer:
            post = message.value
            post_id = post.get("post_id", "?")
            platform = post.get("platform", "?")
            county = post.get("county", "?")
            text = post.get("text", "")

            logger.info(
                f"[{processed:>5}] Consumed ← {TOPIC_RAW}  |  "
                f"offset={message.offset:<6} | {platform:<10} | {county:<12} | "
                f"{text[:60]}..."
            )

            # Forward preprocessed text
            if producer:
                cleaned = basic_preprocess(text)
                fwd_msg = {
                    "post_id": post_id,
                    "platform": platform,
                    "county": county,
                    "original_text": text,
                    "cleaned_text": cleaned,
                    "hashtags": post.get("hashtags", []),
                    "timestamp": post.get("timestamp", datetime.utcnow().isoformat()),
                    "engagement": post.get("engagement", {}),
                }
                await producer.send_and_wait(TOPIC_PREPROCESSED, fwd_msg)
                logger.debug(f"  → Forwarded to {TOPIC_PREPROCESSED}")

            processed += 1
            if count and processed >= count:
                logger.info(f"Reached message limit ({count}). Stopping.")
                break

    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    finally:
        await consumer.stop()
        if producer:
            await producer.stop()
        logger.info(f"Consumer stopped. Total messages consumed: {processed}")


def main():
    parser = argparse.ArgumentParser(description="SentiKenya Kafka Consumer")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka bootstrap server")
    parser.add_argument("--group", default="sentikenya-consumers", help="Consumer group ID")
    parser.add_argument("--forward", action="store_true", help="Forward to preprocessed-posts topic")
    parser.add_argument("--count", type=int, default=None, help="Stop after N messages (default: run forever)")
    args = parser.parse_args()

    asyncio.run(run_consumer(args.broker, args.group, args.forward, args.count))


if __name__ == "__main__":
    main()
