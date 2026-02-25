#!/usr/bin/env python3
"""
SentiKenya Kafka Topic Initializer
====================================
Creates the required Kafka topics for the SentiKenya pipeline:
  - raw-posts           : Ingested social media posts
  - preprocessed-posts  : Cleaned / tokenized text ready for inference
  - sentiment-results   : Final sentiment predictions

Usage:
    python -m kafka.init_topics
    python -m kafka.init_topics --broker kafka:29092 --partitions 6
"""

import argparse
import asyncio
import logging

from aiokafka.admin import AIOKafkaAdminClient, NewTopic

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("sentikenya.topics")

REQUIRED_TOPICS = [
    {"name": "raw-posts", "partitions": 3, "description": "Ingested social media posts from all platforms"},
    {"name": "preprocessed-posts", "partitions": 3, "description": "Cleaned and tokenized text ready for inference"},
    {"name": "sentiment-results", "partitions": 3, "description": "Final sentiment analysis predictions"},
]


async def create_topics(broker: str, partitions_override: int | None = None):
    admin = AIOKafkaAdminClient(bootstrap_servers=broker)
    await admin.start()

    try:
        existing = await admin.list_topics()
        logger.info(f"Existing topics: {existing}")

        topics_to_create = []
        for t in REQUIRED_TOPICS:
            name = t["name"]
            parts = partitions_override or t["partitions"]
            if name in existing:
                logger.info(f"  ✓ Topic '{name}' already exists — skipping")
            else:
                topics_to_create.append(
                    NewTopic(name=name, num_partitions=parts, replication_factor=1)
                )
                logger.info(f"  + Creating topic '{name}' ({parts} partitions)")

        if topics_to_create:
            await admin.create_topics(topics_to_create)
            logger.info(f"Created {len(topics_to_create)} topic(s) successfully.")
        else:
            logger.info("All required topics already exist.")

    finally:
        await admin.close()


def main():
    parser = argparse.ArgumentParser(description="Create SentiKenya Kafka topics")
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--partitions", type=int, default=None, help="Override partition count")
    args = parser.parse_args()
    asyncio.run(create_topics(args.broker, args.partitions))


if __name__ == "__main__":
    main()
