#!/usr/bin/env python3
"""
SentiKenya Kafka Inference Worker
===================================
Connects the inference pipeline to the Kafka message stream.

Flow:
    preprocessed-posts → NLP Pipeline → sentiment-results

Reads cleaned text from ``preprocessed-posts``, runs it through the
sentiment analysis pipeline, and publishes results to ``sentiment-results``.

Usage:
    python -m app.workers.kafka_inference_worker
    python -m app.workers.kafka_inference_worker --broker kafka:29092
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("sentikenya.inference_worker")

TOPIC_IN = "preprocessed-posts"
TOPIC_OUT = "sentiment-results"


async def run_inference_worker(
    broker: str = "localhost:9092",
    group_id: str = "sentikenya-inference",
    node_id: str = "NBO-01",
    count: int | None = None,
):
    """
    Consume from preprocessed-posts, run inference, publish to sentiment-results.
    """
    # Late import to avoid circular deps
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from app.ml.pipeline import NLPPipeline

    # Initialize pipeline
    logger.info(f"Initializing NLP pipeline on node {node_id}...")
    pipeline = NLPPipeline(
        sheng_lexicon_path="./data/sheng_lexicon.json",
        entities_path="./data/kenyan_entities.json",
        node_id=node_id,
    )

    consumer = AIOKafkaConsumer(
        TOPIC_IN,
        bootstrap_servers=broker,
        group_id=group_id,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    await consumer.start()
    await producer.start()
    logger.info(f"Inference worker started: {TOPIC_IN} → {TOPIC_OUT}")

    processed = 0
    total_latency = 0.0

    try:
        async for message in consumer:
            post = message.value
            text = post.get("cleaned_text", post.get("text", ""))
            post_id = post.get("post_id", "unknown")

            t0 = time.time()

            # Run through NLP pipeline
            result = pipeline.process_text(text)

            latency_ms = (time.time() - t0) * 1000
            total_latency += latency_ms

            # Build sentiment result message
            output = {
                "post_id": post_id,
                "text": text[:200],
                "label": result.sentiment.label.value,
                "confidence": round(result.sentiment.confidence, 4),
                "score": round(result.sentiment.score, 4),
                "language": result.language.detected_language.value,
                "topics": result.topics.primary_topic,
                "is_political": result.topics.is_political,
                "county": post.get("county", "Unknown"),
                "platform": post.get("platform", "unknown"),
                "node_id": node_id,
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await producer.send_and_wait(TOPIC_OUT, output)
            processed += 1

            avg_latency = total_latency / processed

            logger.info(
                f"[{processed:>5}] {post_id} → {output['label']:>8} "
                f"(conf={output['confidence']:.3f}, lang={output['language']}) "
                f"│ {latency_ms:.1f}ms │ avg={avg_latency:.1f}ms"
            )

            if count and processed >= count:
                break

    except KeyboardInterrupt:
        logger.info("Worker interrupted.")
    finally:
        await consumer.stop()
        await producer.stop()

        avg = total_latency / max(processed, 1)
        logger.info(
            f"Inference worker stopped. "
            f"Processed: {processed}, Avg latency: {avg:.1f}ms, "
            f"Target: <200ms → {'✅ PASS' if avg < 200 else '⚠️ EXCEEDS TARGET'}"
        )


def main():
    parser = argparse.ArgumentParser(description="SentiKenya Kafka Inference Worker")
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--group", default="sentikenya-inference")
    parser.add_argument("--node", default="NBO-01")
    parser.add_argument("--count", type=int, default=None, help="Stop after N messages")
    args = parser.parse_args()

    asyncio.run(run_inference_worker(args.broker, args.group, args.node, args.count))


if __name__ == "__main__":
    main()
