"""
SentiKenya NLP Worker
======================
Background worker that consumes posts from Kafka,
processes them through the NLP pipeline, stores results,
and broadcasts to WebSocket clients.
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from collections import defaultdict

from app.ml.pipeline import NLPPipeline
from app.services.kafka_service import KafkaService
from app.services.ipfs_service import IPFSService
from app.services.broadcast_service import ConnectionManager
from app.models.schemas import ProcessedPost, CountySentiment, Language

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    Real-time sentiment aggregator that maintains running statistics
    per county, topic, and time window.
    """

    def __init__(self, window_minutes: int = 15):
        self.window_minutes = window_minutes
        self._county_data: Dict[str, List[ProcessedPost]] = defaultdict(list)
        self._topic_data: Dict[str, List[ProcessedPost]] = defaultdict(list)
        self._all_posts: List[ProcessedPost] = []
        self._total_processed = 0
        self._start_time = datetime.utcnow()

    def add_post(self, post: ProcessedPost):
        """Add a processed post to the aggregator."""
        self._all_posts.append(post)
        self._total_processed += 1

        # Index by county
        county = post.raw_post.county or "Unknown"
        self._county_data[county].append(post)

        # Index by topic
        topic = post.topics.primary_topic
        self._topic_data[topic].append(post)

        # Prune old data outside window
        self._prune_old_data()

    def _prune_old_data(self):
        """Remove posts outside the aggregation window."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)

        self._all_posts = [p for p in self._all_posts if p.processed_at > cutoff]

        for county in list(self._county_data.keys()):
            self._county_data[county] = [
                p for p in self._county_data[county] if p.processed_at > cutoff
            ]
            if not self._county_data[county]:
                del self._county_data[county]

        for topic in list(self._topic_data.keys()):
            self._topic_data[topic] = [
                p for p in self._topic_data[topic] if p.processed_at > cutoff
            ]
            if not self._topic_data[topic]:
                del self._topic_data[topic]

    def get_county_sentiment(self, county: str) -> Optional[CountySentiment]:
        """Get aggregated sentiment for a specific county."""
        posts = self._county_data.get(county, [])
        if not posts:
            return None

        sentiments = [p.sentiment.score for p in posts]
        avg_sentiment = sum(sentiments) / len(sentiments)

        # Language distribution
        lang_counts = defaultdict(int)
        for p in posts:
            lang_counts[p.language.detected_language.value] += 1
        total = len(posts)
        lang_dist = {k: round(v / total, 4) for k, v in lang_counts.items()}

        # Trending topics in this county
        topic_counts = defaultdict(int)
        for p in posts:
            topic_counts[p.topics.primary_topic] += 1
        trending = sorted(topic_counts, key=topic_counts.get, reverse=True)[:5]

        # Dominant language
        dominant = max(lang_counts, key=lang_counts.get) if lang_counts else "en"

        # Engagement-weighted sentiment
        engagement_scores = [p.raw_post.engagement_score for p in posts]
        total_engagement = sum(engagement_scores) or 1
        weighted_sentiment = sum(
            p.sentiment.score * (p.raw_post.engagement_score / total_engagement)
            for p in posts
        )

        return CountySentiment(
            county=county,
            time_window_start=datetime.utcnow() - timedelta(minutes=self.window_minutes),
            time_window_end=datetime.utcnow(),
            overall_sentiment=round(avg_sentiment, 4),
            positive_count=sum(1 for s in sentiments if s > 0.25),
            negative_count=sum(1 for s in sentiments if s < -0.25),
            neutral_count=sum(1 for s in sentiments if -0.25 <= s <= 0.25),
            total_posts=len(posts),
            trending_topics=trending,
            dominant_language=Language(dominant),
            language_distribution=lang_dist,
            engagement_weighted_sentiment=round(weighted_sentiment, 4),
        )

    def get_all_county_sentiments(self) -> List[CountySentiment]:
        """Get sentiments for all counties with data."""
        results = []
        for county in self._county_data:
            cs = self.get_county_sentiment(county)
            if cs:
                results.append(cs)
        return sorted(results, key=lambda c: c.total_posts, reverse=True)

    def get_overall_stats(self) -> dict:
        """Get overall system statistics."""
        all_sentiments = [p.sentiment.score for p in self._all_posts]
        avg = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0

        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        rate = self._total_processed / max(uptime, 1) * 60  # posts per minute

        return {
            "total_processed": self._total_processed,
            "window_posts": len(self._all_posts),
            "overall_sentiment": round(avg, 4),
            "positive_pct": round(sum(1 for s in all_sentiments if s > 0.25) / max(len(all_sentiments), 1) * 100, 1),
            "negative_pct": round(sum(1 for s in all_sentiments if s < -0.25) / max(len(all_sentiments), 1) * 100, 1),
            "neutral_pct": round(sum(1 for s in all_sentiments if -0.25 <= s <= 0.25) / max(len(all_sentiments), 1) * 100, 1),
            "active_counties": len(self._county_data),
            "active_topics": len(self._topic_data),
            "processing_rate_per_min": round(rate, 1),
            "uptime_seconds": round(uptime),
        }

    def get_trending_topics(self, limit: int = 10) -> List[dict]:
        """Get trending topics across all counties."""
        topic_stats = []
        for topic, posts in self._topic_data.items():
            sentiments = [p.sentiment.score for p in posts]
            topic_stats.append({
                "topic": topic,
                "count": len(posts),
                "avg_sentiment": round(sum(sentiments) / len(sentiments), 4) if sentiments else 0,
                "is_political": any(p.topics.is_political for p in posts),
            })
        return sorted(topic_stats, key=lambda t: t["count"], reverse=True)[:limit]


class NLPWorker:
    """
    Background worker that:
    1. Consumes posts from Kafka stream
    2. Processes through NLP pipeline
    3. Pins raw data to IPFS
    4. Aggregates county/topic sentiments
    5. Broadcasts results via WebSocket
    6. Detects sentiment anomalies/alerts
    """

    def __init__(
        self,
        pipeline: NLPPipeline,
        kafka: KafkaService,
        ipfs: IPFSService,
        broadcaster: ConnectionManager,
        alert_threshold: float = 0.5,
    ):
        self.pipeline = pipeline
        self.kafka = kafka
        self.ipfs = ipfs
        self.broadcaster = broadcaster
        self.aggregator = SentimentAggregator(window_minutes=15)
        self.alert_threshold = alert_threshold
        self._running = False
        self._processed_count = 0
        self._error_count = 0

    async def start(self):
        """Start the NLP processing worker."""
        logger.info("NLP Worker starting...")
        self._running = True

        # Process posts from Kafka stream
        async for raw_post in self.kafka.consume_stream():
            if not self._running:
                break

            try:
                # Process through NLP pipeline
                processed = self.pipeline.process_post(raw_post)
                self._processed_count += 1

                # Pin to IPFS
                ipfs_cid = await self.ipfs.pin_post(
                    raw_post.model_dump(mode="json")
                )
                if ipfs_cid:
                    processed.ipfs_cid = ipfs_cid

                # Aggregate
                self.aggregator.add_post(processed)

                # Broadcast to WebSocket clients
                await self._broadcast_post(processed)

                # Check for alerts
                await self._check_alerts(processed)

                # Periodic stats broadcast
                if self._processed_count % 10 == 0:
                    await self._broadcast_stats()

                logger.debug(
                    f"Processed post {raw_post.post_id}: "
                    f"lang={processed.language.detected_language.value}, "
                    f"sentiment={processed.sentiment.score:.3f}, "
                    f"topic={processed.topics.primary_topic} "
                    f"({processed.total_processing_time_ms:.1f}ms)"
                )

            except Exception as e:
                self._error_count += 1
                logger.error(f"Worker error processing post: {e}", exc_info=True)

    async def stop(self):
        """Stop the worker gracefully."""
        logger.info("NLP Worker stopping...")
        self._running = False

    async def _broadcast_post(self, post: ProcessedPost):
        """Broadcast processed post data to WebSocket clients."""
        broadcast_data = {
            "post_id": post.raw_post.post_id,
            "text": post.raw_post.text,
            "platform": post.raw_post.platform.value,
            "county": post.raw_post.county,
            "language": post.language.detected_language.value,
            "language_confidence": post.language.confidence,
            "sentiment_label": post.sentiment.label.value,
            "sentiment_score": post.sentiment.score,
            "sentiment_confidence": post.sentiment.confidence,
            "topic": post.topics.primary_topic,
            "is_political": post.topics.is_political,
            "entities_count": len(post.entities.entities),
            "engagement_score": post.raw_post.engagement_score,
            "processing_time_ms": post.total_processing_time_ms,
            "node_id": post.node_id,
            "ipfs_cid": post.ipfs_cid,
        }
        await self.broadcaster.broadcast_processed_post(broadcast_data)

    async def _broadcast_stats(self):
        """Broadcast aggregated statistics."""
        stats = self.aggregator.get_overall_stats()
        stats["worker_errors"] = self._error_count
        stats["ipfs_stats"] = self.ipfs.get_stats()
        stats["ws_stats"] = self.broadcaster.get_stats()
        await self.broadcaster.broadcast_stats(stats)

    async def _check_alerts(self, post: ProcessedPost):
        """Check for sentiment anomalies that should trigger alerts."""
        # Alert on strong negative sentiment for political topics
        if (
            post.topics.is_political
            and post.sentiment.score < -self.alert_threshold
            and post.raw_post.engagement_score > 100
        ):
            await self.broadcaster.broadcast_alert({
                "type": "negative_political_spike",
                "county": post.raw_post.county,
                "topic": post.topics.primary_topic,
                "sentiment": post.sentiment.score,
                "engagement": post.raw_post.engagement_score,
                "text_preview": post.raw_post.text[:100],
            })

    def get_worker_stats(self) -> dict:
        """Return worker statistics."""
        return {
            "running": self._running,
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "error_rate": round(
                self._error_count / max(self._processed_count, 1) * 100, 2
            ),
            "aggregator_stats": self.aggregator.get_overall_stats(),
        }
