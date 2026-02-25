"""
SentiKenya Ingestion Manager
===============================
Orchestrates all social media connectors and feeds data into the
processing pipeline via Kafka (or direct processing in dev mode).

Architecture:
                ┌──────────────┐
                │   Twitter    │──┐
                │  (stream)    │  │
                └──────────────┘  │     ┌─────────────┐     ┌───────────┐
                ┌──────────────┐  ├────▶│  Ingestion   │────▶│   Kafka   │
                │   Reddit     │──┤     │   Manager    │     │  (queue)  │
                │  (polling)   │  │     └──────┬──────┘     └─────┬─────┘
                └──────────────┘  │            │                  │
                ┌──────────────┐  │            ▼                  ▼
                │  Facebook    │──┘     ┌─────────────┐     ┌───────────┐
                │  (polling)   │        │  Direct NLP  │     │ NLP Worker│
                └──────────────┘        │  (dev mode)  │     │ (prod)    │
                                        └─────────────┘     └───────────┘

Features:
- Async parallel ingestion from all platforms
- Automatic connector health monitoring
- Graceful shutdown with drain
- Dev mode: processes posts directly (no Kafka needed)
- Metrics dashboard for monitoring
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Optional, Dict, List, Callable

from app.models.schemas import RawPost
from app.ingestion.base import BaseConnector, KenyaTrackingConfig
from app.ingestion.twitter_connector import TwitterConnector
from app.ingestion.reddit_connector import RedditConnector
from app.ingestion.facebook_connector import FacebookConnector

logger = logging.getLogger(__name__)


class IngestionManager:
    """
    Central manager for all social media ingestion.

    Responsibilities:
    - Create and configure connectors based on available API keys
    - Run all connectors concurrently
    - Push ingested posts to Kafka or process directly
    - Monitor connector health
    - Handle graceful shutdown
    """

    def __init__(
        self,
        # Reddit (default — works with no API keys)
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        subreddits: Optional[List[str]] = None,
        reddit_poll_interval: int = 120,

        # Optional: Twitter/Facebook (require API keys)
        twitter_bearer_token: Optional[str] = None,
        twitter_mode: str = "search",
        facebook_access_token: Optional[str] = None,
        facebook_pages: Optional[List[str]] = None,

        # Processing
        kafka_service=None,
        kafka_topic: str = "sentikenya.raw.posts",
        direct_processor: Optional[Callable] = None,

        # Config
        config: KenyaTrackingConfig = None,
    ):
        self.config = config or KenyaTrackingConfig()
        if subreddits:
            self.config.subreddits = subreddits
        self.kafka_service = kafka_service
        self.kafka_topic = kafka_topic
        self.direct_processor = direct_processor

        self._connectors: Dict[str, BaseConnector] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

        # Global metrics
        self.metrics = {
            "total_ingested": 0,
            "total_processed": 0,
            "total_errors": 0,
            "started_at": None,
            "platforms_active": [],
        }

        # ── Reddit (always available — no keys needed) ──
        self._connectors["reddit"] = RedditConnector(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            config=self.config,
            poll_interval=reddit_poll_interval,
        )
        auth_mode = "OAuth" if reddit_client_id else "public"
        logger.info(f"[ingestion] Reddit connector registered ({auth_mode}, subs: {self.config.subreddits})")

        # ── Optional: Twitter ──
        if twitter_bearer_token:
            self._connectors["twitter"] = TwitterConnector(
                bearer_token=twitter_bearer_token,
                config=self.config,
                mode=twitter_mode,
            )
            logger.info(f"[ingestion] Twitter connector registered ({twitter_mode} mode)")

        # ── Optional: Facebook ──
        if facebook_access_token:
            self._connectors["facebook"] = FacebookConnector(
                access_token=facebook_access_token,
                page_ids=facebook_pages,
                config=self.config,
            )
            logger.info("[ingestion] Facebook connector registered")

    async def start(self):
        """Start all connectors concurrently."""
        if not self._connectors:
            logger.warning("[ingestion] No connectors to start")
            return

        self._running = True
        self.metrics["started_at"] = datetime.utcnow().isoformat()
        self.metrics["platforms_active"] = list(self._connectors.keys())

        logger.info(
            f"[ingestion] Starting {len(self._connectors)} connectors: "
            f"{', '.join(self._connectors.keys())}"
        )

        # Launch each connector as a concurrent task
        for name, connector in self._connectors.items():
            task = asyncio.create_task(
                self._run_connector(name, connector),
                name=f"ingestion-{name}",
            )
            self._tasks[name] = task

    async def _run_connector(self, name: str, connector: BaseConnector):
        """
        Run a single connector's stream and process each post.

        Each post is either:
        - Pushed to Kafka for async processing (production)
        - Processed directly via the NLP pipeline (development)
        """
        logger.info(f"[ingestion:{name}] Starting stream...")

        try:
            async for post in connector.stream():
                if not self._running:
                    break

                try:
                    await self._handle_post(post, name)
                    self.metrics["total_ingested"] += 1
                except Exception as e:
                    self.metrics["total_errors"] += 1
                    logger.error(f"[ingestion:{name}] Failed to handle post: {e}")

        except asyncio.CancelledError:
            logger.info(f"[ingestion:{name}] Task cancelled")
        except Exception as e:
            logger.error(f"[ingestion:{name}] Fatal error: {e}")

    async def _handle_post(self, post: RawPost, source: str):
        """
        Route an ingested post to processing.

        Priority:
        1. Kafka (production — async, scalable)
        2. Direct processor (dev — synchronous, no Kafka needed)
        3. Log only (no processing configured)
        """
        post_dict = post.model_dump(mode="json")

        # Route 1: Kafka
        if self.kafka_service and hasattr(self.kafka_service, 'produce'):
            try:
                await self.kafka_service.produce(self.kafka_topic, post_dict)
                self.metrics["total_processed"] += 1
                return
            except Exception as e:
                logger.warning(f"[ingestion] Kafka publish failed: {e}, trying direct")

        # Route 2: Direct processing
        if self.direct_processor:
            try:
                result = self.direct_processor(post)
                self.metrics["total_processed"] += 1

                # Log high-engagement or strong-sentiment posts
                if hasattr(result, 'sentiment'):
                    score = abs(result.sentiment.score)
                    engagement = sum(post.engagement.values())
                    if score > 0.7 or engagement > 100:
                        logger.info(
                            f"[ingestion:{source}] 🔥 "
                            f"[{result.sentiment.label.value}:{result.sentiment.score:+.2f}] "
                            f"({engagement} engagement) "
                            f"\"{post.text[:80]}...\""
                        )
                return
            except Exception as e:
                logger.error(f"[ingestion] Direct processing failed: {e}")

        # Route 3: Log only
        logger.debug(
            f"[ingestion:{source}] Ingested (no processor): "
            f"[{post.platform.value}] \"{post.text[:60]}...\""
        )

    async def stop(self):
        """Gracefully stop all connectors and drain remaining posts."""
        logger.info("[ingestion] Stopping all connectors...")
        self._running = False

        # Signal all connectors to stop
        for connector in self._connectors.values():
            connector.stop()

        # Cancel and wait for all tasks
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._tasks.clear()
        logger.info("[ingestion] All connectors stopped")

    def get_status(self) -> dict:
        """Full ingestion status with per-connector metrics."""
        connector_status = {}
        for name, connector in self._connectors.items():
            task = self._tasks.get(name)
            connector_status[name] = {
                **connector.get_metrics(),
                "task_running": task is not None and not task.done() if task else False,
            }

        return {
            "running": self._running,
            "connectors": connector_status,
            "global": self.metrics,
        }

    @property
    def active_connectors(self) -> List[str]:
        """List of currently active connector names."""
        return [
            name for name, task in self._tasks.items()
            if task and not task.done()
        ]
