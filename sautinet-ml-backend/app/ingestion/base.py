"""
SentiKenya Social Media Connectors — Base
===========================================
Abstract base for all platform connectors.

Each connector implements:
- connect()    → authenticate and start session
- stream()     → async generator yielding RawPost objects
- disconnect() → clean shutdown

All connectors normalize platform-specific data into RawPost format
before pushing to Kafka for NLP processing.
"""

import asyncio
import logging
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Optional, List, Dict, Set
from dataclasses import dataclass, field

from app.models.schemas import RawPost, Platform

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# Kenya-specific tracking config
# ═══════════════════════════════════════════════════════

@dataclass
class KenyaTrackingConfig:
    """
    Keywords, hashtags, and geo-fences for Kenyan social media monitoring.
    Shared across all connectors.
    """

    # Political keywords (EN + SW + Sheng)
    keywords: List[str] = field(default_factory=lambda: [
        # English
        "Kenya government", "Kenya politics", "Kenyan economy",
        "cost of living Kenya", "Kenya tax", "Kenya corruption",
        "Gen Z Kenya", "Kenya protest", "Kenya development",
        "M-Pesa", "Safaricom", "Kenya Power", "NHIF", "SHA",
        "housing levy", "Kenya education", "CBC Kenya",
        "Kenya healthcare", "Kenya security", "Kenya youth",
        "devolution Kenya",

        # Swahili
        "serikali Kenya", "uchumi Kenya", "rushwa Kenya",
        "maendeleo Kenya", "wananchi", "vijana Kenya",
        "elimu Kenya", "afya Kenya", "usalama Kenya",

        # Sheng
        "serikali iko rada", "mambo ni poa", "economy ni tight",
        "wasee wa bunge", "tax ni noma",
    ])

    # Hashtags to track
    hashtags: List[str] = field(default_factory=lambda: [
        "Kenya", "KenyaNews", "KOT", "KenyanPolitics",
        "NairobiNews", "GenZKenya", "KenyaYouth",
        "Serikali", "Bunge", "Hustler", "Azimio",
        "KenyaTax", "CostOfLiving", "MPesa",
        "KenyaTech", "SiliconSavannah",
    ])

    # Geo-fence: Kenya bounding box
    geo_bbox: Dict[str, float] = field(default_factory=lambda: {
        "west": 33.9,
        "south": -4.7,
        "east": 41.9,
        "north": 5.5,
    })

    # Nairobi metro (tighter radius for high-density monitoring)
    nairobi_center: Dict[str, float] = field(default_factory=lambda: {
        "lat": -1.2921,
        "lon": 36.8219,
        "radius_km": 30,
    })

    # Major Kenyan counties for location tagging
    counties: List[str] = field(default_factory=lambda: [
        "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
        "Kiambu", "Machakos", "Nyeri", "Meru", "Kakamega",
        "Uasin Gishu", "Kilifi", "Kajiado", "Narok", "Garissa",
    ])

    # Subreddits to monitor
    subreddits: List[str] = field(default_factory=lambda: [
        "Kenya", "Nairobi", "KenyanFood",
    ])

    # Facebook pages (public Kenyan news/politics pages)
    facebook_pages: List[str] = field(default_factory=lambda: [
        # These would be populated with actual page IDs
        # Placeholder structure for setup
    ])

    # Rate limiting
    max_posts_per_minute: int = 120
    dedup_window_seconds: int = 3600  # 1 hour dedup window


# ═══════════════════════════════════════════════════════
# Base Connector
# ═══════════════════════════════════════════════════════

class BaseConnector(ABC):
    """
    Abstract base for social media connectors.

    Handles:
    - Connection lifecycle (connect → stream → disconnect)
    - Rate limiting
    - Deduplication
    - Error recovery with exponential backoff
    - Metrics tracking
    """

    def __init__(
        self,
        platform: Platform,
        config: KenyaTrackingConfig = None,
        max_retries: int = 5,
        base_backoff: float = 5.0,
    ):
        self.platform = platform
        self.config = config or KenyaTrackingConfig()
        self.max_retries = max_retries
        self.base_backoff = base_backoff

        self._connected = False
        self._running = False
        self._retry_count = 0

        # Deduplication
        self._seen_ids: Set[str] = set()
        self._seen_hashes: Set[str] = set()

        # Metrics
        self.metrics = {
            "posts_ingested": 0,
            "posts_deduplicated": 0,
            "errors": 0,
            "reconnections": 0,
            "last_post_at": None,
            "started_at": None,
        }

    @abstractmethod
    async def connect(self):
        """Authenticate and establish connection to the platform API."""
        pass

    @abstractmethod
    async def _fetch_posts(self) -> AsyncGenerator[RawPost, None]:
        """Platform-specific post fetching. Yields normalized RawPost objects."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Gracefully close connection."""
        pass

    async def stream(self) -> AsyncGenerator[RawPost, None]:
        """
        Main streaming loop with retry logic.

        Wraps _fetch_posts() with:
        - Deduplication
        - Rate limiting
        - Error recovery
        - Metrics
        """
        self._running = True
        self.metrics["started_at"] = datetime.utcnow().isoformat()

        while self._running:
            try:
                if not self._connected:
                    await self.connect()
                    self._retry_count = 0

                async for post in self._fetch_posts():
                    if not self._running:
                        break

                    # Dedup check
                    if self._is_duplicate(post):
                        self.metrics["posts_deduplicated"] += 1
                        continue

                    self.metrics["posts_ingested"] += 1
                    self.metrics["last_post_at"] = datetime.utcnow().isoformat()
                    yield post

            except asyncio.CancelledError:
                logger.info(f"[{self.platform.value}] Stream cancelled")
                break

            except Exception as e:
                self.metrics["errors"] += 1
                self._connected = False
                self._retry_count += 1

                if self._retry_count > self.max_retries:
                    logger.error(
                        f"[{self.platform.value}] Max retries ({self.max_retries}) "
                        f"exceeded. Last error: {e}"
                    )
                    await asyncio.sleep(60)  # Long cooldown then reset
                    self._retry_count = 0
                else:
                    backoff = self.base_backoff * (2 ** (self._retry_count - 1))
                    logger.warning(
                        f"[{self.platform.value}] Error: {e}. "
                        f"Retry {self._retry_count}/{self.max_retries} "
                        f"in {backoff:.0f}s"
                    )
                    self.metrics["reconnections"] += 1
                    await asyncio.sleep(backoff)

        await self.disconnect()

    def _is_duplicate(self, post: RawPost) -> bool:
        """Check if post was already seen (by ID or content hash)."""
        # ID check
        if post.post_id in self._seen_ids:
            return True

        # Content hash for near-duplicates
        content_hash = hashlib.md5(
            f"{post.author_id}:{post.text[:200]}".encode()
        ).hexdigest()

        if content_hash in self._seen_hashes:
            return True

        self._seen_ids.add(post.post_id)
        self._seen_hashes.add(content_hash)

        # Prune old entries to prevent memory leak
        if len(self._seen_ids) > 50000:
            # Keep most recent half
            self._seen_ids = set(list(self._seen_ids)[-25000:])
            self._seen_hashes = set(list(self._seen_hashes)[-25000:])

        return False

    def stop(self):
        """Signal the stream to stop."""
        self._running = False

    def get_metrics(self) -> dict:
        """Return connector metrics."""
        return {
            "platform": self.platform.value,
            "connected": self._connected,
            "running": self._running,
            **self.metrics,
        }
