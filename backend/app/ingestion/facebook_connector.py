"""
SentiKenya Facebook Connector
================================
Monitors public Kenyan Facebook pages and groups for sentiment data.

Facebook Graph API v18.0:
- Page public content: page posts, comments, reactions
- Page search: find public pages by keyword
- Rate limit: 200 calls/hour per user token, 4800/hour per app token

Requires: FACEBOOK_ACCESS_TOKEN (Page or App token) in env/settings.

Note: Facebook has progressively restricted public data access.
This connector focuses on what's available:
- Public page posts (pages must be public)
- Comments on public posts
- Reaction counts
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, List, Dict

import httpx

from app.models.schemas import RawPost, Platform
from app.ingestion.base import BaseConnector, KenyaTrackingConfig

logger = logging.getLogger(__name__)


# Major public Kenyan pages to monitor
DEFAULT_KENYA_PAGES = {
    # News outlets
    "NationMedia": "Daily Nation - Kenya's leading newspaper",
    "StandardKenya": "The Standard - Kenyan news",
    "CitizenTVKenya": "Citizen TV - Kenyan broadcast",
    "KTNNewsKE": "KTN News",
    "TV47Kenya": "TV47 Kenya",
    # Government
    "StateHouseKenya": "State House Kenya",
    "ParliamentKE": "Parliament of Kenya",
    # Institutions
    "SafaricomPLC": "Safaricom",
    "KenyaPower": "Kenya Power",
    "CBaboraKenya": "Central Bank of Kenya",
    # Civil society
    "Transparency.International.Kenya": "TI Kenya",
}


class FacebookConnector(BaseConnector):
    """
    Facebook Graph API connector for Kenyan public page monitoring.

    Monitors:
    - Public page posts from Kenyan news outlets and institutions
    - Comments on those posts
    - Reaction metrics (like, love, angry, sad, etc.)
    """

    GRAPH_URL = "https://graph.facebook.com/v18.0"

    def __init__(
        self,
        access_token: str,
        page_ids: Optional[List[str]] = None,
        config: KenyaTrackingConfig = None,
        poll_interval: int = 300,  # 5 minutes between polls
    ):
        super().__init__(platform=Platform.FACEBOOK, config=config)
        self.access_token = access_token
        self.page_ids = page_ids or list(DEFAULT_KENYA_PAGES.keys())
        self.poll_interval = poll_interval

        self._client: Optional[httpx.AsyncClient] = None
        self._last_seen: Dict[str, str] = {}  # page_id → last post timestamp

    async def connect(self):
        """Initialize client and verify token."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        # Verify token
        try:
            resp = await self._client.get(
                f"{self.GRAPH_URL}/me",
                params={"access_token": self.access_token},
            )
            if resp.status_code != 200:
                error = resp.json().get("error", {})
                raise ConnectionError(
                    f"Facebook auth failed: {error.get('message', resp.status_code)}"
                )
            me = resp.json()
            logger.info(f"[facebook] Authenticated as: {me.get('name', 'unknown')}")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot reach Facebook API: {e}")

        self._connected = True
        logger.info(f"[facebook] Connected, monitoring {len(self.page_ids)} pages")

    async def _fetch_posts(self) -> AsyncGenerator[RawPost, None]:
        """Poll monitored pages for new posts and comments."""
        while self._running:
            for page_id in self.page_ids:
                try:
                    # Fetch page posts
                    async for post in self._fetch_page_posts(page_id):
                        yield post

                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.warning(f"[facebook] Error fetching {page_id}: {e}")

                # Respect rate limits
                await asyncio.sleep(2)

            await asyncio.sleep(self.poll_interval)

    async def _fetch_page_posts(
        self,
        page_id: str,
        limit: int = 10,
    ) -> AsyncGenerator[RawPost, None]:
        """Fetch recent posts from a public Facebook page."""
        fields = (
            "id,message,created_time,full_picture,permalink_url,"
            "shares,reactions.summary(total_count),"
            "comments.summary(total_count),"
            "reactions.type(LIKE).summary(total_count).as(like_count),"
            "reactions.type(LOVE).summary(total_count).as(love_count),"
            "reactions.type(ANGRY).summary(total_count).as(angry_count),"
            "reactions.type(SAD).summary(total_count).as(sad_count)"
        )

        params = {
            "fields": fields,
            "limit": limit,
            "access_token": self.access_token,
        }

        # Use since parameter if we've seen posts before
        since = self._last_seen.get(page_id)
        if since:
            params["since"] = since

        resp = await self._client.get(
            f"{self.GRAPH_URL}/{page_id}/posts",
            params=params,
        )

        if resp.status_code == 400:
            error = resp.json().get("error", {})
            if error.get("code") == 190:
                raise ConnectionError("Facebook access token expired")
            logger.warning(f"[facebook] Page {page_id} error: {error.get('message')}")
            return

        if resp.status_code != 200:
            logger.error(f"[facebook] {page_id} returned {resp.status_code}")
            return

        data = resp.json()
        posts = data.get("data", [])

        for post_data in posts:
            post = self._parse_page_post(post_data, page_id)
            if post:
                yield post

            # Also fetch comments on this post
            async for comment in self._fetch_post_comments(post_data["id"]):
                yield comment

        # Track last seen timestamp
        if posts:
            self._last_seen[page_id] = posts[0].get("created_time", "")

    async def _fetch_post_comments(
        self,
        post_id: str,
        limit: int = 25,
    ) -> AsyncGenerator[RawPost, None]:
        """Fetch comments on a specific post."""
        params = {
            "fields": "id,message,created_time,from,like_count,comment_count",
            "limit": limit,
            "order": "reverse_chronological",
            "access_token": self.access_token,
        }

        resp = await self._client.get(
            f"{self.GRAPH_URL}/{post_id}/comments",
            params=params,
        )

        if resp.status_code != 200:
            return

        data = resp.json()
        for comment_data in data.get("data", []):
            comment = self._parse_comment(comment_data, post_id)
            if comment:
                yield comment

    def _parse_page_post(self, data: dict, page_id: str) -> Optional[RawPost]:
        """Normalize a Facebook page post into RawPost."""
        text = data.get("message", "").strip()
        if not text or len(text) < 10:
            return None

        # Timestamp
        created_time = data.get("created_time", "")
        try:
            timestamp = datetime.fromisoformat(created_time.replace("+0000", "+00:00"))
        except (ValueError, TypeError):
            timestamp = datetime.utcnow()

        # Engagement metrics
        reactions = data.get("reactions", {}).get("summary", {})
        comments = data.get("comments", {}).get("summary", {})
        shares = data.get("shares", {})

        # Reaction breakdown (sentiment signal)
        like_count = data.get("like_count", {})
        if isinstance(like_count, dict):
            like_count = like_count.get("summary", {}).get("total_count", 0)
        angry_count = data.get("angry_count", {})
        if isinstance(angry_count, dict):
            angry_count = angry_count.get("summary", {}).get("total_count", 0)

        return RawPost(
            post_id=f"fb_{data.get('id', '')}",
            platform=Platform.FACEBOOK,
            text=text[:5000],
            author_id=page_id,
            author_handle=page_id,
            timestamp=timestamp,
            county=self._detect_county(text),
            engagement={
                "likes": reactions.get("total_count", 0),
                "retweets": shares.get("count", 0),
                "replies": comments.get("total_count", 0),
                "shares": shares.get("count", 0),
            },
            raw_metadata={
                "page_id": page_id,
                "permalink": data.get("permalink_url", ""),
                "has_image": bool(data.get("full_picture")),
                "reactions_breakdown": {
                    "like": like_count if isinstance(like_count, int) else 0,
                    "angry": angry_count if isinstance(angry_count, int) else 0,
                },
            },
        )

    def _parse_comment(self, data: dict, parent_post_id: str) -> Optional[RawPost]:
        """Normalize a Facebook comment into RawPost."""
        text = data.get("message", "").strip()
        if not text or len(text) < 10:
            return None

        created_time = data.get("created_time", "")
        try:
            timestamp = datetime.fromisoformat(created_time.replace("+0000", "+00:00"))
        except (ValueError, TypeError):
            timestamp = datetime.utcnow()

        from_data = data.get("from", {})

        return RawPost(
            post_id=f"fc_{data.get('id', '')}",
            platform=Platform.FACEBOOK,
            text=text[:5000],
            author_id=from_data.get("id"),
            author_handle=from_data.get("name"),
            timestamp=timestamp,
            county=self._detect_county(text),
            engagement={
                "likes": data.get("like_count", 0),
                "retweets": 0,
                "replies": data.get("comment_count", 0),
                "shares": 0,
            },
            parent_post_id=f"fb_{parent_post_id}",
            raw_metadata={
                "is_comment": True,
            },
        )

    def _detect_county(self, text: str) -> Optional[str]:
        """Match county from post text."""
        text_lower = text.lower()
        for county in self.config.counties:
            if county.lower() in text_lower:
                return county
        return None

    async def disconnect(self):
        """Close client."""
        self._running = False
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("[facebook] Disconnected")
