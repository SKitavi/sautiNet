"""
SentiKenya Reddit Connector
==============================
Monitors Kenyan subreddits for sentiment data.

Reddit is valuable because:
- r/Kenya has long-form opinions (richer sentiment signals)
- Less bot activity than Twitter
- Anonymous → more honest opinions
- Good for economic sentiment (r/Kenya discusses cost of living heavily)

Uses Reddit's public JSON API (no OAuth needed for read-only public data)
or OAuth for higher rate limits (100 req/min vs 10 req/min).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, Dict, List

import httpx

from app.models.schemas import RawPost, Platform
from app.ingestion.base import BaseConnector, KenyaTrackingConfig

logger = logging.getLogger(__name__)


class RedditConnector(BaseConnector):
    """
    Reddit connector for Kenyan subreddit monitoring.

    Monitors:
    - r/Kenya (main)
    - r/Nairobi (city-specific)

    Two modes:
    - Public JSON API (no auth, 10 req/min)
    - OAuth API (with credentials, 100 req/min)
    """

    PUBLIC_BASE = "https://www.reddit.com"
    OAUTH_BASE = "https://oauth.reddit.com"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "SentiKenya/1.0 (Kenyan Sentiment Analysis)",
        config: KenyaTrackingConfig = None,
        poll_interval: int = 120,  # seconds between polls
    ):
        super().__init__(platform=Platform.REDDIT, config=config)
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.poll_interval = poll_interval

        self._client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._use_oauth = bool(client_id and client_secret)
        self._last_seen: Dict[str, str] = {}  # subreddit → last post fullname

    async def connect(self):
        """Set up HTTP client and optionally authenticate via OAuth."""
        headers = {"User-Agent": self.user_agent}

        if self._use_oauth:
            # OAuth2 app-only auth
            auth_client = httpx.AsyncClient()
            try:
                resp = await auth_client.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=(self.client_id, self.client_secret),
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": self.user_agent},
                )
                if resp.status_code == 200:
                    self._access_token = resp.json().get("access_token")
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    logger.info("[reddit] OAuth authenticated (100 req/min)")
                else:
                    logger.warning(f"[reddit] OAuth failed ({resp.status_code}), using public API")
                    self._use_oauth = False
            finally:
                await auth_client.aclose()

        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._connected = True
        logger.info(
            f"[reddit] Connected ({'OAuth' if self._use_oauth else 'public'} mode), "
            f"monitoring: {', '.join(self.config.subreddits)}"
        )

    async def _fetch_posts(self) -> AsyncGenerator[RawPost, None]:
        """Poll subreddits for new posts and comments."""
        while self._running:
            for subreddit in self.config.subreddits:
                try:
                    # Fetch new posts
                    async for post in self._fetch_subreddit(subreddit, "new"):
                        yield post

                    # Fetch hot posts (for engagement-weighted sentiment)
                    async for post in self._fetch_subreddit(subreddit, "hot"):
                        yield post

                    # Fetch comments on hot posts (richer sentiment)
                    async for comment in self._fetch_comments(subreddit):
                        yield comment

                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.warning(f"[reddit] Error fetching r/{subreddit}: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _fetch_subreddit(
        self,
        subreddit: str,
        sort: str = "new",
        limit: int = 25,
    ) -> AsyncGenerator[RawPost, None]:
        """Fetch posts from a subreddit."""
        base = self.OAUTH_BASE if self._use_oauth else self.PUBLIC_BASE
        url = f"{base}/r/{subreddit}/{sort}.json"

        params = {"limit": limit, "raw_json": 1}
        after = self._last_seen.get(f"{subreddit}_{sort}")
        if after and sort == "new":
            params["before"] = after  # Only get newer posts

        resp = await self._client.get(url, params=params)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            logger.warning(f"[reddit] Rate limited, waiting {retry_after}s")
            await asyncio.sleep(retry_after)
            return

        if resp.status_code != 200:
            logger.error(f"[reddit] r/{subreddit}/{sort} returned {resp.status_code}")
            return

        data = resp.json()
        posts = data.get("data", {}).get("children", [])

        for post_data in posts:
            if post_data.get("kind") != "t3":
                continue

            post = self._parse_post(post_data["data"], subreddit)
            if post:
                yield post

        # Track last seen for pagination
        if posts:
            self._last_seen[f"{subreddit}_{sort}"] = posts[0]["data"]["name"]

    async def _fetch_comments(
        self,
        subreddit: str,
        limit: int = 50,
    ) -> AsyncGenerator[RawPost, None]:
        """Fetch recent comments from a subreddit."""
        base = self.OAUTH_BASE if self._use_oauth else self.PUBLIC_BASE
        url = f"{base}/r/{subreddit}/comments.json"

        params = {"limit": limit, "raw_json": 1}

        resp = await self._client.get(url, params=params)
        if resp.status_code != 200:
            return

        data = resp.json()
        comments = data.get("data", {}).get("children", [])

        for comment_data in comments:
            if comment_data.get("kind") != "t1":
                continue

            comment = self._parse_comment(comment_data["data"], subreddit)
            if comment:
                yield comment

    def _parse_post(self, data: dict, subreddit: str) -> Optional[RawPost]:
        """Normalize a Reddit post into RawPost."""
        # Combine title + selftext for full content
        title = data.get("title", "").strip()
        body = data.get("selftext", "").strip()
        text = f"{title}. {body}" if body else title

        if not text or len(text) < 10:
            return None

        # Skip removed/deleted
        if data.get("removed_by_category") or text in ("[removed]", "[deleted]"):
            return None

        # Timestamp
        created_utc = data.get("created_utc", 0)
        timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)

        # Flair as topic hint
        flair = data.get("link_flair_text", "")

        return RawPost(
            post_id=f"rd_{data.get('id', '')}",
            platform=Platform.REDDIT,
            text=text[:5000],
            author_id=data.get("author"),
            author_handle=data.get("author"),
            timestamp=timestamp,
            location=None,
            county=self._detect_county_from_text(text),
            engagement={
                "likes": max(data.get("score", 0), 0),
                "retweets": 0,
                "replies": data.get("num_comments", 0),
                "shares": data.get("num_crossposts", 0),
            },
            hashtags=[],
            mentions=[],
            parent_post_id=None,
            raw_metadata={
                "subreddit": subreddit,
                "flair": flair,
                "upvote_ratio": data.get("upvote_ratio", 0),
                "is_self": data.get("is_self", True),
                "permalink": data.get("permalink", ""),
                "url": data.get("url", ""),
            },
        )

    def _parse_comment(self, data: dict, subreddit: str) -> Optional[RawPost]:
        """Normalize a Reddit comment into RawPost."""
        text = data.get("body", "").strip()

        if not text or len(text) < 15:
            return None
        if text in ("[removed]", "[deleted]"):
            return None

        created_utc = data.get("created_utc", 0)
        timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)

        return RawPost(
            post_id=f"rc_{data.get('id', '')}",
            platform=Platform.REDDIT,
            text=text[:5000],
            author_id=data.get("author"),
            author_handle=data.get("author"),
            timestamp=timestamp,
            county=self._detect_county_from_text(text),
            engagement={
                "likes": max(data.get("score", 0), 0),
                "retweets": 0,
                "replies": 0,
                "shares": 0,
            },
            parent_post_id=f"rd_{data.get('link_id', '').replace('t3_', '')}",
            raw_metadata={
                "subreddit": subreddit,
                "permalink": data.get("permalink", ""),
                "is_comment": True,
                "depth": data.get("depth", 0),
            },
        )

    def _detect_county_from_text(self, text: str) -> Optional[str]:
        """Try to detect county mention in text."""
        text_lower = text.lower()
        for county in self.config.counties:
            if county.lower() in text_lower:
                return county
        return None

    async def disconnect(self):
        """Close HTTP client."""
        self._running = False
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("[reddit] Disconnected")
