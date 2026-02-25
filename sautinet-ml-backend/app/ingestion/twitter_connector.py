"""
SentiKenya Twitter/X Connector
================================
Connects to Twitter API v2 for real-time Kenyan tweet ingestion.

Two modes:
1. Filtered Stream (real-time) — persistent connection, tweets pushed as they happen
2. Recent Search (polling) — periodic search for keywords, backfills gaps

Requires: TWITTER_BEARER_TOKEN in env/settings.

Twitter API v2 rate limits:
- Filtered stream: 50 rules, 1 connection (Basic tier)
- Recent search: 450 requests/15min (Basic), 300 tweets per request
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, List, Dict

import httpx

from app.models.schemas import RawPost, Platform
from app.ingestion.base import BaseConnector, KenyaTrackingConfig

logger = logging.getLogger(__name__)


class TwitterConnector(BaseConnector):
    """
    Twitter/X API v2 connector.

    Supports:
    - Filtered stream with Kenya-specific rules
    - Recent search polling for gap-filling
    - Geo-fenced monitoring (Kenya bounding box)
    - Automatic rule management
    """

    BASE_URL = "https://api.twitter.com/2"

    def __init__(
        self,
        bearer_token: str,
        config: KenyaTrackingConfig = None,
        mode: str = "stream",  # "stream" or "search"
        search_interval: int = 60,  # seconds between search polls
    ):
        super().__init__(platform=Platform.TWITTER, config=config)
        self.bearer_token = bearer_token
        self.mode = mode
        self.search_interval = search_interval

        self._client: Optional[httpx.AsyncClient] = None
        self._stream_response = None
        self._last_search_id: Optional[str] = None

        # Tweet fields to request
        self.tweet_fields = [
            "created_at", "author_id", "geo", "lang",
            "public_metrics", "entities", "context_annotations",
        ]
        self.user_fields = ["username", "location", "verified"]
        self.place_fields = ["full_name", "country", "geo"]
        self.expansions = ["author_id", "geo.place_id"]

    async def connect(self):
        """Initialize HTTP client and set up stream rules."""
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "SentiKenya/1.0",
            },
            timeout=httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0),
        )

        # Verify credentials
        try:
            resp = await self._client.get(f"{self.BASE_URL}/tweets/counts/recent",
                                          params={"query": "Kenya"})
            if resp.status_code == 401:
                raise ConnectionError("Invalid Twitter bearer token")
            elif resp.status_code == 403:
                raise ConnectionError("Twitter API access level insufficient — need Basic or higher")
            logger.info("[twitter] Credentials verified")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot reach Twitter API: {e}")

        # Set up filtered stream rules if in stream mode
        if self.mode == "stream":
            await self._setup_stream_rules()

        self._connected = True
        logger.info(f"[twitter] Connected in {self.mode} mode")

    async def _setup_stream_rules(self):
        """
        Configure filtered stream rules for Kenya monitoring.

        Twitter allows max 25 rules (Basic) or 1000 (Enterprise).
        We create focused rules to maximize coverage within limits.
        """
        # Get existing rules
        resp = await self._client.get(f"{self.BASE_URL}/tweets/search/stream/rules")
        existing = resp.json().get("data", [])

        # Delete all existing rules first
        if existing:
            ids = [r["id"] for r in existing]
            await self._client.post(
                f"{self.BASE_URL}/tweets/search/stream/rules",
                json={"delete": {"ids": ids}}
            )
            logger.info(f"[twitter] Deleted {len(ids)} existing stream rules")

        # Build Kenya-focused rules
        rules = [
            # Rule 1: Kenya geo-fence (any tweet from Kenya)
            {
                "value": f"place_country:KE -is:retweet",
                "tag": "kenya-geo",
            },
            # Rule 2: Kenyan political keywords (English)
                {
                "value": (
                    "(Kenya government OR Kenya politics OR Kenya economy "
                    "OR Kenya corruption OR Kenya tax OR cost of living Kenya) "
                    "lang:en -is:retweet"
                ),
                "tag": "kenya-politics-en",
            },
            # Rule 3: Kenyan keywords (Swahili)
            {
                "value": (
                    "(serikali Kenya OR uchumi Kenya OR rushwa Kenya "
                    "OR wananchi OR maendeleo Kenya) "
                    "-is:retweet"
                ),
                "tag": "kenya-politics-sw",
            },
            # Rule 4: Kenyan brands/institutions
            {
                "value": (
                    "(Safaricom OR \"M-Pesa\" OR \"Kenya Power\" OR NHIF OR SHA "
                    "OR \"housing levy\") "
                    "-is:retweet"
                ),
                "tag": "kenya-brands",
            },
            # Rule 5: KOT and Kenyan hashtags
            {
                "value": (
                    "(#KOT OR #KenyaNews OR #NairobiNews OR #GenZKenya "
                    "OR #KenyaTax OR #CostOfLiving) "
                    "-is:retweet"
                ),
                "tag": "kenya-hashtags",
            },
            # Rule 6: Nairobi specific
            {
                "value": (
                    "(Nairobi traffic OR Nairobi county OR Nairobi news) "
                    "lang:en -is:retweet"
                ),
                "tag": "nairobi",
            },
            # Rule 7: Youth and tech
            {
                "value": (
                    "(\"Silicon Savannah\" OR Kenya tech OR Kenya startup "
                    "OR Kenya youth unemployment) "
                    "-is:retweet"
                ),
                "tag": "kenya-tech-youth",
            },
        ]

        # Add rules
        resp = await self._client.post(
            f"{self.BASE_URL}/tweets/search/stream/rules",
            json={"add": rules}
        )
        result = resp.json()

        if "errors" in result:
            for err in result["errors"]:
                logger.warning(f"[twitter] Rule error: {err.get('title')}: {err.get('detail')}")

        created = result.get("meta", {}).get("summary", {}).get("created", 0)
        logger.info(f"[twitter] Created {created} stream rules")

    async def _fetch_posts(self) -> AsyncGenerator[RawPost, None]:
        """Fetch posts via stream or search depending on mode."""
        if self.mode == "stream":
            async for post in self._filtered_stream():
                yield post
        else:
            async for post in self._search_poll():
                yield post

    async def _filtered_stream(self) -> AsyncGenerator[RawPost, None]:
        """
        Connect to Twitter filtered stream.

        Maintains persistent HTTP connection. Twitter pushes matching
        tweets as newline-delimited JSON.
        """
        params = {
            "tweet.fields": ",".join(self.tweet_fields),
            "user.fields": ",".join(self.user_fields),
            "place.fields": ",".join(self.place_fields),
            "expansions": ",".join(self.expansions),
        }

        logger.info("[twitter] Opening filtered stream...")

        async with self._client.stream(
            "GET",
            f"{self.BASE_URL}/tweets/search/stream",
            params=params,
            timeout=None,  # Keep-alive connection
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise ConnectionError(
                    f"Stream returned {response.status_code}: {body.decode()}"
                )

            logger.info("[twitter] Stream connected, receiving tweets...")
            self._stream_response = response

            async for line in response.aiter_lines():
                if not line.strip():
                    continue  # Heartbeat

                try:
                    data = json.loads(line)
                    if "data" in data:
                        post = self._parse_tweet(data)
                        if post:
                            yield post
                except json.JSONDecodeError:
                    logger.debug(f"[twitter] Non-JSON line: {line[:100]}")
                except Exception as e:
                    logger.warning(f"[twitter] Parse error: {e}")

    async def _search_poll(self) -> AsyncGenerator[RawPost, None]:
        """
        Poll recent search endpoint periodically.

        Good for:
        - Lower-tier API access (no stream available)
        - Gap-filling when stream disconnects
        - Historical backfill
        """
        while self._running:
            try:
                # Build search query
                query = self._build_search_query()

                params = {
                    "query": query,
                    "max_results": 100,
                    "tweet.fields": ",".join(self.tweet_fields),
                    "user.fields": ",".join(self.user_fields),
                    "expansions": ",".join(self.expansions),
                    "sort_order": "recency",
                }

                if self._last_search_id:
                    params["since_id"] = self._last_search_id

                resp = await self._client.get(
                    f"{self.BASE_URL}/tweets/search/recent",
                    params=params,
                )

                if resp.status_code == 429:
                    # Rate limited — wait for reset
                    reset = int(resp.headers.get("x-rate-limit-reset", 0))
                    wait = max(reset - int(datetime.now(timezone.utc).timestamp()), 15)
                    logger.warning(f"[twitter] Rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code != 200:
                    logger.error(f"[twitter] Search error {resp.status_code}: {resp.text[:200]}")
                    await asyncio.sleep(self.search_interval)
                    continue

                data = resp.json()
                tweets = data.get("data", [])
                includes = data.get("includes", {})

                # Build user lookup
                users = {u["id"]: u for u in includes.get("users", [])}

                for tweet_data in tweets:
                    post = self._parse_tweet(
                        {"data": tweet_data, "includes": includes},
                        users=users,
                    )
                    if post:
                        yield post

                # Track pagination
                if tweets:
                    self._last_search_id = tweets[0]["id"]
                    logger.info(f"[twitter] Search returned {len(tweets)} tweets")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[twitter] Search error: {e}")

            await asyncio.sleep(self.search_interval)

    def _build_search_query(self) -> str:
        """Build Twitter search query for Kenya content."""
        # Combine key terms with OR, filter retweets
        terms = [
            "Kenya government", "Kenya economy", "serikali Kenya",
            "#KOT", "#KenyaNews", "Safaricom", "M-Pesa",
            "Nairobi", "cost of living Kenya", "Kenya corruption",
        ]
        query = f"({' OR '.join(terms)}) -is:retweet lang:en OR lang:sw"

        # Twitter query max 512 chars for Basic tier
        if len(query) > 512:
            query = query[:510]

        return query

    def _parse_tweet(
        self,
        data: dict,
        users: Dict = None,
    ) -> Optional[RawPost]:
        """
        Normalize a Twitter API v2 response into RawPost.

        Handles:
        - Tweet fields → text, timestamp, IDs
        - User expansion → author handle, location
        - Place expansion → geo coordinates, county detection
        - Entities → hashtags, mentions
        - Metrics → engagement counts
        """
        tweet = data.get("data", {})
        includes = data.get("includes", {})

        text = tweet.get("text", "").strip()
        if not text or len(text) < 10:
            return None

        # Author info
        author_id = tweet.get("author_id")
        author_handle = None
        author_location = None

        if users and author_id in users:
            user = users[author_id]
            author_handle = user.get("username")
            author_location = user.get("location")
        elif includes.get("users"):
            for user in includes["users"]:
                if user["id"] == author_id:
                    author_handle = user.get("username")
                    author_location = user.get("location")
                    break

        # Geo
        lat, lon = None, None
        place_name = None
        if includes.get("places"):
            place = includes["places"][0]
            place_name = place.get("full_name")
            bbox = place.get("geo", {}).get("bbox", [])
            if len(bbox) >= 4:
                lat = (bbox[1] + bbox[3]) / 2
                lon = (bbox[0] + bbox[2]) / 2

        # County detection from location string
        county = self._detect_county(place_name or author_location or "")

        # Entities
        entities = tweet.get("entities", {})
        hashtags = [h["tag"] for h in entities.get("hashtags", [])]
        mentions = [m["username"] for m in entities.get("mentions", [])]

        # Engagement
        metrics = tweet.get("public_metrics", {})

        # Timestamp
        created_at = tweet.get("created_at")
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()

        return RawPost(
            post_id=f"tw_{tweet.get('id', '')}",
            platform=Platform.TWITTER,
            text=text,
            author_id=author_id,
            author_handle=author_handle,
            timestamp=timestamp,
            location=place_name or author_location,
            county=county,
            latitude=lat,
            longitude=lon,
            engagement={
                "likes": metrics.get("like_count", 0),
                "retweets": metrics.get("retweet_count", 0),
                "replies": metrics.get("reply_count", 0),
                "shares": metrics.get("quote_count", 0),
            },
            hashtags=hashtags,
            mentions=mentions,
            raw_metadata={
                "lang": tweet.get("lang"),
                "matching_rules": [
                    r.get("tag") for r in data.get("matching_rules", [])
                ],
                "context_annotations": tweet.get("context_annotations", []),
            },
        )

    def _detect_county(self, location_str: str) -> Optional[str]:
        """Try to match location string to a Kenyan county."""
        if not location_str:
            return None
        loc_lower = location_str.lower()
        for county in self.config.counties:
            if county.lower() in loc_lower:
                return county
        return None

    async def disconnect(self):
        """Close stream and HTTP client."""
        self._running = False
        self._connected = False
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("[twitter] Disconnected")
