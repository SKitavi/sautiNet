"""
SentiKenya Data Models
======================
Pydantic schemas for posts, sentiments, and system events.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import hashlib
import json


# ── Enums ──

class Language(str, Enum):
    ENGLISH = "en"
    SWAHILI = "sw"
    SHENG = "sh"
    UNKNOWN = "unk"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Platform(str, Enum):
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    REDDIT = "reddit"
    KENYANS = "kenyans_co_ke"
    TELEGRAM = "telegram"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Raw Post ──

class RawPost(BaseModel):
    """Raw social media post as ingested from platforms."""

    post_id: str = Field(..., description="Platform-specific post ID")
    platform: Platform
    text: str = Field(..., min_length=1, max_length=5000)
    author_id: Optional[str] = None
    author_handle: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    location: Optional[str] = None
    county: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    engagement: Dict[str, int] = Field(default_factory=lambda: {
        "likes": 0, "retweets": 0, "replies": 0, "shares": 0
    })
    hashtags: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    parent_post_id: Optional[str] = None  # for replies/threads
    media_urls: List[str] = Field(default_factory=list)
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash for deduplication and IPFS addressing."""
        content = json.dumps({
            "post_id": self.post_id,
            "platform": self.platform.value,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def engagement_score(self) -> float:
        """Weighted engagement score."""
        weights = {"likes": 1, "retweets": 3, "replies": 2, "shares": 4}
        return sum(self.engagement.get(k, 0) * w for k, w in weights.items())


# ── Language Detection ──

class LanguageDetection(BaseModel):
    """Result of language detection on a post."""

    detected_language: Language
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: Dict[str, float] = Field(default_factory=dict)
    contains_code_switching: bool = False  # Swahili-English mix
    sheng_indicators: List[str] = Field(default_factory=list)


# ── Sentiment Analysis ──

class SentimentResult(BaseModel):
    """Sentiment analysis result for a single post."""

    label: SentimentLabel
    score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    model_used: str = "afrisenti-xlmr"
    processing_time_ms: float = 0.0


# ── Topic Classification ──

class TopicResult(BaseModel):
    """Topic classification result."""

    primary_topic: str
    topic_scores: Dict[str, float] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    is_political: bool = False
    political_subtopic: Optional[str] = None  # e.g., "election", "policy", "governance"


# ── Named Entity Recognition ──

class Entity(BaseModel):
    """A single named entity extracted from text."""

    text: str
    label: str  # PERSON, ORG, COUNTY, POLITICAL_PARTY, GOV_BODY, etc.
    start: int
    end: int
    confidence: float = 0.0
    normalized: Optional[str] = None  # e.g., "William Ruto" -> "PRESIDENT_RUTO"


class NERResult(BaseModel):
    """Named Entity Recognition results."""

    entities: List[Entity] = Field(default_factory=list)
    political_figures: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    counties_mentioned: List[str] = Field(default_factory=list)


# ── Full Processed Post ──

class ProcessedPost(BaseModel):
    """Fully processed post with all NLP annotations."""

    # Identity
    id: str = Field(..., description="Internal processing ID")
    raw_post: RawPost
    content_hash: str

    # NLP Results
    language: LanguageDetection
    sentiment: SentimentResult
    topics: TopicResult
    entities: NERResult

    # Processing metadata
    node_id: str = "NBO-01"
    processing_status: ProcessingStatus = ProcessingStatus.COMPLETED
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    total_processing_time_ms: float = 0.0

    # Storage
    ipfs_cid: Optional[str] = None  # IPFS Content ID for raw data
    merkle_root: Optional[str] = None  # For cross-node verification

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ── Aggregated County Sentiment ──

class CountySentiment(BaseModel):
    """Aggregated sentiment for a county over a time window."""

    county: str
    time_window_start: datetime
    time_window_end: datetime
    overall_sentiment: float = Field(..., ge=-1.0, le=1.0)
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    total_posts: int = 0
    trending_topics: List[str] = Field(default_factory=list)
    dominant_language: Language = Language.ENGLISH
    language_distribution: Dict[str, float] = Field(default_factory=dict)
    top_entities: List[str] = Field(default_factory=list)
    engagement_weighted_sentiment: float = 0.0


# ── WebSocket Events ──

class WSEvent(BaseModel):
    """WebSocket broadcast event."""

    event_type: str  # "new_post", "sentiment_update", "alert", "county_update"
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    node_id: str = "NBO-01"


# ── Node Health ──

class NodeHealth(BaseModel):
    """Health status of a processing node."""

    node_id: str
    region: str
    status: str = "healthy"
    posts_processed: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    active_workers: int = 0
    queue_depth: int = 0
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool = False
    gpu_available: bool = False


# ── API Responses ──

class SentimentAnalysisRequest(BaseModel):
    """API request for on-demand sentiment analysis."""

    text: str = Field(..., min_length=1, max_length=5000)
    platform: Optional[Platform] = None
    county: Optional[str] = None
    detect_language: bool = True


class SentimentAnalysisResponse(BaseModel):
    """API response for sentiment analysis."""

    text: str
    language: LanguageDetection
    sentiment: SentimentResult
    topics: TopicResult
    entities: NERResult
    processing_time_ms: float


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request."""

    texts: List[str] = Field(..., min_length=1, max_length=100)
    detect_language: bool = True


class TrendingResponse(BaseModel):
    """Trending topics response."""

    topics: List[Dict[str, Any]]
    time_range: str
    total_posts_analyzed: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
