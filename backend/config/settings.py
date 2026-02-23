"""
SentiKenya Configuration
========================
Central configuration for the decentralized sentiment analysis platform.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ──
    APP_NAME: str = "SentiKenya"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Database ──
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sentikenya"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Kafka ──
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_RAW_POSTS: str = "sentikenya.raw.posts"
    KAFKA_TOPIC_PROCESSED: str = "sentikenya.processed.sentiments"
    KAFKA_CONSUMER_GROUP: str = "sentikenya-nlp-workers"

    # ── IPFS ──
    IPFS_API_URL: str = "http://localhost:5001"
    IPFS_GATEWAY_URL: str = "https://ipfs.io/ipfs"
    ENABLE_IPFS: bool = False

    # ── ML Models ──
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    FINETUNED_MODEL_PATH: str = "./models/sentikenya-v1/best"
    SHENG_LEXICON_PATH: str = "./data/sheng_lexicon.json"
    MODEL_CACHE_DIR: str = "./model_cache"
    USE_GPU: bool = False
    BATCH_SIZE: int = 32
    MAX_SEQUENCE_LENGTH: int = 256

    # ── Social Platform APIs ──
    TWITTER_BEARER_TOKEN: Optional[str] = None
    TWITTER_API_KEY: Optional[str] = None
    TWITTER_API_SECRET: Optional[str] = None
    TWITTER_MODE: str = "search"  # "stream" or "search"
    FACEBOOK_ACCESS_TOKEN: Optional[str] = None
    FACEBOOK_PAGES: Optional[str] = None  # comma-separated page IDs
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    TIKTOK_API_KEY: Optional[str] = None

    # ── Ingestion ──
    ENABLE_INGESTION: bool = True  # auto-start connectors on boot

    # ── Processing ──
    NODE_ID: str = "NBO-01"
    NODE_REGION: str = "nairobi"
    PROCESSING_WORKERS: int = 4
    WEBSOCKET_BROADCAST_INTERVAL: float = 1.0

    # ── Rate Limiting ──
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
