"""
SentiKenya — Decentralized Sentiment Analysis Platform
=======================================================
Main application entry point.

Starts the FastAPI server with:
- NLP Pipeline (language detection, sentiment, topics, NER)
- Kafka consumer (simulated or real)
- IPFS integration (simulated or real)
- WebSocket real-time broadcasting
- REST API for on-demand analysis

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from app.ml.pipeline import NLPPipeline
from app.services.kafka_service import KafkaService
from app.services.ipfs_service import IPFSService
from app.services.broadcast_service import ConnectionManager
from app.workers.nlp_worker import NLPWorker
from app.api.routes import api_router, ws_router, configure_routes

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sentikenya")

# ── Global instances ──
settings = get_settings()
pipeline: NLPPipeline = None
kafka: KafkaService = None
ipfs: IPFSService = None
broadcaster: ConnectionManager = None
worker: NLPWorker = None
worker_task: asyncio.Task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global pipeline, kafka, ipfs, broadcaster, worker, worker_task

    logger.info("=" * 60)
    logger.info(f"  SentiKenya v{settings.APP_VERSION}")
    logger.info(f"  Node: {settings.NODE_ID} ({settings.NODE_REGION})")
    logger.info("=" * 60)

    # ── Initialize NLP Pipeline ──
    logger.info("Initializing NLP Pipeline...")
    pipeline = NLPPipeline(
        sheng_lexicon_path=settings.SHENG_LEXICON_PATH,
        entities_path="./data/kenyan_entities.json",
        node_id=settings.NODE_ID,
    )
    await pipeline.initialize()
    logger.info("NLP Pipeline ready")

    # ── Initialize Services ──
    logger.info("Initializing services...")

    kafka = KafkaService(
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        simulate=True,  # Set False in production with real Kafka
    )
    await kafka.connect()

    ipfs = IPFSService(
        api_url=settings.IPFS_API_URL,
        gateway_url=settings.IPFS_GATEWAY_URL,
        enabled=settings.ENABLE_IPFS,
    )

    broadcaster = ConnectionManager()

    # ── Initialize Worker ──
    worker = NLPWorker(
        pipeline=pipeline,
        kafka=kafka,
        ipfs=ipfs,
        broadcaster=broadcaster,
    )

    # Configure API routes with dependencies
    configure_routes(pipeline, worker, broadcaster, ipfs)

    # ── Start background worker ──
    logger.info("Starting NLP background worker...")
    worker_task = asyncio.create_task(worker.start())

    logger.info("=" * 60)
    logger.info("  SentiKenya is LIVE")
    logger.info(f"  API:       http://{settings.HOST}:{settings.PORT}/api/v1")
    logger.info(f"  WebSocket: ws://{settings.HOST}:{settings.PORT}/ws/feed")
    logger.info(f"  Docs:      http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 60)

    yield

    # ── Shutdown ──
    logger.info("Shutting down SentiKenya...")
    if worker:
        await worker.stop()
    if worker_task:
        worker_task.cancel()
    if kafka:
        await kafka.disconnect()
    logger.info("SentiKenya shutdown complete")


# ── Create FastAPI App ──
app = FastAPI(
    title="SentiKenya",
    description=(
        "Decentralized Sentiment Analysis Platform for Kenyan Social Media.\n\n"
        "Analyzes English, Swahili, and Sheng posts from X/Twitter, Facebook, "
        "TikTok, Reddit, and Kenyans.co.ke in real-time.\n\n"
        "Features:\n"
        "- Multilingual sentiment analysis (EN/SW/SH)\n"
        "- Custom Sheng tokenizer and lexicon\n"
        "- County-level sentiment heatmaps\n"
        "- Topic and political sentiment classification\n"
        "- Kenyan NER (counties, parties, government bodies)\n"
        "- IPFS-backed immutable data storage\n"
        "- Real-time WebSocket streaming\n"
        "- Decentralized multi-node consensus"
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ──
app.include_router(api_router)
app.include_router(ws_router)


# ── Root endpoint ──
@app.get("/", tags=["root"])
async def root():
    return {
        "name": "SentiKenya",
        "version": settings.APP_VERSION,
        "description": "Decentralized Sentiment Analysis for Kenyan Social Media",
        "node": settings.NODE_ID,
        "status": "operational",
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "batch": "/api/v1/analyze/batch",
            "counties": "/api/v1/counties",
            "trending": "/api/v1/trending",
            "stats": "/api/v1/stats",
            "health": "/api/v1/health",
            "ws_feed": "/ws/feed",
            "docs": "/docs",
        },
    }


# ── Error handlers ──
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
