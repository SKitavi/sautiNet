"""
SentiKenya API Routes
======================
REST API and WebSocket endpoints for the sentiment analysis platform.
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel, Field

from app.models.schemas import (
    SentimentAnalysisRequest, SentimentAnalysisResponse,
    BatchAnalysisRequest, NodeHealth, CountySentiment,
)

logger = logging.getLogger(__name__)

# Create routers
api_router = APIRouter(prefix="/api/v1", tags=["sentiment"])
ws_router = APIRouter(tags=["websocket"])


# ── Dependency injection (set by main app) ──
_pipeline = None
_worker = None
_broadcaster = None
_ipfs = None
_ingestion = None


def configure_routes(pipeline, worker, broadcaster, ipfs, ingestion=None):
    """Configure route dependencies. Called during app startup."""
    global _pipeline, _worker, _broadcaster, _ipfs, _ingestion
    _pipeline = pipeline
    _worker = worker
    _broadcaster = broadcaster
    _ipfs = ipfs
    _ingestion = ingestion


# ══════════════════════════════════════════════════════
# REST API Endpoints
# ══════════════════════════════════════════════════════

# ── Sprint 5: Lightweight /predict endpoint ──

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class PredictResponse(BaseModel):
    label: str
    confidence: float
    language: str


@api_router.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    """
    Lightweight inference endpoint.

    POST /predict → {text: string} → {label, confidence, language}

    This is the sprint-spec compliant endpoint for edge inference.
    """
    import time as _time

    if not _pipeline:
        raise HTTPException(status_code=503, detail="NLP pipeline not initialized")

    t0 = _time.time()
    result = _pipeline.process_text(request.text)
    latency_ms = (_time.time() - t0) * 1000

    return PredictResponse(
        label=result.sentiment.label.value,
        confidence=round(result.sentiment.confidence, 4),
        language=result.language.detected_language.value,
    )


@api_router.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_text(request: SentimentAnalysisRequest):
    """
    Analyze sentiment of a single text.

    Runs the text through the full NLP pipeline:
    - Language detection (English/Swahili/Sheng)
    - Sentiment analysis
    - Topic classification
    - Named entity recognition

    Returns all analysis results with processing time.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="NLP pipeline not initialized")

    result = _pipeline.process_text(request.text)
    return result


@api_router.post("/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Batch analyze multiple texts.

    Max 100 texts per request.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="NLP pipeline not initialized")

    results = []
    for text in request.texts:
        result = _pipeline.process_text(text)
        results.append(result.model_dump(mode="json"))

    return {
        "results": results,
        "total": len(results),
        "batch_id": f"batch_{int(datetime.utcnow().timestamp())}",
    }


@api_router.get("/counties")
async def get_county_sentiments():
    """
    Get current sentiment data for all active counties.

    Returns aggregated sentiment scores, post counts,
    trending topics, and language distributions per county.
    """
    if not _worker:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    counties = _worker.aggregator.get_all_county_sentiments()
    return {
        "counties": [c.model_dump(mode="json") for c in counties],
        "total_counties": len(counties),
        "generated_at": datetime.utcnow().isoformat(),
    }


@api_router.get("/counties/{county_name}")
async def get_county_detail(county_name: str):
    """Get detailed sentiment for a specific county."""
    if not _worker:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    result = _worker.aggregator.get_county_sentiment(county_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"No data for county: {county_name}")
    return result.model_dump(mode="json")


@api_router.get("/trending")
async def get_trending_topics(limit: int = Query(10, ge=1, le=50)):
    """
    Get trending topics across all counties.

    Returns topics sorted by volume with sentiment scores
    and political classification.
    """
    if not _worker:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    topics = _worker.aggregator.get_trending_topics(limit=limit)
    return {
        "topics": topics,
        "total": len(topics),
        "generated_at": datetime.utcnow().isoformat(),
    }


@api_router.get("/stats")
async def get_system_stats():
    """
    Get comprehensive system statistics.

    Includes processing rates, model info, IPFS stats,
    WebSocket connections, and pipeline health.
    """
    stats = {}

    if _pipeline:
        stats["pipeline"] = _pipeline.get_pipeline_stats()

    if _worker:
        stats["worker"] = _worker.get_worker_stats()

    if _ipfs:
        stats["ipfs"] = _ipfs.get_stats()

    if _broadcaster:
        stats["websocket"] = _broadcaster.get_stats()

    stats["generated_at"] = datetime.utcnow().isoformat()
    return stats


@api_router.get("/health")
async def health_check():
    """Node health check endpoint for monitoring and inter-node communication."""
    pipeline_stats = _pipeline.get_pipeline_stats() if _pipeline else {}

    return NodeHealth(
        node_id=pipeline_stats.get("node_id", "unknown"),
        region="nairobi",
        status="healthy" if _pipeline else "degraded",
        posts_processed=_worker._processed_count if _worker else 0,
        active_workers=1 if _worker and _worker._running else 0,
        model_loaded=pipeline_stats.get("model_loaded", False),
    ).model_dump(mode="json")


@api_router.get("/ingestion/status")
async def ingestion_status():
    """
    Real-time status of all social media connectors.

    Shows which platforms are active, posts ingested, errors, etc.
    """
    if not _ingestion:
        return {
            "running": False,
            "message": "Ingestion not configured. Add API keys to enable.",
            "connectors": {},
        }
    return _ingestion.get_status()


@api_router.get("/language/detect")
async def detect_language(text: str = Query(..., min_length=1)):
    """Detect language of input text (English/Swahili/Sheng)."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result = _pipeline.language_detector.detect(text)
    return result.model_dump()


@api_router.get("/ipfs/{cid}")
async def get_ipfs_content(cid: str):
    """Retrieve raw post data from IPFS by content ID."""
    if not _ipfs:
        raise HTTPException(status_code=503, detail="IPFS not configured")

    content = await _ipfs.get_content(cid)
    if not content:
        raise HTTPException(status_code=404, detail=f"Content not found: {cid}")
    return content


# ══════════════════════════════════════════════════════
# WebSocket Endpoints
# ══════════════════════════════════════════════════════

@ws_router.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sentiment feed.

    Streams processed posts as they're analyzed.
    Clients receive: language, sentiment, topic, entities, and metadata.
    """
    if not _broadcaster:
        await websocket.close(code=1011, reason="Broadcaster not initialized")
        return

    await _broadcaster.connect(websocket, channels=["feed", "alerts"])

    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            # Client can send filter preferences
            logger.debug(f"WS client message: {data}")
    except WebSocketDisconnect:
        _broadcaster.disconnect(websocket)


@ws_router.websocket("/ws/counties")
async def websocket_counties(websocket: WebSocket):
    """WebSocket endpoint for county-level sentiment updates."""
    if not _broadcaster:
        await websocket.close(code=1011)
        return

    await _broadcaster.connect(websocket, channels=["counties", "alerts"])

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _broadcaster.disconnect(websocket)


@ws_router.websocket("/ws/all")
async def websocket_all(websocket: WebSocket):
    """WebSocket endpoint subscribing to all channels."""
    if not _broadcaster:
        await websocket.close(code=1011)
        return

    await _broadcaster.connect(websocket, channels=["all"])

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _broadcaster.disconnect(websocket)
