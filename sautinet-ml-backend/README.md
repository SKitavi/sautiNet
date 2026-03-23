# 🇰🇪 SautiNet

**Decentralized Sentiment Analysis for Kenyan Social Media**

A distributed NLP platform that analyzes political sentiment and public opinion across Kenyan social media in real-time, supporting **English**, **Swahili**, and **Sheng**.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                       │
│  Twitter/X API │ Facebook Graph │ TikTok │ Web Scrapers      │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                      APACHE KAFKA                            │
│  sentikenya.raw.posts │ Partitioned by County │ 3 Consumers  │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   NLP PROCESSING ENGINE                       │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  Language    │  │  Sentiment   │  │  Topic            │    │
│  │  Detection   │──▶  Engine      │  │  Classifier       │    │
│  │  (EN/SW/SH) │  │  (Hybrid)    │  │  (14 categories)  │    │
│  └─────────────┘  └──────────────┘  └──────────────────┘    │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │  Sheng      │  │  Entity      │  │  Sentiment        │    │
│  │  Tokenizer  │  │  Extractor   │  │  Aggregator       │    │
│  │  (Custom)   │  │  (Kenyan NER)│  │  (County-level)   │    │
│  └─────────────┘  └──────────────┘  └──────────────────┘    │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  DECENTRALIZED STORAGE                        │
│  IPFS (Raw Data) │ TimescaleDB (Time-series) │ Redis (Cache) │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    API GATEWAY (FastAPI)                      │
│  REST Endpoints │ WebSocket Streams │ GraphQL (planned)      │
└──────────────────────────────────────────────────────────────┘
```

## Project Structure

```
sentikenya/
├── app/
│   ├── api/
│   │   └── routes.py          # REST API + WebSocket endpoints
│   ├── ml/
│   │   ├── language_detector.py   # EN/SW/SH detection
│   │   ├── sheng_tokenizer.py     # Custom Sheng tokenizer
│   │   ├── sentiment_engine.py    # Multilingual sentiment
│   │   ├── topic_classifier.py    # Kenyan topic classification
│   │   ├── entity_extractor.py    # Kenyan NER
│   │   └── pipeline.py           # Full NLP orchestrator
│   ├── services/
│   │   ├── kafka_service.py       # Kafka producer/consumer
│   │   ├── ipfs_service.py        # IPFS + Merkle trees
│   │   └── broadcast_service.py   # WebSocket broadcasting
│   ├── workers/
│   │   └── nlp_worker.py         # Background processing worker
│   ├── models/
│   │   └── schemas.py            # Pydantic data models
│   └── main.py                   # FastAPI application
├── config/
│   └── settings.py               # Environment configuration
├── data/
│   ├── sheng_lexicon.json        # 62-word Sheng vocabulary
│   └── kenyan_entities.json      # 47 counties, parties, orgs
├── tests/
│   └── test_pipeline.py          # Full test suite
├── docker-compose.yml            # 3-node cluster + infra
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### Development (Simulated Mode)
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_pipeline.py

# Start server (simulated Kafka + IPFS)
uvicorn app.main:app --reload --port 8000
```

### Production (Docker Cluster)
```bash
# Start 3-node cluster with Kafka, TimescaleDB, Redis, IPFS
docker-compose up -d

# Nodes:
# - NBO-01 (Nairobi):  http://localhost:8000
# - MSA-01 (Mombasa):  http://localhost:8001
# - KSM-01 (Kisumu):   http://localhost:8002
# - Kafka UI:          http://localhost:8090
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze` | Analyze single text |
| POST | `/api/v1/analyze/batch` | Batch analyze (max 100) |
| GET | `/api/v1/counties` | All county sentiments |
| GET | `/api/v1/counties/{name}` | Single county detail |
| GET | `/api/v1/trending` | Trending topics |
| GET | `/api/v1/stats` | System statistics |
| GET | `/api/v1/health` | Node health check |
| GET | `/api/v1/language/detect?text=...` | Language detection |
| WS | `/ws/feed` | Real-time sentiment feed |
| WS | `/ws/counties` | County updates stream |
| WS | `/ws/all` | All channels |

### Example: Analyze Text
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Manze hii serikali iko rada wasee mambo ni poa"}'
```

Response:
```json
{
  "text": "Manze hii serikali iko rada wasee mambo ni poa",
  "language": {
    "detected_language": "sh",
    "confidence": 0.99,
    "contains_code_switching": true,
    "sheng_indicators": ["manze", "rada", "wasee", "poa"]
  },
  "sentiment": {
    "label": "positive",
    "score": 0.35,
    "confidence": 0.72,
    "model_used": "sheng-hybrid"
  },
  "topics": {
    "primary_topic": "governance",
    "is_political": true
  },
  "processing_time_ms": 0.6
}
```

## NLP Pipeline

### Language Detection
- **English**: Function word patterns (the, is, are, with)
- **Swahili**: Morphological analysis (prefixes: wa-, ni-, ana-; suffixes: -isha, -ika)
- **Sheng**: Custom lexicon matching + code-switching detection

### Sentiment Analysis (Hybrid Engine)
| Language | Strategy | Confidence |
|----------|----------|------------|
| English | Transformer (AfriSenti XLM-R) | ~88% |
| Swahili | Transformer + phrase patterns | ~82% |
| Sheng | Lexicon rules + transformer blend | ~72% |
| Mixed | Ensemble (weighted by lang scores) | ~70% |

### Sheng Tokenizer
Custom-built because no standard NLP library supports Sheng:
- SMS normalization (`bt` → `but`, `2day` → `today`)
- Phonetic spelling resolution (`manzi` → `manze`)
- Compound expression detection (`ni noma` → `ni_noma`)
- Sentiment modifier tagging (intensifiers, negations)

### Entity Recognition
- **47 Kenyan counties** + aliases (e.g., `nai` → `Nairobi`)
- **9 political parties** + abbreviations
- **Government bodies**: Parliament, KRA, IEBC, EACC, etc.
- **Organizations**: Safaricom, M-Pesa, universities, media
- **Monetary values**: KES, billions, Ksh patterns

## Decentralization

- **IPFS**: Raw posts pinned for tamper-proof audit trail
- **Merkle Trees**: Batch verification across nodes
- **3 Processing Nodes**: Nairobi, Mombasa, Kisumu
- **Consensus**: Multi-node sentiment scoring verification

## Tech Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn
- **ML/NLP**: Transformers (AfriSenti), Custom Sheng engine
- **Streaming**: Apache Kafka, WebSockets
- **Storage**: TimescaleDB, IPFS, Redis
- **Deployment**: Docker, docker-compose
