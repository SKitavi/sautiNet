# SentiKenya — Kafka Setup Guide

## Overview

Apache Kafka serves as the message streaming backbone for SentiKenya's real-time social media ingestion pipeline. Posts flow through three topics:

```
Social Media APIs
       │
       ▼
  ┌──────────┐     ┌─────────────────────┐     ┌────────────────────┐
  │ raw-posts │ ──► │ preprocessed-posts   │ ──► │ sentiment-results  │
  └──────────┘     └─────────────────────┘     └────────────────────┘
    (ingested)        (cleaned / tokenised)       (label + confidence)
```

## Prerequisites

- Docker & Docker Compose v2+
- Python 3.11+
- `aiokafka` (included in `requirements.txt`)

## Quick Start

### 1. Start the Kafka Cluster

```bash
docker compose up -d zookeeper kafka kafka-ui
```

Verify Kafka is healthy:

```bash
docker compose ps            # kafka container should show "healthy"
docker exec sentikenya-kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

Kafka UI is available at **http://localhost:8090**.

### 2. Create Topics

```bash
python -m kafka.init_topics
# or with a remote broker:
python -m kafka.init_topics --broker kafka:29092
```

This creates three topics with 3 partitions each:

| Topic                 | Purpose                                | Partitions |
|-----------------------|----------------------------------------|------------|
| `raw-posts`           | Raw social media posts from ingestion  | 3          |
| `preprocessed-posts`  | Cleaned, tokenised text                | 3          |
| `sentiment-results`   | Final `{label, confidence, language}`  | 3          |

### 3. Run the Producer

```bash
# Produce 50 simulated posts (default)
python -m kafka.producer

# Produce continuously
python -m kafka.producer --continuous --interval 1.0

# Custom broker
python -m kafka.producer --broker kafka:29092 --count 100
```

### 4. Run the Consumer

```bash
# Consume and log
python -m kafka.consumer

# Consume, clean, and forward to preprocessed-posts
python -m kafka.consumer --forward

# Consume only 20 messages
python -m kafka.consumer --count 20
```

## End-to-End Test

Open **three terminals**:

```bash
# Terminal 1 — produce
python -m kafka.producer --count 30

# Terminal 2 — consume raw-posts → forward to preprocessed-posts
python -m kafka.consumer --forward --count 30

# Terminal 3 — verify messages arrived in preprocessed-posts
docker exec sentikenya-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic preprocessed-posts \
  --from-beginning --max-messages 5
```

Expected output: JSON messages with `post_id`, `cleaned_text`, `county`, etc.

## Docker Compose Services

| Service               | Port  | Description                       |
|-----------------------|-------|-----------------------------------|
| `zookeeper`           | 2181  | Kafka coordination                |
| `kafka`               | 9092  | Kafka broker (external listener)  |
| `kafka-ui`            | 8090  | Web UI for topic inspection       |

## Configuration

Environment variables (set in `docker-compose.yml`):

```
KAFKA_BOOTSTRAP_SERVERS=kafka:29092     # internal Docker address
KAFKA_AUTO_CREATE_TOPICS_ENABLE=true    # fallback auto-creation
```

## Troubleshooting

| Problem                     | Fix                                                      |
|-----------------------------|----------------------------------------------------------|
| `NoBrokersAvailable`        | Ensure Kafka container is healthy: `docker compose ps`   |
| Consumer stuck              | Reset offsets: add `--group new-group-name`               |
| Topic not visible in UI     | Run `python -m kafka.init_topics` to create explicitly   |
| Port conflict on 9092       | Change external port in `docker-compose.yml`             |
