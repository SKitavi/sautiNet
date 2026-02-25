"""
SentiKenya NLP Pipeline
========================
Orchestrates the full NLP processing chain:
  Text → Language Detection → Sentiment Analysis → Topic Classification → NER

This is the main entry point for all text processing in the system.
"""

import time
import uuid
import hashlib
import json
import logging
from typing import List, Optional

from app.models.schemas import (
    RawPost, ProcessedPost, ProcessingStatus,
    LanguageDetection, SentimentResult, TopicResult, NERResult,
    SentimentAnalysisResponse,
)
from app.ml.language_detector import KenyanLanguageDetector
from app.ml.sentiment_engine import SentimentEngine, ModelConfig
from app.ml.topic_classifier import KenyanTopicClassifier
from app.ml.entity_extractor import KenyanEntityExtractor
from app.ml.sheng_tokenizer import ShengTokenizer

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Orchestrates the complete NLP processing pipeline.

    Processing order:
    1. Language Detection (determines downstream strategy)
    2. Sentiment Analysis (strategy depends on language)
    3. Topic Classification (keyword + context)
    4. Named Entity Recognition (rule-based + patterns)
    5. Post-processing and aggregation

    Thread-safe: Each pipeline instance can process concurrently.
    """

    def __init__(
        self,
        sheng_lexicon_path: str = "./data/sheng_lexicon.json",
        entities_path: str = "./data/kenyan_entities.json",
        node_id: str = "NBO-01",
        model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        finetuned_path: str = "./models/sentikenya-v1/best",
        custom_model_path: str = "./models/custom-bilstm-v1",
        use_gpu: bool = False,
    ):
        self.node_id = node_id
        self._initialized = False

        # Initialize components
        self.sheng_tokenizer = ShengTokenizer(lexicon_path=sheng_lexicon_path)
        self.language_detector = KenyanLanguageDetector(sheng_lexicon_path=sheng_lexicon_path)
        self.sentiment_engine = SentimentEngine(
            sheng_tokenizer=self.sheng_tokenizer,
            model_config=ModelConfig(
                model_name=model_name,
                finetuned_path=finetuned_path,
                custom_model_path=custom_model_path,
                use_gpu=use_gpu,
            ),
        )
        self.topic_classifier = KenyanTopicClassifier(entities_path=entities_path)
        self.entity_extractor = KenyanEntityExtractor(entities_path=entities_path)

        logger.info(f"NLP Pipeline initialized on node {node_id}")

    async def initialize(self):
        """
        Async initialization — loads ML models.
        Call this during application startup.
        """
        logger.info("Loading ML models...")
        await self.sentiment_engine.load_model()
        self._initialized = True
        logger.info("NLP Pipeline fully initialized and ready for processing")

    def process_text(self, text: str) -> SentimentAnalysisResponse:
        """
        Process a single text through the full pipeline.

        Returns a SentimentAnalysisResponse with all analysis results.
        Quick API for on-demand analysis.
        """
        start_time = time.time()

        # Step 1: Language Detection
        language = self.language_detector.detect(text)

        # Step 2: Sentiment Analysis
        sentiment = self.sentiment_engine.analyze(text, language)

        # Step 3: Topic Classification
        topics = self.topic_classifier.classify(text)

        # Step 4: Named Entity Recognition
        entities = self.entity_extractor.extract(text)

        total_ms = (time.time() - start_time) * 1000

        return SentimentAnalysisResponse(
            text=text,
            language=language,
            sentiment=sentiment,
            topics=topics,
            entities=entities,
            processing_time_ms=round(total_ms, 2),
        )

    def process_post(self, raw_post: RawPost) -> ProcessedPost:
        """
        Process a full RawPost through the pipeline.

        Returns a ProcessedPost with all annotations and metadata.
        Used by the Kafka consumer workers for stream processing.
        """
        start_time = time.time()

        try:
            # Step 1: Language Detection
            language = self.language_detector.detect(raw_post.text)

            # Step 2: Sentiment Analysis
            sentiment = self.sentiment_engine.analyze(raw_post.text, language)

            # Step 3: Topic Classification
            topics = self.topic_classifier.classify(
                raw_post.text,
                entities=raw_post.hashtags,
            )

            # Step 4: NER
            entities = self.entity_extractor.extract(raw_post.text)

            # If county not provided, try to infer from entities
            if not raw_post.county and entities.counties_mentioned:
                raw_post.county = entities.counties_mentioned[0]

            total_ms = (time.time() - start_time) * 1000

            return ProcessedPost(
                id=str(uuid.uuid4()),
                raw_post=raw_post,
                content_hash=raw_post.content_hash,
                language=language,
                sentiment=sentiment,
                topics=topics,
                entities=entities,
                node_id=self.node_id,
                processing_status=ProcessingStatus.COMPLETED,
                total_processing_time_ms=round(total_ms, 2),
            )

        except Exception as e:
            logger.error(f"Pipeline processing failed for post {raw_post.post_id}: {e}")
            total_ms = (time.time() - start_time) * 1000

            # Return failed result with partial data
            return ProcessedPost(
                id=str(uuid.uuid4()),
                raw_post=raw_post,
                content_hash=raw_post.content_hash,
                language=LanguageDetection(
                    detected_language="unk", confidence=0.0, all_scores={}
                ),
                sentiment=SentimentResult(
                    label="neutral", score=0.0, confidence=0.0,
                    model_used="failed"
                ),
                topics=TopicResult(primary_topic="unknown", topic_scores={}),
                entities=NERResult(),
                node_id=self.node_id,
                processing_status=ProcessingStatus.FAILED,
                total_processing_time_ms=round(total_ms, 2),
            )

    async def process_batch(self, posts: List[RawPost]) -> List[ProcessedPost]:
        """
        Process a batch of posts through the pipeline.

        For production: This would use async batching for transformer inference.
        """
        results = []
        for post in posts:
            result = self.process_post(post)
            results.append(result)
        return results

    def get_pipeline_stats(self) -> dict:
        """Return pipeline configuration and status."""
        return {
            "node_id": self.node_id,
            "initialized": self._initialized,
            "model_loaded": self.sentiment_engine._model_loaded,
            "active_model": self.sentiment_engine._active_model_name,
            "custom_model_loaded": self.sentiment_engine._custom_loaded,
            "sheng_vocab_size": len(self.sheng_tokenizer.vocabulary),
            "entity_counties": len(self.entity_extractor.counties),
            "topic_categories": len(self.topic_classifier.topic_keywords),
            "components": {
                "language_detector": "active",
                "sentiment_engine": self.sentiment_engine._active_model_name,
                "topic_classifier": "active",
                "entity_extractor": "active",
                "sheng_tokenizer": "active",
            }
        }
