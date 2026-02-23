"""
SentiKenya Sentiment Engine
=============================
Multilingual sentiment analysis for Kenyan social media.

Architecture:
- Primary: Transformer-based model (AfriSenti XLM-R) for EN/SW
- Secondary: Custom rule-based system for Sheng (no training data exists)
- Ensemble: Weighted combination based on language detection confidence
- Post-processing: Sheng modifiers (intensifiers, negations) adjust scores

The Sheng challenge:
- No labeled sentiment dataset exists for Sheng
- Code-switching makes standard models unreliable
- We use a hybrid approach: transformer base + lexicon adjustment
"""

import logging
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from app.models.schemas import (
    Language, SentimentLabel, SentimentResult, LanguageDetection,
)
from app.ml.sheng_tokenizer import ShengTokenizer, ShengToken

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for sentiment model loading."""
    model_name: str
    use_gpu: bool = False
    max_length: int = 256
    batch_size: int = 32


class SentimentEngine:
    """
    Multi-strategy sentiment analyzer.

    Strategy selection based on detected language:
    - English: Transformer model (XLM-R fine-tuned on African tweets)
    - Swahili: Same transformer + Swahili-specific post-processing
    - Sheng: Hybrid (transformer base + custom lexicon + rule engine)
    - Mixed/Code-switch: Ensemble of all three strategies
    """

    # ── Sentiment lexicon for rule-based Sheng scoring ──
    POSITIVE_PATTERNS = {
        "poa": 0.5, "fiti": 0.4, "sawa": 0.2, "mnoma": 0.6,
        "rada": 0.3, "legit": 0.4, "mresh": 0.5, "cheza": 0.3,
        "morio": 0.3, "kupendeza": 0.6, "nzuri": 0.5,
        "good": 0.4, "great": 0.6, "amazing": 0.7, "love": 0.6,
        "best": 0.6, "happy": 0.5, "proud": 0.5, "progress": 0.4,
        "improve": 0.3, "success": 0.5, "win": 0.5, "hope": 0.3,
        "better": 0.3, "excellent": 0.7, "beautiful": 0.5,
        "nzuri": 0.5, "bora": 0.5, "furaha": 0.5, "amani": 0.4,
        "maendeleo": 0.4, "tumaini": 0.4, "upendo": 0.5,
    }

    NEGATIVE_PATTERNS = {
        "noma": -0.4, "ngori": -0.4, "mbaya": -0.6, "tight": -0.4,
        "wack": -0.5, "imeisha": -0.3, "staki": -0.3, "hatutaki": -0.4,
        "bad": -0.5, "worst": -0.7, "terrible": -0.7, "hate": -0.6,
        "corrupt": -0.6, "steal": -0.5, "suffering": -0.6,
        "expensive": -0.4, "poor": -0.4, "fail": -0.5, "crisis": -0.5,
        "problem": -0.3, "wrong": -0.4, "sad": -0.5, "angry": -0.5,
        "unfair": -0.5, "shame": -0.5, "disappointed": -0.5,
        "mbaya": -0.6, "hatari": -0.5, "rushwa": -0.6, "umaskini": -0.5,
        "mauaji": -0.7, "uhalifu": -0.5, "ukosefu": -0.4,
        "hasira": -0.5, "huzuni": -0.5, "dhuluma": -0.6,
    }

    # ── Swahili-specific sentiment markers ──
    SWAHILI_SENTIMENT = {
        # Positive governance
        "inafanya kazi": 0.4, "kazi nzuri": 0.5, "tunashukuru": 0.4,
        "maendeleo": 0.4, "mabadiliko": 0.3, "haki": 0.3,
        # Negative governance
        "inafanya vibaya": -0.5, "wanaibia": -0.6, "rushwa": -0.6,
        "bei imepanda": -0.4, "maisha magumu": -0.5, "hakuna kazi": -0.5,
        "tunahitaji mabadiliko": -0.3, "hali mbaya": -0.5,
        "serikali imeshindwa": -0.5, "tunateseka": -0.6,
    }

    def __init__(
        self,
        sheng_tokenizer: Optional[ShengTokenizer] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        self.sheng_tokenizer = sheng_tokenizer or ShengTokenizer()
        self.model_config = model_config or ModelConfig(
            model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
        )
        self._transformer_model = None
        self._tokenizer = None
        self._model_loaded = False

    async def load_model(self):
        """
        Load the transformer model for sentiment analysis.

        In production, this loads the fine-tuned AfriSenti model.
        Falls back to rule-based if model loading fails.
        """
        try:
            # Attempt to load transformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            logger.info(f"Loading sentiment model: {self.model_config.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name,
                cache_dir="./model_cache"
            )
            self._transformer_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name,
                cache_dir="./model_cache"
            )
            if self.model_config.use_gpu and torch.cuda.is_available():
                self._transformer_model = self._transformer_model.cuda()

            self._transformer_model.eval()
            self._model_loaded = True
            logger.info("Sentiment model loaded successfully")

        except ImportError:
            logger.warning("Transformers not installed, using rule-based engine only")
            self._model_loaded = False
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}. Using rule-based fallback.")
            self._model_loaded = False

    def analyze(
        self,
        text: str,
        language: LanguageDetection,
    ) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Routes to appropriate strategy based on detected language.
        """
        start_time = time.time()

        if language.detected_language == Language.SHENG:
            result = self._analyze_sheng(text)
        elif language.detected_language == Language.SWAHILI:
            result = self._analyze_swahili(text)
        elif language.detected_language == Language.ENGLISH:
            result = self._analyze_english(text)
        else:
            # Unknown — try ensemble
            result = self._analyze_ensemble(text, language)

        # If code-switching detected, blend with ensemble
        if language.contains_code_switching and language.detected_language != Language.UNKNOWN:
            ensemble = self._analyze_ensemble(text, language)
            # Weighted blend: 70% primary, 30% ensemble
            result.score = result.score * 0.7 + ensemble.score * 0.3
            result.confidence *= 0.9  # Slightly lower confidence for mixed

        # Finalize
        result.label = self._score_to_label(result.score)
        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def _analyze_sheng(self, text: str) -> SentimentResult:
        """
        Sheng sentiment analysis using hybrid approach.

        1. Tokenize with ShengTokenizer (handles non-standard spelling)
        2. Look up sentiment bias for known Sheng words
        3. Apply modifiers (intensifiers flip/amplify scores)
        4. If transformer available, blend with model prediction
        """
        tokens = self.sheng_tokenizer.tokenize(text)

        # Rule-based scoring from Sheng lexicon
        lexicon_score = self._lexicon_score(text, tokens)

        # If transformer available, get model score too
        if self._model_loaded:
            model_score = self._transformer_score(text)
            # Blend: 40% lexicon (Sheng-specific), 60% model (general patterns)
            final_score = lexicon_score * 0.4 + model_score * 0.6
            confidence = 0.72  # Lower confidence for Sheng
        else:
            final_score = lexicon_score
            confidence = 0.55  # Rule-based only

        return SentimentResult(
            label=self._score_to_label(final_score),
            score=round(max(min(final_score, 1.0), -1.0), 4),
            confidence=round(confidence, 4),
            probabilities=self._score_to_probs(final_score),
            model_used="sheng-hybrid",
        )

    def _analyze_swahili(self, text: str) -> SentimentResult:
        """Swahili sentiment with phrase-level pattern matching."""
        # Check Swahili-specific phrases first
        phrase_score = 0.0
        phrase_matches = 0
        text_lower = text.lower()

        for phrase, score in self.SWAHILI_SENTIMENT.items():
            if phrase in text_lower:
                phrase_score += score
                phrase_matches += 1

        # General lexicon scoring
        tokens = self.sheng_tokenizer.tokenize(text)
        lexicon_score = self._lexicon_score(text, tokens)

        if phrase_matches > 0:
            combined = (phrase_score / phrase_matches) * 0.6 + lexicon_score * 0.4
        else:
            combined = lexicon_score

        # Transformer if available
        if self._model_loaded:
            model_score = self._transformer_score(text)
            final_score = combined * 0.3 + model_score * 0.7
            confidence = 0.82
        else:
            final_score = combined
            confidence = 0.60

        return SentimentResult(
            label=self._score_to_label(final_score),
            score=round(max(min(final_score, 1.0), -1.0), 4),
            confidence=round(confidence, 4),
            probabilities=self._score_to_probs(final_score),
            model_used="swahili-hybrid" if not self._model_loaded else "afrisenti-xlmr",
        )

    def _analyze_english(self, text: str) -> SentimentResult:
        """English sentiment — most straightforward, transformer preferred."""
        if self._model_loaded:
            score = self._transformer_score(text)
            confidence = 0.88
            model_name = "afrisenti-xlmr"
        else:
            tokens = self.sheng_tokenizer.tokenize(text)
            score = self._lexicon_score(text, tokens)
            confidence = 0.62
            model_name = "lexicon-rule-based"

        return SentimentResult(
            label=self._score_to_label(score),
            score=round(max(min(score, 1.0), -1.0), 4),
            confidence=round(confidence, 4),
            probabilities=self._score_to_probs(score),
            model_used=model_name,
        )

    def _analyze_ensemble(self, text: str, language: LanguageDetection) -> SentimentResult:
        """Ensemble of all strategies, weighted by language detection scores."""
        en_result = self._analyze_english(text)
        sw_result = self._analyze_swahili(text)
        sh_result = self._analyze_sheng(text)

        # Weight by language detection confidence
        scores = language.all_scores
        en_weight = scores.get("en", 0.33)
        sw_weight = scores.get("sw", 0.33)
        sh_weight = scores.get("sh", 0.33)

        total_weight = en_weight + sw_weight + sh_weight or 1.0
        final_score = (
            en_result.score * en_weight +
            sw_result.score * sw_weight +
            sh_result.score * sh_weight
        ) / total_weight

        avg_confidence = (
            en_result.confidence * en_weight +
            sw_result.confidence * sw_weight +
            sh_result.confidence * sh_weight
        ) / total_weight

        return SentimentResult(
            label=self._score_to_label(final_score),
            score=round(max(min(final_score, 1.0), -1.0), 4),
            confidence=round(avg_confidence * 0.85, 4),  # Ensemble penalty
            probabilities=self._score_to_probs(final_score),
            model_used="ensemble",
        )

    def _lexicon_score(self, text: str, tokens: List[ShengToken]) -> float:
        """
        Rule-based sentiment scoring using lexicons.

        Process:
        1. Sum sentiment biases from Sheng lexicon tokens
        2. Match positive/negative pattern words
        3. Apply modifiers (negation flips, intensifiers scale)
        """
        text_lower = text.lower()
        words = text_lower.split()

        if not words:
            return 0.0

        # Score from Sheng token metadata
        token_scores = []
        for token in tokens:
            if abs(token.sentiment_bias) > 0.01:
                token_scores.append(token.sentiment_bias)

        # Score from general lexicon
        for word in words:
            if word in self.POSITIVE_PATTERNS:
                token_scores.append(self.POSITIVE_PATTERNS[word])
            elif word in self.NEGATIVE_PATTERNS:
                token_scores.append(self.NEGATIVE_PATTERNS[word])

        if not token_scores:
            return 0.0

        raw_score = sum(token_scores) / len(token_scores)

        # Apply modifiers from ShengTokenizer
        modifiers = self.sheng_tokenizer.get_sentiment_modifiers(tokens)
        for pos, modifier in modifiers:
            if modifier < 0:  # Negation
                raw_score *= -0.8  # Flip with slight dampening
            else:  # Intensifier
                raw_score *= modifier

        return max(min(raw_score, 1.0), -1.0)

    def _transformer_score(self, text: str) -> float:
        """
        Get sentiment score from transformer model.

        Maps model output (typically 3-class: neg/neu/pos) to -1 to 1 scale.
        """
        if not self._model_loaded:
            return 0.0

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_config.max_length,
                padding=True,
            )

            if self.model_config.use_gpu and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._transformer_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            # Map: index 0=negative, 1=neutral, 2=positive
            neg, neu, pos = probs.tolist()
            score = pos - neg  # Range: -1 to 1

            return round(score, 4)

        except Exception as e:
            logger.error(f"Transformer inference failed: {e}")
            return 0.0

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Map continuous score to discrete label."""
        if score > 0.25:
            return SentimentLabel.POSITIVE
        elif score < -0.25:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL

    def _score_to_probs(self, score: float) -> Dict[str, float]:
        """Convert score to pseudo-probability distribution."""
        # Simple conversion from continuous score
        if score >= 0:
            pos_prob = 0.33 + (score * 0.67)
            neg_prob = 0.33 * (1 - score)
        else:
            neg_prob = 0.33 + (abs(score) * 0.67)
            pos_prob = 0.33 * (1 + score)

        neu_prob = max(0, 1.0 - pos_prob - neg_prob)

        total = pos_prob + neg_prob + neu_prob
        return {
            "positive": round(pos_prob / total, 4),
            "negative": round(neg_prob / total, 4),
            "neutral": round(neu_prob / total, 4),
        }

    async def analyze_batch(
        self,
        texts: List[str],
        languages: List[LanguageDetection],
    ) -> List[SentimentResult]:
        """Batch sentiment analysis."""
        results = []
        for text, lang in zip(texts, languages):
            results.append(self.analyze(text, lang))
        return results
