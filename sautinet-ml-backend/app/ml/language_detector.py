"""
SentiKenya Language Detector
=============================
Detects English, Swahili, and Sheng in social media text.
Sheng detection is custom-built since no standard NLP library supports it.
"""

import re
import json
import logging
from typing import Dict, Tuple
from pathlib import Path

from app.models.schemas import Language, LanguageDetection

logger = logging.getLogger(__name__)


class KenyanLanguageDetector:
    """
    Multi-stage language detector optimized for Kenyan social media.

    Detection strategy:
    1. Check for Sheng indicators first (most specific)
    2. Check Swahili grammar patterns
    3. Check English patterns
    4. Score code-switching ratio for mixed-language posts

    Sheng is the hardest to detect because:
    - No standardized orthography
    - Borrows heavily from English AND Swahili
    - Varies by Nairobi neighborhood
    - Evolves rapidly (new slang monthly)
    """

    # ── Swahili morphological patterns ──
    SWAHILI_PREFIXES = [
        "wa", "m", "ki", "vi", "u", "ku", "pa", "mu",
        "ni", "una", "ana", "tuna", "wana", "ina",
        "hana", "sina", "hatuna", "hawana",
    ]
    SWAHILI_SUFFIXES = ["isha", "ika", "ana", "wa", "iwa", "eza"]
    SWAHILI_FUNCTION_WORDS = {
        "na", "ya", "wa", "kwa", "ni", "la", "za", "cha",
        "vya", "katika", "hii", "hiyo", "hizi", "ile", "kama",
        "lakini", "au", "ama", "bali", "ingawa", "kwamba",
        "kuhusu", "pamoja", "kabla", "baada", "wakati",
        "sababu", "kila", "yote", "wote", "mengi", "nyingi",
    }
    SWAHILI_VERBS = {
        "anasema", "wanasema", "ninasema", "unasema",
        "anafanya", "wanafanya", "ninafanya",
        "anaenda", "wanaenda", "ninaenda",
        "anakuja", "wanakuja", "ninakuja",
        "anapenda", "tunapenda", "ninapenda",
        "anataka", "tunataka", "ninataka",
        "anajua", "tunajua", "ninajua",
        "anaishi", "tunaishi", "ninaishi",
    }

    # ── English patterns ──
    ENGLISH_FUNCTION_WORDS = {
        "the", "is", "are", "was", "were", "have", "has", "had",
        "will", "would", "could", "should", "can", "may", "might",
        "this", "that", "these", "those", "it", "they", "we",
        "with", "from", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "because",
        "government", "president", "people", "country", "need",
    }

    def __init__(self, sheng_lexicon_path: str = "./data/sheng_lexicon.json"):
        self.sheng_lexicon: Dict = {}
        self.sheng_indicators: set = set()
        self.code_switch_markers: set = set()
        self.negation_words: set = set()
        self._load_sheng_lexicon(sheng_lexicon_path)

    def _load_sheng_lexicon(self, path: str):
        """Load Sheng vocabulary and detection indicators."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.sheng_lexicon = data.get("vocabulary", {})
            self.sheng_indicators = set(data.get("sheng_detection_indicators", []))
            self.code_switch_markers = set(data.get("code_switching_markers", []))
            self.negation_words = set(data.get("negation_words", []))
            logger.info(f"Loaded Sheng lexicon: {len(self.sheng_lexicon)} words, "
                        f"{len(self.sheng_indicators)} indicators")
        except FileNotFoundError:
            logger.warning(f"Sheng lexicon not found at {path}, using empty lexicon")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Sheng lexicon: {e}")

    def _preprocess(self, text: str) -> list:
        """Normalize and tokenize text for analysis."""
        text = text.lower().strip()
        text = re.sub(r"https?://\S+", "", text)         # Remove URLs
        text = re.sub(r"@\w+", "", text)                  # Remove mentions
        text = re.sub(r"#(\w+)", r"\1", text)             # Keep hashtag text
        text = re.sub(r"[^\w\s']", " ", text)             # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        return tokens

    def _score_sheng(self, tokens: list) -> Tuple[float, list]:
        """
        Score likelihood of Sheng based on:
        1. Presence of known Sheng-specific words
        2. Code-switching patterns (English+Swahili mix)
        3. Informal phonetic spellings
        """
        if not tokens:
            return 0.0, []

        sheng_hits = []
        sheng_score = 0.0
        has_english = False
        has_swahili = False

        for token in tokens:
            # Direct Sheng indicator match
            if token in self.sheng_indicators:
                sheng_score += 2.0
                sheng_hits.append(token)

            # Sheng lexicon match
            elif token in self.sheng_lexicon:
                sheng_score += 1.5
                sheng_hits.append(token)

            # Code-switching marker
            elif token in self.code_switch_markers:
                sheng_score += 0.3

            # Track English/Swahili presence
            if token in self.ENGLISH_FUNCTION_WORDS:
                has_english = True
            if token in self.SWAHILI_FUNCTION_WORDS:
                has_swahili = True

        # Bonus for code-switching (Sheng hallmark)
        if has_english and has_swahili:
            sheng_score += 2.0

        # Normalize by token count
        normalized = sheng_score / max(len(tokens), 1)
        return min(normalized, 1.0), sheng_hits

    def _score_swahili(self, tokens: list) -> float:
        """Score Swahili likelihood based on morphological patterns."""
        if not tokens:
            return 0.0

        swahili_score = 0.0
        for token in tokens:
            if token in self.SWAHILI_FUNCTION_WORDS:
                swahili_score += 1.5
            elif token in self.SWAHILI_VERBS:
                swahili_score += 2.0
            else:
                # Check Swahili morphological patterns
                for prefix in self.SWAHILI_PREFIXES:
                    if token.startswith(prefix) and len(token) > len(prefix) + 2:
                        swahili_score += 0.5
                        break
                for suffix in self.SWAHILI_SUFFIXES:
                    if token.endswith(suffix) and len(token) > len(suffix) + 2:
                        swahili_score += 0.3
                        break

        return min(swahili_score / max(len(tokens), 1), 1.0)

    def _score_english(self, tokens: list) -> float:
        """Score English likelihood based on function words and patterns."""
        if not tokens:
            return 0.0

        english_score = 0.0
        for token in tokens:
            if token in self.ENGLISH_FUNCTION_WORDS:
                english_score += 1.5
            # English-specific patterns
            elif token.endswith("ing") or token.endswith("tion") or token.endswith("ness"):
                english_score += 0.8
            elif token.endswith("ly") or token.endswith("ed"):
                english_score += 0.5

        return min(english_score / max(len(tokens), 1), 1.0)

    def detect(self, text: str) -> LanguageDetection:
        """
        Detect language of input text.

        Returns LanguageDetection with scores for all three languages.
        """
        tokens = self._preprocess(text)

        if not tokens:
            return LanguageDetection(
                detected_language=Language.UNKNOWN,
                confidence=0.0,
                all_scores={"en": 0, "sw": 0, "sh": 0},
            )

        # Score each language
        sheng_score, sheng_hits = self._score_sheng(tokens)
        swahili_score = self._score_swahili(tokens)
        english_score = self._score_english(tokens)

        # Determine primary language
        scores = {
            Language.SHENG: sheng_score,
            Language.SWAHILI: swahili_score,
            Language.ENGLISH: english_score,
        }

        # Sheng gets priority when score is above threshold (it's the most specific)
        if sheng_score > 0.25:
            detected = Language.SHENG
            confidence = min(sheng_score * 1.2, 0.99)
        else:
            detected = max(scores, key=scores.get)
            confidence = scores[detected]

        # Check for code-switching
        non_zero_langs = sum(1 for s in scores.values() if s > 0.15)
        code_switching = non_zero_langs >= 2

        # Normalize confidence to 0-1 range
        total = sum(scores.values()) or 1.0
        normalized_scores = {
            lang.value: round(score / total, 4) for lang, score in scores.items()
        }

        return LanguageDetection(
            detected_language=detected,
            confidence=round(min(max(confidence, 0.3), 0.99), 4),
            all_scores=normalized_scores,
            contains_code_switching=code_switching,
            sheng_indicators=sheng_hits[:5],
        )
