"""
SentiKenya Sheng Tokenizer
===========================
Custom tokenizer for Sheng that handles:
- Non-standard orthography (multiple spellings for same word)
- Code-switching between English, Swahili, and Sheng
- Informal internet/SMS-style abbreviations
- Phonetic spellings common in Kenyan social media

This is critical because standard NLP tokenizers fail on Sheng since:
1. No standardized dictionary exists
2. Words are borrowed and modified from multiple source languages
3. Spelling varies by neighborhood, age group, and platform
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ShengToken:
    """A single token with Sheng-specific metadata."""
    text: str
    normalized: str              # Standardized form
    source_language: str         # "sheng", "swahili", "english", "mixed"
    sentiment_bias: float = 0.0  # Inherent sentiment lean (-1 to 1)
    is_intensifier: bool = False
    is_negation: bool = False
    is_slang: bool = False
    english_gloss: str = ""      # English translation
    position: int = 0


class ShengTokenizer:
    """
    Tokenizer specifically designed for Kenyan Sheng.

    Pipeline:
    1. Text normalization (SMS speak, phonetic spelling)
    2. Whitespace tokenization
    3. Sheng vocabulary lookup
    4. Code-switch boundary detection
    5. Sentiment modifier tagging (intensifiers, negations)
    """

    # ── SMS/Internet abbreviations common in Kenyan social media ──
    SMS_MAP = {
        "bt": "but", "cz": "because", "bcz": "because", "bcs": "because",
        "cn": "can", "dnt": "don't", "dnt": "dont", "2day": "today",
        "2moro": "tomorrow", "2morrow": "tomorrow", "2nyt": "tonight",
        "4": "for", "b4": "before", "ur": "your", "u": "you",
        "r": "are", "n": "and", "d": "the", "da": "the",
        "nt": "not", "nw": "now", "hw": "how", "wt": "what",
        "whn": "when", "tht": "that", "thx": "thanks",
        "pls": "please", "plz": "please",
        "gud": "good", "gd": "good",
        "abt": "about", "jst": "just", "wnt": "want",
        "kno": "know", "knw": "know",
        "smth": "something", "smone": "someone",
        "govt": "government", "gvt": "government",
        "pple": "people", "ppl": "people",
    }

    # ── Phonetic normalizations (multiple spellings -> canonical form) ──
    PHONETIC_MAP = {
        # Sheng variants
        "manzi": "manze", "manzeh": "manze", "manzee": "manze",
        "banaa": "bana", "banna": "bana",
        "wassup": "niaje", "wasap": "niaje",
        "wazzee": "wasee", "wase": "wasee",
        "mazee": "maze", "mazeh": "maze",
        "sasa": "sasa",
        # Common Swahili misspellings on social media
        "serekali": "serikali", "serkali": "serikali",
        "rushwa": "rushwa",
        "polis": "polisi", "police": "polisi",
        # Kenyan English adaptations
        "mathree": "matatu", "mathri": "matatu", "mat": "matatu",
        "jamo": "jamhuri", "jamo": "jamhuri",
    }

    # ── Compound Sheng expressions ──
    COMPOUNDS = {
        ("ni", "noma"): "ni_noma",      # "it's tough"
        ("si", "poa"): "si_poa",         # "not cool"
        ("iko", "tight"): "iko_tight",   # "it's difficult"
        ("hii", "life"): "hii_life",     # "this life"
        ("tuko", "rada"): "tuko_rada",   # "we're alert"
        ("no", "cap"): "no_cap",         # "no lie" (Gen-Z Sheng)
        ("on", "God"): "on_god",         # emphasis
    }

    def __init__(self, lexicon_path: str = "./data/sheng_lexicon.json"):
        self.vocabulary: Dict = {}
        self.intensifiers: Dict[str, float] = {}
        self.negation_words: set = set()
        self._load_lexicon(lexicon_path)

    def _load_lexicon(self, path: str):
        """Load Sheng vocabulary from lexicon file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.vocabulary = data.get("vocabulary", {})
            self.intensifiers = data.get("sentiment_intensifiers", {})
            self.negation_words = set(data.get("negation_words", []))
            logger.info(f"ShengTokenizer loaded: {len(self.vocabulary)} vocabulary entries")
        except Exception as e:
            logger.warning(f"Failed to load Sheng lexicon: {e}")

    def _normalize_text(self, text: str) -> str:
        """Pre-tokenization text normalization."""
        text = text.lower().strip()

        # Remove URLs, mentions but keep hashtag text
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#(\w+)", r" \1 ", text)

        # Normalize repeated characters: "pooooa" -> "poa"
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # Normalize emoji-adjacent text
        text = re.sub(r"[^\w\s']", " ", text)

        # Expand SMS abbreviations
        tokens = text.split()
        expanded = []
        for t in tokens:
            expanded.append(self.SMS_MAP.get(t, t))
        text = " ".join(expanded)

        # Apply phonetic normalizations
        tokens = text.split()
        normalized = []
        for t in tokens:
            normalized.append(self.PHONETIC_MAP.get(t, t))

        return " ".join(normalized)

    def _detect_compounds(self, tokens: List[str]) -> List[str]:
        """Merge multi-word Sheng expressions into single tokens."""
        if len(tokens) < 2:
            return tokens

        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1:
                pair = (tokens[i], tokens[i + 1])
                if pair in self.COMPOUNDS:
                    merged.append(self.COMPOUNDS[pair])
                    i += 2
                    continue
            merged.append(tokens[i])
            i += 1
        return merged

    def tokenize(self, text: str) -> List[ShengToken]:
        """
        Full tokenization pipeline.

        Returns list of ShengToken with metadata for downstream processing.
        """
        # Step 1: Normalize
        normalized_text = self._normalize_text(text)
        raw_tokens = normalized_text.split()

        # Step 2: Detect compounds
        raw_tokens = self._detect_compounds(raw_tokens)

        # Step 3: Build ShengTokens with metadata
        tokens = []
        for i, raw in enumerate(raw_tokens):
            token = self._analyze_token(raw, i)
            tokens.append(token)

        return tokens

    def _analyze_token(self, text: str, position: int) -> ShengToken:
        """Analyze a single token and return enriched ShengToken."""
        # Check Sheng vocabulary
        if text in self.vocabulary:
            entry = self.vocabulary[text]
            return ShengToken(
                text=text,
                normalized=text,
                source_language="sheng",
                sentiment_bias=entry.get("sentiment_bias", 0.0),
                is_slang=True,
                english_gloss=entry.get("en", ""),
                position=position,
            )

        # Check if it's an intensifier
        if text in self.intensifiers:
            return ShengToken(
                text=text,
                normalized=text,
                source_language="swahili",
                is_intensifier=True,
                sentiment_bias=0.0,
                position=position,
            )

        # Check if it's a negation
        if text in self.negation_words:
            return ShengToken(
                text=text,
                normalized=text,
                source_language="swahili",
                is_negation=True,
                sentiment_bias=0.0,
                position=position,
            )

        # Determine source language heuristically
        source = self._guess_source_language(text)

        return ShengToken(
            text=text,
            normalized=text,
            source_language=source,
            position=position,
        )

    def _guess_source_language(self, token: str) -> str:
        """Guess whether a token is English, Swahili, or unknown."""
        # Simple heuristic based on common patterns
        swahili_patterns = ["ni", "na", "wa", "ya", "za", "ki", "vi", "ku"]
        english_suffixes = ["ing", "tion", "ness", "ment", "able", "ful", "less", "ly", "ed"]

        for suffix in english_suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                return "english"

        for prefix in swahili_patterns:
            if token.startswith(prefix) and len(token) > 3:
                return "swahili"

        return "unknown"

    def get_sentiment_modifiers(self, tokens: List[ShengToken]) -> List[Tuple[int, float]]:
        """
        Extract sentiment modifiers (intensifiers and negations) with their positions.

        Returns list of (position, modifier_value) tuples.
        Negation: -1.0, Intensifiers: their respective multiplier values.
        """
        modifiers = []
        for token in tokens:
            if token.is_negation:
                modifiers.append((token.position, -1.0))
            elif token.is_intensifier:
                mult = self.intensifiers.get(token.text, 1.0)
                modifiers.append((token.position, mult))
        return modifiers

    def extract_sentiment_words(self, tokens: List[ShengToken]) -> Dict[str, float]:
        """Extract words with inherent sentiment bias."""
        sentiment_words = {}
        for token in tokens:
            if abs(token.sentiment_bias) > 0.05:
                sentiment_words[token.text] = token.sentiment_bias
        return sentiment_words
