#!/usr/bin/env python3
"""
SentiKenya Preprocessing Pipeline
===================================
Standalone preprocessing script that takes raw text and outputs clean,
tokenized inputs ready for sentiment inference.

Supports English, Kiswahili, Sheng, and code-switched text.

Pipeline:
  1. Text cleaning (URLs, mentions, emojis, normalisation)
  2. Language detection
  3. Sheng / SMS normalisation (if detected)
  4. Tokenisation (rule-based + optional transformer tokenizer)
  5. Output: JSON with cleaned text, language, tokens

Usage:
    # Single text
    python scripts/preprocess.py --text "Manze hii economy ni tight bana"

    # From file (one post per line)
    python scripts/preprocess.py --file raw_posts.txt

    # From stdin
    echo "Serikali inafanya kazi nzuri" | python scripts/preprocess.py --stdin

    # With transformer tokenizer
    python scripts/preprocess.py --text "Hello world" --transformer
"""

import argparse
import json
import re
import sys
import os
import logging
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.language_detector import KenyanLanguageDetector
from app.ml.sheng_tokenizer import ShengTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("sentikenya.preprocess")

# ── Political hashtags to preserve as features ──
POLITICAL_HASHTAGS = {
    "kenyandecides", "elections2027", "hustlerfund", "taxreform",
    "anticorruption", "uhc", "genztakeover", "rejectfinancebill",
    "occupyparliament", "azimio", "kenyakwanza", "iebc",
    "buildingbridges", "bbi", "bottomup", "handshake",
}


class PreprocessingPipeline:
    """
    Full preprocessing pipeline for Kenyan social media text.

    Handles:
    - English, Kiswahili, Sheng, and code-switched text
    - URL / mention / emoji removal
    - Hashtag handling (political hashtags preserved as features)
    - SMS / phonetic normalisation via ShengTokenizer
    - Language detection
    - Rule-based tokenisation
    - Optional transformer tokenisation (mBERT / XLM-R)
    """

    def __init__(
        self,
        sheng_lexicon_path: str = "./data/sheng_lexicon.json",
        use_transformer: bool = False,
        transformer_model: str = "bert-base-multilingual-cased",
    ):
        self.language_detector = KenyanLanguageDetector(sheng_lexicon_path=sheng_lexicon_path)
        self.sheng_tokenizer = ShengTokenizer(lexicon_path=sheng_lexicon_path)
        self._transformer_tokenizer = None

        if use_transformer:
            self._load_transformer_tokenizer(transformer_model)

    def _load_transformer_tokenizer(self, model_name: str):
        """Load a HuggingFace transformer tokenizer."""
        try:
            from app.ml.tokenizer_wrapper import TransformerTokenizerWrapper
            self._transformer_tokenizer = TransformerTokenizerWrapper(model_name=model_name)
            logger.info(f"Transformer tokenizer loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load transformer tokenizer: {e}. Using rule-based only.")

    def clean_text(self, text: str) -> dict:
        """
        Clean raw social media text.

        Returns dict with cleaned text and extracted features.
        """
        original = text

        # Extract hashtags before cleaning
        raw_hashtags = re.findall(r"#(\w+)", text)
        political_tags = [t for t in raw_hashtags if t.lower() in POLITICAL_HASHTAGS]
        non_political_tags = [t for t in raw_hashtags if t.lower() not in POLITICAL_HASHTAGS]

        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Remove mentions but keep the text context
        text = re.sub(r"@\w+", "", text)

        # Preserve political hashtag text, remove # symbol
        for tag in political_tags:
            text = text.replace(f"#{tag}", f" __POL_{tag.upper()}__ ")

        # Remove remaining hashtag symbols but keep words
        text = re.sub(r"#(\w+)", r"\1", text)

        # Handle emojis — remove but count them for sentiment signal
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF"
            "\U00002702-\U000027B0\U0001FA00-\U0001FA6F]+",
            flags=re.UNICODE,
        )
        emojis_found = emoji_pattern.findall(text)
        text = emoji_pattern.sub(" ", text)

        # Normalise repeated chars: "noooooo" → "noo"
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # Remove remaining non-alphanumeric (keep apostrophes)
        text = re.sub(r"[^\w\s']", " ", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return {
            "cleaned_text": text,
            "hashtags": raw_hashtags,
            "political_hashtags": political_tags,
            "emojis_count": len(emojis_found),
            "original_length": len(original),
            "cleaned_length": len(text),
        }

    def process(self, text: str) -> dict:
        """
        Full preprocessing pipeline for a single text.

        Returns:
            dict with keys: cleaned_text, language, tokens, features, etc.
        """
        # Step 1: Clean
        cleaning = self.clean_text(text)
        cleaned = cleaning["cleaned_text"]

        # Step 2: Language detection
        lang_result = self.language_detector.detect(cleaned)
        lang = lang_result.detected_language.value

        # Step 3: Sheng / SMS normalisation (run always — it handles all langs gracefully)
        sheng_tokens = self.sheng_tokenizer.tokenize(cleaned)
        rule_tokens = [t.text for t in sheng_tokens]

        # Step 4: Sentiment modifiers (from Sheng tokenizer)
        sentiment_words = self.sheng_tokenizer.extract_sentiment_words(sheng_tokens)
        modifiers = self.sheng_tokenizer.get_sentiment_modifiers(sheng_tokens)

        # Step 5: Transformer tokenisation (optional)
        transformer_output = None
        if self._transformer_tokenizer:
            transformer_output = self._transformer_tokenizer.tokenize(cleaned)

        result = {
            "original_text": text,
            "cleaned_text": cleaned,
            "language": lang,
            "language_confidence": lang_result.confidence,
            "language_scores": lang_result.all_scores,
            "contains_code_switching": lang_result.contains_code_switching,
            "tokens": rule_tokens,
            "token_count": len(rule_tokens),
            "political_hashtags": cleaning["political_hashtags"],
            "sentiment_words": sentiment_words,
            "sentiment_modifiers": [{"position": p, "value": v} for p, v in modifiers],
        }

        if transformer_output:
            result["transformer_tokens"] = transformer_output["tokens"]
            result["transformer_input_ids"] = transformer_output["input_ids"]
            result["transformer_attention_mask"] = transformer_output["attention_mask"]

        return result

    def process_batch(self, texts: List[str]) -> List[dict]:
        """Process multiple texts."""
        return [self.process(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description="SentiKenya Preprocessing Pipeline")
    parser.add_argument("--text", type=str, help="Single text to preprocess")
    parser.add_argument("--file", type=str, help="File with one post per line")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--transformer", action="store_true", help="Also run transformer tokenizer")
    parser.add_argument("--model", default="bert-base-multilingual-cased", help="Transformer model name")
    parser.add_argument("--lexicon", default="./data/sheng_lexicon.json", help="Sheng lexicon path")
    parser.add_argument("--output", type=str, help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        sheng_lexicon_path=args.lexicon,
        use_transformer=args.transformer,
        transformer_model=args.model,
    )

    texts: List[str] = []

    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.stdin:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    else:
        parser.print_help()
        sys.exit(1)

    results = pipeline.process_batch(texts)

    output = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info(f"Results written to {args.output}")
    else:
        print(output)

    logger.info(f"Preprocessed {len(results)} text(s)")


if __name__ == "__main__":
    main()
