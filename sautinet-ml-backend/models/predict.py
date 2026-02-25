#!/usr/bin/env python3
"""
SentiKenya Inference Script
=============================
Loads a fine-tuned checkpoint and predicts sentiment for input text.

Returns: {label, confidence, language} for each input.

Usage:
    python models/predict.py --text "Serikali inafanya kazi nzuri"
    python models/predict.py --text "Manze hii economy ni tight bana"
    python models/predict.py --file posts.txt
    echo "Corruption is bad" | python models/predict.py --stdin
    python models/predict.py --text "hello" --model ./models/checkpoints/best
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add project root to path so we can use the language detector
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.language_detector import KenyanLanguageDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("sentikenya.predict")

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

DEFAULT_MODEL_PATH = "./models/checkpoints/best"
FALLBACK_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"


class SentiKenyaPredictor:
    """
    Loads a fine-tuned model checkpoint and performs sentiment inference.

    Returns {label, confidence, language} for any input text.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        lexicon_path: str = "./data/sheng_lexicon.json",
        device: str = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load model
        actual_path = model_path
        if not os.path.isdir(model_path):
            logger.warning(f"Checkpoint not found at {model_path}, using base model: {FALLBACK_MODEL}")
            actual_path = FALLBACK_MODEL

        logger.info(f"Loading model from: {actual_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(actual_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(actual_path)
        self.model.to(self.device)
        self.model.eval()

        # Language detector
        self.lang_detector = KenyanLanguageDetector(sheng_lexicon_path=lexicon_path)
        logger.info(f"Predictor ready on {self.device}")

    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text.

        Returns:
            {
                "label": "positive" | "negative" | "neutral",
                "confidence": float (0-1),
                "language": "en" | "sw" | "sh" | "unk"
            }
        """
        # Language detection
        lang_result = self.lang_detector.detect(text)
        language = lang_result.detected_language.value

        # Tokenize and infer
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        return {
            "label": LABEL_MAP[pred_idx],
            "confidence": round(confidence, 4),
            "language": language,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts."""
        return [self.predict(t) for t in texts]


def main():
    parser = argparse.ArgumentParser(description="SentiKenya Sentiment Predictor")
    parser.add_argument("--text", type=str, help="Single text to classify")
    parser.add_argument("--file", type=str, help="File with one post per line")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model checkpoint path")
    parser.add_argument("--lexicon", default="./data/sheng_lexicon.json")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    predictor = SentiKenyaPredictor(model_path=args.model, lexicon_path=args.lexicon)

    texts: List[str] = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file) as f:
            texts = [l.strip() for l in f if l.strip()]
    elif args.stdin:
        texts = [l.strip() for l in sys.stdin if l.strip()]
    else:
        parser.print_help()
        sys.exit(1)

    results = predictor.predict_batch(texts)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for text, res in zip(texts, results):
            print(f"\n  Text:       {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"  Label:      {res['label']}")
            print(f"  Confidence: {res['confidence']:.4f}")
            print(f"  Language:   {res['language']}")


if __name__ == "__main__":
    main()
