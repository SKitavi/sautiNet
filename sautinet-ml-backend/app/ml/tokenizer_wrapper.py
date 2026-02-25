"""
SentiKenya Transformer Tokenizer Wrapper
==========================================
Wraps HuggingFace AutoTokenizer for multilingual tokenisation.

Supports:
  - bert-base-multilingual-cased  (mBERT)
  - castorini/afriberta_base      (AfriBERTa — recommended for African languages)
  - xlm-roberta-base              (XLM-RoBERTa)
  - cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

Integrates with the existing ShengTokenizer: Sheng-detected text gets
rule-based normalisation first, then transformer tokenisation.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TransformerTokenizerWrapper:
    """
    Wrapper around HuggingFace AutoTokenizer that provides a unified
    interface for multilingual tokenisation of Kenyan social media text.

    The wrapper handles:
    - Loading any HuggingFace-compatible tokenizer by name
    - Max-length truncation and padding
    - Returning both token strings and model-ready tensors
    - Integration with Sheng pre-normalisation
    """

    SUPPORTED_MODELS = {
        "mbert": "bert-base-multilingual-cased",
        "afriberta": "castorini/afriberta_base",
        "xlm-roberta": "xlm-roberta-base",
        "afrisenti": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    }

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        cache_dir: str = "./model_cache",
    ):
        """
        Initialize the transformer tokenizer.

        Args:
            model_name: HuggingFace model name or one of the aliases
                        (mbert, afriberta, xlm-roberta, afrisenti).
            max_length: Maximum token sequence length.
            cache_dir: Directory for caching downloaded models.
        """
        resolved = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.model_name = resolved
        self.max_length = max_length
        self._tokenizer = None

        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                resolved, cache_dir=cache_dir
            )
            logger.info(
                f"TransformerTokenizerWrapper loaded: {resolved} "
                f"(vocab={self._tokenizer.vocab_size}, max_len={max_length})"
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer '{resolved}': {e}")
            raise

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size if self._tokenizer else 0

    def tokenize(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        padding: str = "max_length",
    ) -> Dict:
        """
        Tokenize a single text string.

        Args:
            text: Input text.
            return_tensors: "pt" for PyTorch, None for lists.
            padding: Padding strategy.

        Returns:
            Dict with keys: tokens, input_ids, attention_mask, token_count.
        """
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not loaded.")

        encoding = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=padding,
            return_tensors=return_tensors,
        )

        # Get human-readable tokens
        if return_tensors == "pt":
            ids = encoding["input_ids"].squeeze(0).tolist()
        else:
            ids = encoding["input_ids"]

        tokens = self._tokenizer.convert_ids_to_tokens(ids)

        # Strip padding tokens for the readable list
        readable = [t for t in tokens if t not in ("[PAD]", "<pad>")]

        return {
            "tokens": readable,
            "input_ids": ids if not return_tensors else encoding["input_ids"],
            "attention_mask": (
                encoding["attention_mask"].squeeze(0).tolist()
                if return_tensors == "pt"
                else encoding["attention_mask"]
            ),
            "token_count": len(readable),
        }

    def tokenize_batch(
        self,
        texts: List[str],
        return_tensors: str = "pt",
        padding: str = "max_length",
    ) -> Dict:
        """
        Tokenize a batch of texts.

        Returns dict with batched input_ids, attention_mask tensors.
        """
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not loaded.")

        encoding = self._tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=padding,
            return_tensors=return_tensors,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "batch_size": len(texts),
        }

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not loaded.")
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special)
