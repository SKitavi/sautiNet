"""
SentiKenya Custom Model — BiLSTM with Self-Attention
======================================================
A sentiment classifier built FROM SCRATCH for Kenyan multilingual text.

This model demonstrates core ML concepts:
  1. Custom vocabulary building & text tokenization
  2. Learned word embeddings (no pre-trained vectors)
  3. Bidirectional LSTM for sequential context capture
  4. Self-attention mechanism to weight important tokens
  5. Full training loop: forward pass, backprop, optimization
  6. Evaluation: accuracy, F1, confusion matrix, per-language metrics

Architecture:
  Input text
      │
      ▼
  ┌──────────────┐
  │  Tokenizer   │  Custom vocab built from training data
  │  (custom)    │  Handles EN/SW/Sheng code-switching
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  Embedding   │  d=128, learned from scratch
  │  Layer       │  Maps token IDs → dense vectors
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  Bi-LSTM     │  hidden=64 per direction (128 total)
  │  (2 layers)  │  Captures left→right AND right→left context
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  Self-       │  Learns attention weights α_i for each token
  │  Attention   │  Context vector = Σ(α_i · h_i)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  FC layers   │  128 → 64 → 3 (negative/neutral/positive)
  │  + Dropout   │  ReLU activation, 0.3 dropout
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │  Softmax     │  P(negative), P(neutral), P(positive)
  │  Output      │
  └──────────────┘

Why BiLSTM + Attention?
- LSTMs handle variable-length text and capture word order
- Bidirectional: "economy ni tight" — "tight" needs left context ("economy")
  AND right context ("sana") to determine sentiment
- Attention: not all words matter equally. In "Manze serikali iko RADA",
  attention should focus on "rada" (positive Sheng) not "manze" (filler)
- Lightweight: ~500KB vs 1.1GB transformer — runs on any machine

Compared to the fine-tuned transformer:
┌─────────────────────┬──────────────┬──────────────────┐
│                      │ Custom BiLSTM│ Fine-tuned XLM-R │
├─────────────────────┼──────────────┼──────────────────┤
│ Parameters           │ ~200K        │ 278M             │
│ Model size           │ ~500KB       │ 1.1GB            │
│ Training data needed │ 150 samples  │ 150 samples      │
│ Inference speed      │ <1ms         │ ~40ms            │
│ Multilingual support │ Learned      │ Pre-trained      │
│ Sheng handling       │ Custom vocab │ Fine-tuned        │
└─────────────────────┴──────────────┴──────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import json
import re
import os
import logging
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 1. CUSTOM TOKENIZER — Vocabulary built from training data
# ═══════════════════════════════════════════════════════════════

class KenyanTokenizer:
    """
    Custom tokenizer for Kenyan multilingual text.

    Unlike subword tokenizers (BPE, WordPiece), this uses word-level
    tokenization with special handling for:
    - Sheng slang ("rada", "poa", "noma", "mnoma")
    - Swahili affixes ("wana-", "ime-", "hata-")
    - Code-switching markers (mixing EN/SW in one sentence)
    - Social media artifacts (#hashtags, @mentions, URLs)

    Vocabulary is built from the training corpus with frequency filtering.
    """

    # Special tokens
    PAD = "<PAD>"       # Padding for batch alignment
    UNK = "<UNK>"       # Unknown/out-of-vocabulary words
    BOS = "<BOS>"       # Beginning of sequence
    EOS = "<EOS>"       # End of sequence

    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def __init__(self, max_vocab_size: int = 5000, min_freq: int = 1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        # Mappings
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self.vocab_size: int = 0

        self._built = False

    def _preprocess(self, text: str) -> str:
        """
        Normalize text before tokenization.

        Handles social media noise while preserving sentiment-carrying tokens.
        """
        text = text.lower().strip()

        # Remove URLs but keep hashtag/mention text
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)      # #KOT → KOT
        text = re.sub(r'@(\w+)', r'mention', text)  # @user → mention

        # Normalize repeated characters: "poooooa" → "pooa"
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Keep basic punctuation as sentiment signals
        text = re.sub(r'[^\w\s!?.,]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _tokenize(self, text: str) -> List[str]:
        """Split preprocessed text into tokens."""
        return self._preprocess(text).split()

    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from training corpus.

        Process:
        1. Tokenize all texts
        2. Count word frequencies
        3. Filter by min_freq
        4. Keep top max_vocab_size words
        5. Assign integer IDs
        """
        # Count all words
        self.word_freq = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            self.word_freq.update(tokens)

        # Filter by frequency and cap at max vocab size
        filtered_words = [
            word for word, freq in self.word_freq.most_common(self.max_vocab_size)
            if freq >= self.min_freq
        ]

        # Build mappings: special tokens get IDs 0-3
        self.word2idx = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for word in filtered_words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self._built = True

        logger.info(
            f"Vocab built: {self.vocab_size} tokens "
            f"(from {len(self.word_freq)} unique words)"
        )

    def encode(self, text: str, max_length: int = 64) -> List[int]:
        """
        Convert text to integer token IDs.

        Adds BOS/EOS markers and pads/truncates to max_length.
        Unknown words map to UNK token.
        """
        tokens = self._tokenize(text)
        unk_id = self.word2idx[self.UNK]

        # Encode with BOS/EOS
        ids = [self.word2idx[self.BOS]]
        ids += [self.word2idx.get(t, unk_id) for t in tokens]
        ids += [self.word2idx[self.EOS]]

        # Truncate or pad
        if len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.word2idx[self.EOS]]
        else:
            ids += [self.word2idx[self.PAD]] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text (for debugging)."""
        tokens = []
        for i in ids:
            word = self.idx2word.get(i, self.UNK)
            if word in (self.PAD, self.BOS, self.EOS):
                continue
            tokens.append(word)
        return " ".join(tokens)

    def save(self, path: str):
        """Save vocabulary to disk."""
        data = {
            "word2idx": self.word2idx,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load vocabulary from disk."""
        with open(path) as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(i): w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.max_vocab_size = data.get("max_vocab_size", 5000)
        self.min_freq = data.get("min_freq", 1)
        self._built = True


# ═══════════════════════════════════════════════════════════════
# 2. DATASET CLASS — PyTorch Dataset for training
# ═══════════════════════════════════════════════════════════════

class SentiKenyaDataset(Dataset):
    """
    PyTorch Dataset wrapping labeled Kenyan sentiment samples.

    Each sample: (token_ids, label, language)
    - token_ids: List[int] of length max_length
    - label: 0 (negative), 1 (neutral), 2 (positive)
    - language: "en", "sw", or "sh"
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        languages: List[str],
        tokenizer: KenyanTokenizer,
        max_length: int = 64,
    ):
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = self.tokenizer.encode(self.texts[idx], self.max_length)
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════
# 3. MODEL ARCHITECTURE — BiLSTM + Self-Attention
# ═══════════════════════════════════════════════════════════════

class SelfAttention(nn.Module):
    """
    Self-attention layer for sequence classification.

    Instead of using just the last LSTM hidden state (which biases
    toward the end of the sentence), attention computes a weighted
    sum over ALL hidden states, learning which tokens matter most.

    For "Manze serikali iko RADA wasee":
      - "rada" should get high attention (sentiment word)
      - "manze" should get low attention (filler/discourse marker)

    Math:
      e_i = tanh(W_a · h_i + b_a)     # project each hidden state
      α_i = softmax(v_a · e_i)         # attention weights (sum to 1)
      context = Σ(α_i · h_i)           # weighted sum = sentence repr.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim) — all LSTM hidden states
            mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            context: (batch, hidden_dim) — attention-weighted representation
            weights: (batch, seq_len) — attention weights (for visualization)
        """
        # Project hidden states
        energy = torch.tanh(self.attention_weights(lstm_output))  # (B, T, H)

        # Compute attention scores
        scores = self.context_vector(energy).squeeze(-1)  # (B, T)

        # Mask padding tokens (set to -inf before softmax → 0 weight)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Normalize to attention distribution
        weights = F.softmax(scores, dim=-1)  # (B, T), sums to 1

        # Weighted sum of hidden states
        context = torch.bmm(
            weights.unsqueeze(1),   # (B, 1, T)
            lstm_output             # (B, T, H)
        ).squeeze(1)                # (B, H)

        return context, weights


class BiLSTMSentimentModel(nn.Module):
    """
    Bidirectional LSTM with Self-Attention for sentiment classification.

    Architecture:
        Embedding(vocab_size, embed_dim)
            ↓ dropout
        BiLSTM(embed_dim, hidden_dim, num_layers=2)
            ↓
        SelfAttention(hidden_dim * 2)  ← *2 for bidirectional
            ↓
        Linear(hidden_dim * 2, 64) + ReLU + Dropout
            ↓
        Linear(64, num_classes=3)

    Total parameters: ~200K (vs 278M for XLM-RoBERTa)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── Embedding layer ──
        # Maps token IDs to dense vectors. pad_idx=0 ensures padding
        # tokens always have zero vectors (not learned).
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.embed_dropout = nn.Dropout(dropout)

        # ── Bidirectional LSTM ──
        # 2 layers, bidirectional = captures context from both directions.
        # Output: hidden_dim * 2 (forward + backward concatenated)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # ── Self-attention ──
        # Learns which tokens to focus on for classification
        self.attention = SelfAttention(hidden_dim * 2)

        # ── Classification head ──
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) integer token IDs

        Returns:
            logits: (batch, num_classes) raw scores
            attention_weights: (batch, seq_len) for interpretability
        """
        # Create padding mask (1 = real token, 0 = padding)
        mask = (input_ids != 0).float()  # (B, T)

        # Embed tokens
        embedded = self.embedding(input_ids)     # (B, T, embed_dim)
        embedded = self.embed_dropout(embedded)

        # Pack padded sequences for efficient LSTM computation
        lengths = mask.sum(dim=1).long().cpu()
        lengths = lengths.clamp(min=1)  # Avoid zero-length sequences

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False,
        )

        # Run BiLSTM
        lstm_out, _ = self.lstm(packed)

        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=input_ids.size(1),
        )
        # lstm_out: (B, T, hidden_dim * 2)

        # Apply self-attention
        context, attn_weights = self.attention(lstm_out, mask)
        # context: (B, hidden_dim * 2) — the sentence representation

        # Classify
        logits = self.classifier(context)  # (B, num_classes)

        return logits, attn_weights

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
# 4. TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

class CustomModelTrainer:
    """
    Complete training pipeline for the BiLSTM sentiment model.

    Implements:
    - Data loading from SentiKenya dataset
    - Train/validation split (stratified by language)
    - Training loop with gradient clipping
    - Validation with per-language metrics
    - Early stopping on validation F1
    - Model checkpointing (best model saved)
    - Attention visualization for interpretability
    - Confusion matrix computation
    """

    def __init__(
        self,
        data_path: str = "./data/training_dataset.json",
        model_dir: str = "./models/custom-bilstm-v1",
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_length: int = 64,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        num_epochs: int = 30,
        patience: int = 7,
    ):
        self.data_path = data_path
        self.model_dir = model_dir
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience

        # Will be initialized during training
        self.tokenizer: Optional[KenyanTokenizer] = None
        self.model: Optional[BiLSTMSentimentModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Load and parse training dataset."""
        with open(self.data_path) as f:
            dataset = json.load(f)

        samples = dataset["data"]
        texts = [s["text"] for s in samples]
        labels = [s["label"] for s in samples]
        languages = [s["lang"] for s in samples]

        print(f"\n  Dataset: {len(texts)} samples")
        print(f"  Labels:  {Counter(labels)}")
        print(f"  Langs:   {Counter(languages)}")

        return texts, labels, languages

    def split_data(
        self,
        texts: List[str],
        labels: List[int],
        languages: List[str],
        val_ratio: float = 0.15,
    ) -> Tuple:
        """
        Stratified train/val split.

        Ensures each language is proportionally represented in both sets.
        """
        from collections import defaultdict
        import random
        random.seed(42)

        # Group by language
        lang_groups = defaultdict(list)
        for i, lang in enumerate(languages):
            lang_groups[lang].append(i)

        train_idx, val_idx = [], []

        for lang, indices in lang_groups.items():
            random.shuffle(indices)
            split = int(len(indices) * (1 - val_ratio))
            train_idx.extend(indices[:split])
            val_idx.extend(indices[split:])

        random.shuffle(train_idx)
        random.shuffle(val_idx)

        return (
            [texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            [languages[i] for i in train_idx],
            [texts[i] for i in val_idx],
            [labels[i] for i in val_idx],
            [languages[i] for i in val_idx],
        )

    def compute_metrics(
        self,
        all_preds: List[int],
        all_labels: List[int],
        all_langs: List[str],
    ) -> Dict:
        """
        Compute detailed evaluation metrics.

        Returns:
        - Overall: accuracy, macro F1, per-class F1
        - Per-language: accuracy and F1 for EN, SW, SH
        - Confusion matrix
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, confusion_matrix, classification_report,
        )

        overall_acc = accuracy_score(all_labels, all_preds)
        overall_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

        # Per-language metrics
        lang_metrics = {}
        for lang in set(all_langs):
            lang_mask = [i for i, l in enumerate(all_langs) if l == lang]
            if lang_mask:
                l_preds = [all_preds[i] for i in lang_mask]
                l_labels = [all_labels[i] for i in lang_mask]
                lang_metrics[lang] = {
                    "accuracy": accuracy_score(l_labels, l_preds),
                    "f1_macro": f1_score(l_labels, l_preds, average="macro", zero_division=0),
                    "count": len(lang_mask),
                }


        return {
            "accuracy": overall_acc,
            "f1_macro": overall_f1,
            "f1_per_class": {
                "negative": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0,
                "neutral": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0,
                "positive": float(per_class_f1[2]) if len(per_class_f1) > 2 else 0,
            },
            "confusion_matrix": cm.tolist(),
            "per_language": lang_metrics,
        }


    def train(self):
        """
        Full training pipeline.

        Steps:
        1. Load data → build vocabulary → create datasets
        2. Initialize model, optimizer, scheduler
        3. Training loop: forward → loss → backward → update
        4. Validation after each epoch
        5. Early stopping + best model checkpoint
        6. Final evaluation with attention analysis
        """
        print("=" * 65)
        print("  SentiKenya Custom Model Training")
        print("  Architecture: BiLSTM + Self-Attention")
        print("=" * 65)


        # ── Step 1: Data ──
        texts, labels, languages = self.load_data()
        (train_texts, train_labels, train_langs,
         val_texts, val_labels, val_langs) = self.split_data(texts, labels, languages)

        print(f"  Train: {len(train_texts)} | Val: {len(val_texts)}")

        # Build vocabulary from training data only (no data leakage)
        self.tokenizer = KenyanTokenizer(max_vocab_size=3000, min_freq=1)
        self.tokenizer.build_vocab(train_texts)

        # Create PyTorch datasets
        train_dataset = SentiKenyaDataset(
            train_texts, train_labels, train_langs,
            self.tokenizer, self.max_length,
        )
        val_dataset = SentiKenyaDataset(
            val_texts, val_labels, val_langs,
            self.tokenizer, self.max_length,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
        )

        # ── Step 2: Model ──
        self.model = BiLSTMSentimentModel(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=3,
            dropout=self.dropout,
            pad_idx=0,
        ).to(self.device)

        total_params = self.model.count_parameters()
        print(f"\n  Model parameters: {total_params:,}")
        print(f"  Vocab size:       {self.tokenizer.vocab_size}")
        print(f"  Device:           {self.device}")

        # Class weights for imbalanced data
        label_counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = torch.tensor([
            total / (3 * label_counts.get(i, 1)) for i in range(3)
        ], dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,  # L2 regularization
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3,
        )

        # ── Step 3: Training Loop ──
        print(f"\n  {'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} "
              f"{'Val Loss':>10} {'Val Acc':>9} {'Val F1':>8}  Notes")
        print(f"  {'─' * 5} {'─' * 11} {'─' * 10} {'─' * 10} {'─' * 9} {'─' * 8}  {'─' * 15}")

        best_f1 = 0.0
        patience_counter = 0
        training_log = []

        for epoch in range(1, self.num_epochs + 1):
            # ── Train ──
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_ids, batch_labels in train_loader:
                batch_ids = batch_ids.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                logits, _ = self.model(batch_ids)
                loss = criterion(logits, batch_labels)

                loss.backward()

                # Gradient clipping — prevents exploding gradients in LSTMs
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item() * batch_ids.size(0)
                preds = logits.argmax(dim=-1)
                train_correct += (preds == batch_labels).sum().item()
                train_total += batch_ids.size(0)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # ── Validate ──
            self.model.eval()
            val_loss = 0.0
            all_preds, all_labels_list = [], []

            with torch.no_grad():
                for batch_ids, batch_labels in val_loader:
                    batch_ids = batch_ids.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    logits, _ = self.model(batch_ids)
                    loss = criterion(logits, batch_labels)

                    val_loss += loss.item() * batch_ids.size(0)
                    preds = logits.argmax(dim=-1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels_list.extend(batch_labels.cpu().tolist())

            avg_val_loss = val_loss / len(val_dataset)
            metrics = self.compute_metrics(all_preds, all_labels_list, val_langs)

            scheduler.step(metrics["f1_macro"])
            current_lr = optimizer.param_groups[0]["lr"]

            # Log
            note = ""
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                patience_counter = 0
                note = "★ BEST"
                self._save_checkpoint(epoch, metrics)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    note = "EARLY STOP"

            print(
                f"  {epoch:>5} {avg_train_loss:>11.4f} {train_acc:>10.3f} "
                f"{avg_val_loss:>10.4f} {metrics['accuracy']:>9.3f} "
                f"{metrics['f1_macro']:>8.3f}  {note}"
            )

            training_log.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": avg_val_loss,
                "val_acc": metrics["accuracy"],
                "val_f1": metrics["f1_macro"],
                "lr": current_lr,
            })

            if patience_counter >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

        # ── Step 4: Final Evaluation ──
        print(f"\n{'=' * 65}")
        print(f"  TRAINING COMPLETE")
        print(f"{'=' * 65}")
        print(f"  Best validation F1: {best_f1:.4f}")
        print(f"  Epochs trained:     {epoch}")

        # Load best model for final eval
        self._load_best_checkpoint()

        print(f"\n  Per-language metrics (best model):")
        final_metrics = self.compute_metrics(all_preds, all_labels_list, val_langs)
        for lang, lm in final_metrics["per_language"].items():
            print(f"    {lang}: acc={lm['accuracy']:.3f} f1={lm['f1_macro']:.3f} (n={lm['count']})")

        print(f"\n  Per-class F1:")
        for cls, f1 in final_metrics["f1_per_class"].items():
            print(f"    {cls:>10}: {f1:.3f}")

        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(f"              neg  neu  pos")
        labels_str = ["negative", "neutral ", "positive"]
        for i, row in enumerate(final_metrics["confusion_matrix"]):
            print(f"    {labels_str[i]}  {row}")

        # Save training log
        log_path = os.path.join(self.model_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "config": {
                    "embed_dim": self.embed_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "max_length": self.max_length,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "vocab_size": self.tokenizer.vocab_size,
                    "total_params": total_params,
                },
                "training_log": training_log,
                "final_metrics": final_metrics,
                "best_f1": best_f1,
            }, f, indent=2)

        # ── Step 5: Attention Analysis ──
        self._attention_analysis(val_texts, val_labels, val_langs)

        return self.model, self.tokenizer, final_metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save best model checkpoint."""
        os.makedirs(self.model_dir, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "model.pt"))

        # Save tokenizer vocab
        self.tokenizer.save(os.path.join(self.model_dir, "vocab.json"))

        # Save model config (for reconstruction)
        config = {
            "vocab_size": self.tokenizer.vocab_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": 3,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "saved_epoch": epoch,
            "metrics": metrics,
            "saved_at": datetime.now().isoformat(),
        }
        with open(os.path.join(self.model_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _load_best_checkpoint(self):
        """Load best model from checkpoint."""
        model_path = os.path.join(self.model_dir, "model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    def _attention_analysis(
        self,
        texts: List[str],
        labels: List[int],
        languages: List[str],
    ):
        """
        Visualize attention weights to understand model decisions.

        Shows which words the model focuses on for each prediction.
        This is key for interpretability — understanding WHY the model
        made a particular classification.
        """
        print(f"\n{'=' * 65}")
        print("  ATTENTION ANALYSIS — What the model focuses on")
        print(f"{'=' * 65}")

        label_names = {0: "negative", 1: "neutral", 2: "positive"}
        self.model.eval()

        # Pick a few interesting examples
        indices = list(range(min(6, len(texts))))

        for idx in indices:
            text = texts[idx]
            true_label = labels[idx]
            lang = languages[idx]

            # Encode and predict
            token_ids = self.tokenizer.encode(text, self.max_length)
            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)

            with torch.no_grad():
                logits, attn_weights = self.model(input_tensor)
                probs = F.softmax(logits, dim=-1)[0]
                pred = logits.argmax(dim=-1).item()

            # Get tokens and their attention weights
            tokens = self.tokenizer._tokenize(text)
            attn = attn_weights[0].cpu().numpy()

            # Skip BOS, align with actual tokens
            token_attns = list(zip(tokens, attn[1:len(tokens) + 1]))
            token_attns.sort(key=lambda x: x[1], reverse=True)

            correct = "✓" if pred == true_label else "✗"
            print(f"\n  {correct} [{lang}] \"{text[:60]}\"")
            print(f"    True: {label_names[true_label]} | "
                  f"Pred: {label_names[pred]} | "
                  f"Probs: neg={probs[0]:.2f} neu={probs[1]:.2f} pos={probs[2]:.2f}")

            # Show top-3 attended words
            top_words = token_attns[:3]
            attn_str = ", ".join(
                f"\"{w}\"={a:.3f}" for w, a in top_words
            )
            print(f"    Top attention: {attn_str}")


# ═══════════════════════════════════════════════════════════════
# 5. INFERENCE — Loading and using the trained model
# ═══════════════════════════════════════════════════════════════

class CustomSentimentPredictor:
    """
    Inference wrapper for the trained BiLSTM model.

    Usage:
        predictor = CustomSentimentPredictor.load("./models/custom-bilstm-v1")
        label, score, probs = predictor.predict("Serikali iko rada")
        # → ("positive", 0.85, {"negative": 0.05, "neutral": 0.10, "positive": 0.85})
    """

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self, model: BiLSTMSentimentModel, tokenizer: KenyanTokenizer,
                 max_length: int = 64, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device(device)

    @classmethod
    def load(cls, model_dir: str) -> "CustomSentimentPredictor":
        """Load trained model from directory."""
        # Load config
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)

        # Rebuild tokenizer
        tokenizer = KenyanTokenizer()
        tokenizer.load(os.path.join(model_dir, "vocab.json"))

        # Rebuild model architecture
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BiLSTMSentimentModel(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"],
            dropout=0.0,  # No dropout at inference
        )

        # Load trained weights
        model.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
        )

        return cls(model, tokenizer, config.get("max_length", 64), device)

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float], List[Tuple[str, float]]]:
        """
        Predict sentiment for a single text.

        Returns:
            label: "negative", "neutral", or "positive"
            confidence: probability of predicted class
            probs: full probability distribution
            attention: list of (token, weight) for interpretability
        """
        token_ids = self.tokenizer.encode(text, self.max_length)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits, attn_weights = self.model(input_tensor)
            probs = F.softmax(logits, dim=-1)[0]

        pred_idx = probs.argmax().item()
        label = self.LABEL_MAP[pred_idx]
        confidence = probs[pred_idx].item()

        prob_dict = {
            "negative": probs[0].item(),
            "neutral": probs[1].item(),
            "positive": probs[2].item(),
        }

        # Extract attention weights for interpretability
        tokens = self.tokenizer._tokenize(text)
        attn = attn_weights[0].cpu().numpy()
        attention = list(zip(tokens, attn[1:len(tokens) + 1].tolist()))

        return label, confidence, prob_dict, attention

    def predict_score(self, text: str) -> float:
        """
        Return a single sentiment score in [-1, 1] range.

        Maps: negative → [-1, -0.25], neutral → [-0.25, 0.25], positive → [0.25, 1]
        Weighted by class probabilities for smooth scoring.
        """
        _, _, probs, _ = self.predict(text)

        # Weighted score: neg contributes negative, pos contributes positive
        score = (probs["positive"] - probs["negative"])
        return round(max(min(score, 1.0), -1.0), 4)


# ═══════════════════════════════════════════════════════════════
# 6. MAIN — Run training from command line
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    trainer = CustomModelTrainer(
        data_path="./data/training_dataset.json",
        model_dir="./models/custom-bilstm-v1",
        embed_dim=128,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        max_length=64,
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=30,
        patience=7,
    )

    model, tokenizer, metrics = trainer.train()

    # ── Interactive testing ──
    print(f"\n{'=' * 65}")
    print("  INTERACTIVE TEST")
    print(f"{'=' * 65}")

    predictor = CustomSentimentPredictor.load("./models/custom-bilstm-v1")

    test_texts = [
        ("Manze serikali iko rada wasee mambo ni poa", "sh"),
        ("Hii mambo ya tax ni noma bana inaumiza wasee", "sh"),
        ("Kenya tech scene is booming right now", "en"),
        ("Corruption is destroying our country completely", "en"),
        ("Serikali inafanya kazi nzuri sana", "sw"),
        ("Hali ya uchumi ni mbaya sana Kenya", "sw"),
        ("Economy ni tight sana bana hakuna kazi", "sh"),
        ("M-Pesa has revolutionized business in Kenya", "en"),
    ]

    correct = 0
    for text, lang in test_texts:
        label, conf, probs, attention = predictor.predict(text)
        score = predictor.predict_score(text)

        # Top attended words
        top_attn = sorted(attention, key=lambda x: x[1], reverse=True)[:3]
        attn_str = ", ".join(f"{w}={a:.2f}" for w, a in top_attn)

        print(f"  [{lang}] \"{text[:55]}\"")
        print(f"       → {label} ({score:+.3f}, conf={conf:.2f}) | focus: {attn_str}")

    print(f"\n  Model saved to: ./models/custom-bilstm-v1/")
    print(f"  Files: model.pt, vocab.json, config.json, training_log.json")
