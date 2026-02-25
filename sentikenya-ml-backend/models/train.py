"""
SentiKenya Fine-Tuning Pipeline (v2 — sprint-compliant)
=========================================================
Fine-tunes XLM-RoBERTa on labeled Kenyan sentiment data (EN/SW/SH).

Changes from v1:
  ✓ 80 / 10 / 10  train / validation / test split (was 85/15)
  ✓ Early stopping with configurable patience (default 2 epochs)
  ✓ 5 epochs default (was 8) — early stopping may halt sooner
  ✓ Checkpoint path: models/checkpoints/  (was models/sentikenya-v1/)
  ✓ Final evaluation on held-out TEST set
  ✓ Per-class F1 and per-language metrics logged

Key design decisions (unchanged):
  - Freeze lower transformer layers (0-8) to preserve multilingual knowledge
  - Only train top 3 layers + classifier head for domain adaptation
  - Weighted CE loss for class imbalance
  - Evaluate per-language to track Sheng improvement specifically

Usage:
    python models/train.py
    python models/train.py --epochs 5 --lr 2e-5
"""

import json
import logging
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("sentikenya.training")


# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Fine-tuning hyperparameters — sprint-compliant defaults."""
    # Model
    base_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    model_cache_dir: str = "./model_cache"
    output_dir: str = "./models/checkpoints"   # ← fixed path
    num_labels: int = 3  # negative=0, neutral=1, positive=2

    # Training
    epochs: int = 5          # ← 3-5 per spec
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 128

    # Early stopping
    early_stopping_patience: int = 2   # ← NEW: stop if no F1 improvement for N epochs

    # Layer freezing
    freeze_layers: int = 9  # freeze 0-8, train 9-11 + head

    # Data split: 80 / 10 / 10
    dataset_path: str = "./data/training_dataset.json"
    train_ratio: float = 0.80    # ← was 0.85
    val_ratio: float = 0.10      # ← NEW explicit
    test_ratio: float = 0.10     # ← NEW
    seed: int = 42

    use_class_weights: bool = True
    save_best_model: bool = True


# ═══════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════

class KenyanSentimentDataset(Dataset):
    def __init__(self, texts, labels, langs, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.langs = langs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "lang": self.langs[idx],
        }


def load_dataset(config: TrainingConfig):
    """Load dataset with 80/10/10 split."""
    with open(config.dataset_path, "r") as f:
        data = json.load(f)

    samples = data["data"]
    random.seed(config.seed)
    random.shuffle(samples)

    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    langs = [s["lang"] for s in samples]

    n = len(texts)
    train_end = int(n * config.train_ratio)
    val_end = train_end + int(n * config.val_ratio)

    train_data = (texts[:train_end], labels[:train_end], langs[:train_end])
    val_data = (texts[train_end:val_end], labels[train_end:val_end], langs[train_end:val_end])
    test_data = (texts[val_end:], labels[val_end:], langs[val_end:])

    logger.info(f"Dataset: {n} total → {train_end} train / {val_end - train_end} val / {n - val_end} test")

    for name, (t, l, la) in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        lang_dist = {}
        for lang in la:
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        logger.info(f"  {name}: {len(t)} samples, langs={lang_dist}")

    return train_data, val_data, test_data, data.get("metadata", {})


def compute_class_weights(labels: List[int], num_classes: int = 3) -> torch.Tensor:
    counts = [0] * num_classes
    for l in labels:
        counts[l] += 1
    total = len(labels)
    weights = [total / (num_classes * max(c, 1)) for c in counts]
    w = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class weights: neg={w[0]:.3f}, neu={w[1]:.3f}, pos={w[2]:.3f}")
    return w


# ═══════════════════════════════════════════════════════
# Model Setup
# ═══════════════════════════════════════════════════════

def setup_model(config: TrainingConfig):
    logger.info(f"Loading base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, cache_dir=config.model_cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, cache_dir=config.model_cache_dir, num_labels=config.num_labels,
    )

    frozen = 0
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
        frozen += param.numel()
    for i in range(config.freeze_layers):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False
            frozen += param.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = frozen + trainable
    logger.info(f"Params: {total:,} total, {trainable:,} trainable ({trainable/total*100:.1f}%), {frozen:,} frozen")
    return model, tokenizer


# ═══════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════

def evaluate(model, dataloader, device) -> Dict:
    model.eval()
    all_preds, all_labels, all_langs = [], [], []
    total_loss = 0.0
    n_batches = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            total_loss += loss_fn(out.logits, labels).item()
            n_batches += 1
            preds = torch.argmax(out.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_langs.extend(batch["lang"])

    label_names = ["negative", "neutral", "positive"]
    results = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "f1_weighted": round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=label_names, output_dict=True, zero_division=0,
        ),
    }

    lang_metrics = {}
    for lang in set(all_langs):
        lp = [p for p, l in zip(all_preds, all_langs) if l == lang]
        ll = [lb for lb, l in zip(all_labels, all_langs) if l == lang]
        if ll:
            lang_metrics[lang] = {
                "accuracy": round(accuracy_score(ll, lp), 4),
                "f1_macro": round(f1_score(ll, lp, average="macro", zero_division=0), 4),
                "samples": len(ll),
            }
    results["per_language"] = lang_metrics
    return results


# ═══════════════════════════════════════════════════════
# Training Loop (with early stopping)
# ═══════════════════════════════════════════════════════

def train(config: TrainingConfig = None):
    config = config or TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data — 80/10/10
    (tr_t, tr_l, tr_la), (va_t, va_l, va_la), (te_t, te_l, te_la), meta = load_dataset(config)

    model, tokenizer = setup_model(config)
    model.to(device)

    train_ds = KenyanSentimentDataset(tr_t, tr_l, tr_la, tokenizer, config.max_length)
    val_ds = KenyanSentimentDataset(va_t, va_l, va_la, tokenizer, config.max_length)
    test_ds = KenyanSentimentDataset(te_t, te_l, te_la, tokenizer, config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    if config.use_class_weights:
        cw = compute_class_weights(tr_l).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=cw)
    else:
        loss_fn = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # Baseline
    baseline = evaluate(model, val_loader, device)
    logger.info(f"Baseline → acc={baseline['accuracy']}, f1={baseline['f1_macro']}")

    best_f1 = baseline["f1_macro"]
    best_epoch = -1
    patience_counter = 0
    training_log = []
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  TRAINING: {config.epochs} epochs, early_stop_patience={config.early_stopping_patience}")
    print(f"{'═'*60}\n")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(out.logits, dim=-1)
            epoch_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        dt = time.time() - t0
        train_acc = correct / max(total, 1)
        avg_loss = epoch_loss / len(train_loader)

        val_res = evaluate(model, val_loader, device)
        val_f1 = val_res["f1_macro"]

        lang_str = " | ".join(
            f"{la}: {m['f1_macro']:.3f}" for la, m in sorted(val_res["per_language"].items())
        )
        improved = ""
        if val_f1 > best_f1:
            improved = " ★ BEST"
            best_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            if config.save_best_model:
                save_path = os.path.join(config.output_dir, "best")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1:>2}/{config.epochs} │ loss={avg_loss:.4f} │ "
              f"train_acc={train_acc:.3f} │ val_f1={val_f1:.4f}{improved} │ "
              f"{lang_str} │ {dt:.1f}s")

        training_log.append({
            "epoch": epoch + 1, "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4), "val_f1": val_f1,
            "val_acc": val_res["accuracy"], "per_language": val_res["per_language"],
        })

        # ── Early stopping ──
        if patience_counter >= config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {config.early_stopping_patience} epochs)")
            break

    # ── Final evaluation on TEST set ──
    print(f"\n{'═'*60}")
    print(f"  FINAL EVALUATION ON HELD-OUT TEST SET")
    print(f"{'═'*60}")

    if config.save_best_model and best_epoch > 0:
        best_path = os.path.join(config.output_dir, "best")
        model = AutoModelForSequenceClassification.from_pretrained(best_path)
        model.to(device)

    test_res = evaluate(model, test_loader, device)
    print(f"  Test Accuracy:    {test_res['accuracy']:.4f}")
    print(f"  Test F1 (macro):  {test_res['f1_macro']:.4f}")
    if test_res["per_language"]:
        for la, m in sorted(test_res["per_language"].items()):
            print(f"    {la}: acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  n={m['samples']}")

    # Save final + log
    final_path = os.path.join(config.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    log_path = os.path.join(config.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {k: str(v) for k, v in config.__dict__.items()},
            "baseline_f1": baseline["f1_macro"],
            "best_epoch": best_epoch, "best_f1": best_f1,
            "test_results": {k: v for k, v in test_res.items() if k != "classification_report"},
            "epochs": training_log,
        }, f, indent=2)
    logger.info(f"Training log → {log_path}")

    return model, tokenizer, test_res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        batch_size=args.batch,
    )
    train(config)
