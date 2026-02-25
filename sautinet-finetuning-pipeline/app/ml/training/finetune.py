"""
SentiKenya Fine-Tuning Pipeline
=================================
Fine-tunes XLM-RoBERTa on labeled Kenyan sentiment data (EN/SW/SH).

Key design decisions:
- Freeze lower transformer layers (0-8) to preserve multilingual knowledge
- Only train top 3 layers + classifier head for domain adaptation
- Use weighted loss to handle class imbalance
- Evaluate per-language to track Sheng improvement specifically
- Export both full model and ONNX for production inference
"""

import json
import logging
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("sentikenya.training")


# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Fine-tuning hyperparameters."""
    # Model
    base_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    model_cache_dir: str = "./model_cache"
    output_dir: str = "./models/sentikenya-v1"
    num_labels: int = 3  # negative=0, neutral=1, positive=2

    # Training
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 128

    # Layer freezing: freeze layers 0 through freeze_layers-1
    freeze_layers: int = 9  # Freeze 0-8, train 9-11 + head

    # Data
    dataset_path: str = "./data/training_dataset.json"
    val_split: float = 0.15
    seed: int = 42

    # Class weights for imbalanced data
    use_class_weights: bool = True

    # Evaluation
    eval_every_n_steps: int = 20
    save_best_model: bool = True


# ═══════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════

class KenyanSentimentDataset(Dataset):
    """PyTorch dataset for Kenyan sentiment training data."""

    def __init__(self, texts, labels, langs, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.langs = langs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "lang": self.langs[idx],
        }


def load_dataset(config: TrainingConfig) -> Tuple[List, List, List]:
    """Load and split the training dataset."""
    with open(config.dataset_path, "r") as f:
        data = json.load(f)

    samples = data["data"]
    random.seed(config.seed)
    random.shuffle(samples)

    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    langs = [s["lang"] for s in samples]

    # Stratified-ish split
    split_idx = int(len(texts) * (1 - config.val_split))
    train_data = (texts[:split_idx], labels[:split_idx], langs[:split_idx])
    val_data = (texts[split_idx:], labels[split_idx:], langs[split_idx:])

    logger.info(f"Dataset loaded: {len(texts)} total, {split_idx} train, {len(texts) - split_idx} val")

    # Distribution stats
    for split_name, (t, l, la) in [("Train", train_data), ("Val", val_data)]:
        lang_dist = {}
        label_dist = {0: 0, 1: 0, 2: 0}
        for lang, label in zip(la, l):
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
            label_dist[label] += 1
        logger.info(f"  {split_name}: langs={lang_dist}, labels={label_dist}")

    return train_data, val_data, data.get("metadata", {})


def compute_class_weights(labels: List[int], num_classes: int = 3) -> torch.Tensor:
    """Compute inverse frequency class weights."""
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
    """Load base model and apply layer freezing strategy."""
    logger.info(f"Loading base model: {config.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, cache_dir=config.model_cache_dir
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        cache_dir=config.model_cache_dir,
        num_labels=config.num_labels,
    )

    # Freeze lower layers to preserve multilingual representations
    frozen_params = 0
    trainable_params = 0

    # Freeze embeddings
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
        frozen_params += param.numel()

    # Freeze encoder layers 0 through freeze_layers-1
    for i in range(config.freeze_layers):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False
            frozen_params += param.numel()

    # Count trainable
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()

    total = frozen_params + trainable_params
    logger.info(f"Parameters: {total:,} total, {trainable_params:,} trainable "
                f"({trainable_params/total*100:.1f}%), {frozen_params:,} frozen")
    logger.info(f"Frozen layers: embeddings + encoder[0:{config.freeze_layers}]")
    logger.info(f"Trainable: encoder[{config.freeze_layers}:12] + classifier head")

    return model, tokenizer


# ═══════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════

def evaluate(model, dataloader, device, label_names=None) -> Dict:
    """
    Full evaluation with per-language breakdown.

    Returns overall and per-language metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_langs = []
    total_loss = 0.0
    n_batches = 0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_langs.extend(batch["lang"])

    # Overall metrics
    label_names = label_names or ["negative", "neutral", "positive"]
    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average="macro")
    overall_f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    results = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": round(overall_acc, 4),
        "f1_macro": round(overall_f1, 4),
        "f1_weighted": round(overall_f1_weighted, 4),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=label_names, output_dict=True
        ),
    }

    # Per-language metrics
    lang_metrics = {}
    for lang in set(all_langs):
        lang_preds = [p for p, l in zip(all_preds, all_langs) if l == lang]
        lang_labels = [lb for lb, l in zip(all_labels, all_langs) if l == lang]

        if lang_labels:
            lang_metrics[lang] = {
                "accuracy": round(accuracy_score(lang_labels, lang_preds), 4),
                "f1_macro": round(f1_score(lang_labels, lang_preds, average="macro", zero_division=0), 4),
                "samples": len(lang_labels),
            }

    results["per_language"] = lang_metrics
    return results


def print_eval_results(results: Dict, header: str = "Evaluation"):
    """Pretty-print evaluation results."""
    print(f"\n{'─'*60}")
    print(f"  {header}")
    print(f"{'─'*60}")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print(f"  F1 (weighted): {results['f1_weighted']:.4f}")

    if "per_language" in results:
        print(f"\n  Per-Language Breakdown:")
        print(f"  {'Lang':<8} {'Acc':>8} {'F1':>8} {'Samples':>8}")
        print(f"  {'─'*36}")
        for lang, m in sorted(results["per_language"].items()):
            print(f"  {lang:<8} {m['accuracy']:>8.4f} {m['f1_macro']:>8.4f} {m['samples']:>8}")

    cr = results.get("classification_report", {})
    if cr:
        print(f"\n  Per-Class:")
        print(f"  {'Class':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'─'*48}")
        for cls_name in ["negative", "neutral", "positive"]:
            if cls_name in cr:
                c = cr[cls_name]
                print(f"  {cls_name:<12} {c['precision']:>8.4f} {c['recall']:>8.4f} "
                      f"{c['f1-score']:>8.4f} {c['support']:>8.0f}")
    print()


# ═══════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════

def train(config: TrainingConfig = None):
    """
    Full fine-tuning pipeline.

    1. Load dataset + compute class weights
    2. Load model + freeze layers
    3. Train with weighted CE loss + cosine LR
    4. Evaluate per-language every N steps
    5. Save best model checkpoint
    """
    config = config or TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load data ──
    (train_texts, train_labels, train_langs), \
    (val_texts, val_labels, val_langs), metadata = load_dataset(config)

    # ── Setup model ──
    model, tokenizer = setup_model(config)
    model.to(device)

    # ── Create datasets ──
    train_dataset = KenyanSentimentDataset(
        train_texts, train_labels, train_langs, tokenizer, config.max_length
    )
    val_dataset = KenyanSentimentDataset(
        val_texts, val_labels, val_langs, tokenizer, config.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # ── Loss with class weights ──
    if config.use_class_weights:
        class_weights = compute_class_weights(train_labels).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # ── Optimizer (only trainable params) ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # ── Baseline evaluation ──
    logger.info("Running baseline evaluation (before fine-tuning)...")
    baseline = evaluate(model, val_loader, device)
    print_eval_results(baseline, "BASELINE (Pre-Fine-Tuning)")

    # ── Training ──
    best_f1 = baseline["f1_macro"]
    best_epoch = -1
    global_step = 0
    training_log = []

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  TRAINING STARTED")
    print(f"  Epochs: {config.epochs}, Steps/epoch: {len(train_loader)}, Total: {total_steps}")
    print(f"  LR: {config.learning_rate}, Batch: {config.batch_size}")
    print(f"{'═'*60}\n")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Track metrics
            preds = torch.argmax(outputs.logits, dim=-1)
            epoch_loss += loss.item()
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            global_step += 1

        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_acc = epoch_correct / max(epoch_total, 1)
        avg_loss = epoch_loss / len(train_loader)

        # Evaluate
        val_results = evaluate(model, val_loader, device)
        val_f1 = val_results["f1_macro"]

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(epoch_acc, 4),
            "val_loss": round(val_results["loss"], 4),
            "val_acc": val_results["accuracy"],
            "val_f1_macro": val_f1,
            "per_language": val_results["per_language"],
            "lr": scheduler.get_last_lr()[0],
            "time_s": round(epoch_time, 1),
        }
        training_log.append(log_entry)

        # Print epoch summary
        lang_str = " | ".join(
            f"{lang}: {m['f1_macro']:.3f}"
            for lang, m in sorted(val_results["per_language"].items())
        )
        improved = " ★ BEST" if val_f1 > best_f1 else ""
        print(f"  Epoch {epoch+1:>2}/{config.epochs} │ "
              f"loss={avg_loss:.4f} │ train_acc={epoch_acc:.3f} │ "
              f"val_f1={val_f1:.4f}{improved} │ "
              f"{lang_str} │ {epoch_time:.1f}s")

        # Save best model
        if val_f1 > best_f1 and config.save_best_model:
            best_f1 = val_f1
            best_epoch = epoch + 1
            save_path = os.path.join(config.output_dir, "best")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"  Saved best model (F1={best_f1:.4f}) to {save_path}")

    # ── Final evaluation ──
    print(f"\n{'═'*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best epoch: {best_epoch}, Best val F1: {best_f1:.4f}")
    print(f"{'═'*60}")

    # Load best model for final eval
    if config.save_best_model and best_epoch > 0:
        best_path = os.path.join(config.output_dir, "best")
        model = AutoModelForSequenceClassification.from_pretrained(best_path)
        model.to(device)

    final_results = evaluate(model, val_loader, device)
    print_eval_results(final_results, "FINAL EVALUATION (Best Model)")

    # Save final model
    final_path = os.path.join(config.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training log
    log_path = os.path.join(config.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "config": {k: str(v) for k, v in config.__dict__.items()},
            "baseline": {k: v for k, v in baseline.items() if k != "classification_report"},
            "final": {k: v for k, v in final_results.items() if k != "classification_report"},
            "epochs": training_log,
            "best_epoch": best_epoch,
            "best_f1": best_f1,
        }, f, indent=2)
    logger.info(f"Training log saved to {log_path}")

    return model, tokenizer, final_results


# ═══════════════════════════════════════════════════════
# Interactive Testing
# ═══════════════════════════════════════════════════════

def test_model(model_path: str, test_texts: List[Tuple[str, str]] = None):
    """Load a fine-tuned model and test on sample texts."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    labels = ["negative", "neutral", "positive"]

    if not test_texts:
        test_texts = [
            ("Manze hii serikali iko rada wasee mambo ni poa", "sh"),
            ("Hii mambo ya tax ni noma bana inaumiza wasee", "sh"),
            ("Wasee wa bunge wanajipangia salary kubwa sisi tunapiga hustle", "sh"),
            ("Cheki vile Safaricom wamebadilisha game ya tech Kenya", "sh"),
            ("Serikali inafanya kazi nzuri katika ujenzi wa barabara", "sw"),
            ("Hali ya uchumi ni mbaya sana kwa wananchi", "sw"),
            ("Corruption is destroying our country", "en"),
            ("Kenya's tech ecosystem is growing faster than ever", "en"),
        ]

    print(f"\n{'═'*70}")
    print(f"  MODEL TEST: {model_path}")
    print(f"{'═'*70}")

    for text, lang in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = torch.argmax(probs).item()
        score = probs[2].item() - probs[0].item()

        print(f"\n  [{lang}] \"{text[:60]}...\"")
        print(f"       Prediction: {labels[pred_idx]} (score: {score:+.3f})")
        print(f"       Probs: neg={probs[0]:.3f} neu={probs[1]:.3f} pos={probs[2]:.3f}")


# ═══════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    config = TrainingConfig(
        epochs=8,
        batch_size=16,
        learning_rate=2e-5,
        freeze_layers=9,
    )

    model, tokenizer, results = train(config)

    # Test the fine-tuned model
    test_model(os.path.join(config.output_dir, "best"))
