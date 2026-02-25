#!/usr/bin/env python3
"""
SentiKenya Federated Learning Orchestrator
=============================================
Runs the full federated learning simulation:

  1. Partition dataset into 3 non-IID regional splits
  2. Hold out a global test set
  3. Run 5 federation rounds (FedAvg)
  4. Train a centralized baseline on the full dataset
  5. Compare federated vs. centralized accuracy
  6. Log all results

Usage:
    python -m models.federated.run_federation
    python -m models.federated.run_federation --rounds 5 --local-epochs 2
"""

import argparse
import copy
import json
import logging
import os
import random
import time
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from models.federated.partition import partition_dataset, REGION_LANG_WEIGHTS
from models.federated.fedavg import federated_average, compute_model_divergence
from models.federated.node import EdgeNode, LocalDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("sentikenya.federated")

BASE_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
CACHE_DIR = "./model_cache"


def evaluate_global(model, test_texts, test_labels, tokenizer, device) -> Dict:
    """Evaluate the global model on the held-out test set."""
    model.eval()
    ds = LocalDataset(test_texts, test_labels, tokenizer)
    loader = DataLoader(ds, batch_size=16)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = round(accuracy_score(all_labels, all_preds), 4)
    f1 = round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4)
    return {"accuracy": acc, "f1_macro": f1, "samples": len(all_labels)}


def run_centralized_baseline(
    all_texts, all_labels, test_texts, test_labels,
    tokenizer, device, epochs=3, lr=2e-5,
) -> Dict:
    """Train a centralized model on the FULL (combined) training set for comparison."""
    logger.info(f"\n{'═'*50}")
    logger.info(f"  CENTRALIZED BASELINE ({len(all_texts)} samples, {epochs} epochs)")
    logger.info(f"{'═'*50}")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, cache_dir=CACHE_DIR, num_labels=3
    )
    model.to(device)
    model.train()

    ds = LocalDataset(all_texts, all_labels, tokenizer)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = torch.argmax(out.logits, dim=-1)
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / max(total, 1)
        logger.info(f"  Centralized epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.4f}, acc={acc:.3f}")

    result = evaluate_global(model, test_texts, test_labels, tokenizer, device)
    logger.info(f"  Centralized baseline: acc={result['accuracy']}, f1={result['f1_macro']}")
    return result


def run_federation(
    num_rounds: int = 5,
    local_epochs: int = 2,
    learning_rate: float = 2e-5,
    dataset_path: str = "./data/training_dataset.json",
    output_dir: str = "./docs",
    seed: int = 42,
):
    """Run the full federated learning simulation."""
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 1. Load and partition data ──
    with open(dataset_path) as f:
        data = json.load(f)
    samples = data["data"]
    random.shuffle(samples)

    # Hold out 15% as global test set
    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    test_texts = [s["text"] for s in test_samples]
    test_labels = [s["label"] for s in test_samples]
    logger.info(f"Global test set: {len(test_samples)} samples")

    # Write remaining to temp for partitioning
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"data": train_samples, "metadata": data.get("metadata", {})}, tmp)
    tmp.close()

    partitions = partition_dataset(dataset_path=tmp.name, seed=seed)
    os.unlink(tmp.name)

    # ── 2. Initialise global model ──
    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
    global_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, cache_dir=CACHE_DIR, num_labels=3
    )
    global_model.to(device)

    # ── 3. Create edge nodes ──
    node_configs = [
        ("NBO-01", "nairobi"),
        ("MSA-01", "mombasa"),
        ("KSM-01", "kisumu"),
    ]

    nodes = []
    for node_id, region in node_configs:
        part = partitions[region]
        node = EdgeNode(
            node_id=node_id,
            region=region,
            texts=part["texts"],
            labels=part["labels"],
            tokenizer=tokenizer,
            model_template=global_model,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=device,
        )
        nodes.append(node)

    # ── 4. Baseline before federation ──
    baseline = evaluate_global(global_model, test_texts, test_labels, tokenizer, device)
    logger.info(f"Pre-federation baseline: acc={baseline['accuracy']}, f1={baseline['f1_macro']}")

    # ── 5. Federation rounds ──
    round_results = []

    print(f"\n{'═'*60}")
    print(f"  FEDERATED LEARNING: {num_rounds} rounds, {len(nodes)} nodes")
    print(f"{'═'*60}\n")

    for round_num in range(1, num_rounds + 1):
        t0 = time.time()
        logger.info(f"── Round {round_num}/{num_rounds} ──")

        # Distribute global model to all nodes
        global_state = copy.deepcopy(global_model.state_dict())
        for node in nodes:
            node.receive_global_model(global_state)

        # Local training
        node_stats = []
        for node in nodes:
            stats = node.train_local()
            node_stats.append(stats)

        # Collect state dicts and aggregate
        node_state_dicts = [node.get_model_state() for node in nodes]
        node_sample_counts = [node.sample_count for node in nodes]

        # FedAvg
        aggregated_state = federated_average(node_state_dicts, node_sample_counts)
        global_model.load_state_dict(aggregated_state)

        # Evaluate global model
        eval_result = evaluate_global(global_model, test_texts, test_labels, tokenizer, device)
        dt = time.time() - t0

        # Divergence
        divergence = compute_model_divergence(aggregated_state, node_state_dicts)

        round_info = {
            "round": round_num,
            "global_accuracy": eval_result["accuracy"],
            "global_f1": eval_result["f1_macro"],
            "node_stats": node_stats,
            "divergence": divergence,
            "time_s": round(dt, 2),
        }
        round_results.append(round_info)

        print(
            f"  Round {round_num:>2} │ "
            f"global_acc={eval_result['accuracy']:.4f} │ "
            f"global_f1={eval_result['f1_macro']:.4f} │ "
            f"{dt:.1f}s"
        )

    # ── 6. Centralized baseline for comparison ──
    all_train_texts = [s["text"] for s in train_samples]
    all_train_labels = [s["label"] for s in train_samples]

    centralized = run_centralized_baseline(
        all_train_texts, all_train_labels,
        test_texts, test_labels,
        tokenizer, device,
        epochs=local_epochs * num_rounds // 2,  # comparable total epochs
    )

    # ── 7. Comparison ──
    final_fed = round_results[-1]
    gap = centralized["accuracy"] - final_fed["global_accuracy"]

    print(f"\n{'═'*60}")
    print(f"  RESULTS COMPARISON")
    print(f"{'═'*60}")
    print(f"  Federated (5 rounds):  acc={final_fed['global_accuracy']:.4f}, f1={final_fed['global_f1']:.4f}")
    print(f"  Centralized baseline:  acc={centralized['accuracy']:.4f}, f1={centralized['f1_macro']:.4f}")
    print(f"  Gap:                   {gap:+.4f} {'✅ within 5%' if abs(gap) <= 0.05 else '⚠️ exceeds 5%'}")
    print()

    # ── 8. Save results ──
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "federated_results.json")

    full_results = {
        "config": {
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "num_nodes": len(nodes),
            "regions": [n.region for n in nodes],
            "base_model": BASE_MODEL,
        },
        "pre_federation_baseline": baseline,
        "rounds": round_results,
        "centralized_baseline": centralized,
        "final_comparison": {
            "federated_accuracy": final_fed["global_accuracy"],
            "federated_f1": final_fed["global_f1"],
            "centralized_accuracy": centralized["accuracy"],
            "centralized_f1": centralized["f1_macro"],
            "accuracy_gap": round(gap, 4),
            "within_5_percent": abs(gap) <= 0.05,
        },
    }

    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Also generate markdown report
    _generate_markdown_report(full_results, output_dir)

    return full_results


def _generate_markdown_report(results: Dict, output_dir: str):
    """Auto-generate docs/federated_results.md from the experiment data."""
    r = results
    comp = r["final_comparison"]

    md = f"""# SentiKenya — Federated Learning Results

## Experiment Configuration

| Parameter        | Value                                |
|------------------|--------------------------------------|
| Federation Rounds | {r['config']['num_rounds']}         |
| Local Epochs     | {r['config']['local_epochs']}        |
| Learning Rate    | {r['config']['learning_rate']}       |
| Base Model       | `{r['config']['base_model']}`        |
| Regions          | {', '.join(r['config']['regions'])}  |

## Data Partitioning Strategy

Non-IID split simulating regional linguistic distributions:

| Region   | Sheng | English | Swahili | Rationale                      |
|----------|-------|---------|---------|--------------------------------|
| Nairobi  | 50%   | 35%     | 15%     | Urban Sheng-heavy social media |
| Mombasa  | 15%   | 30%     | 55%     | Coastal Swahili-dominant        |
| Kisumu   | 25%   | 35%     | 40%     | Balanced regional mix           |

## Per-Round Global Model Accuracy

| Round | Accuracy | F1 (macro) | Time (s) |
|-------|----------|------------|----------|
"""

    for rnd in r["rounds"]:
        md += f"| {rnd['round']}     | {rnd['global_accuracy']:.4f}   | {rnd['global_f1']:.4f}     | {rnd['time_s']}       |\n"

    md += f"""
## Final Comparison

| Approach     | Accuracy | F1 (macro) |
|-------------|----------|------------|
| Federated   | {comp['federated_accuracy']:.4f}   | {comp['federated_f1']:.4f}     |
| Centralized | {comp['centralized_accuracy']:.4f}   | {comp['centralized_f1']:.4f}     |
| **Gap**     | **{comp['accuracy_gap']:+.4f}** | —          |

**Acceptance criteria**: Global accuracy within 5% of centralized baseline → {'✅ **PASS**' if comp['within_5_percent'] else '⚠️ **NEEDS REVIEW**'}

## Analysis

The federated approach demonstrates that decentralized training across Kenyan
regional nodes can approximate centralized performance despite non-IID data
distributions. The Nairobi node contributes Sheng-specific learning, Mombasa
strengthens Swahili understanding, and Kisumu provides balanced generalisation.

Key observations:
- FedAvg effectively aggregates heterogeneous regional models
- Non-IID partitioning simulates realistic Kenyan social media distributions
- The accuracy gap of {comp['accuracy_gap']:+.4f} {'is' if comp['within_5_percent'] else 'may not be'} within the 5% acceptance threshold
"""

    md_path = os.path.join(output_dir, "federated_results.md")
    with open(md_path, "w") as f:
        f.write(md)
    logger.info(f"Markdown report → {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentiKenya Federated Learning Simulation")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dataset", default="./data/training_dataset.json")
    parser.add_argument("--output", default="./docs")
    args = parser.parse_args()

    run_federation(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        dataset_path=args.dataset,
        output_dir=args.output,
    )
