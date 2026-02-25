"""
SentiKenya Federated Edge Node
================================
Simulates an edge node in the federated learning setup.

Each node:
  - Holds a local partition of the training data
  - Trains the model locally for E epochs per federation round
  - Returns its updated model weights to the central server

Three simulated nodes:
  - Nairobi  (NBO) — heavy Sheng
  - Mombasa  (MSA) — heavy Swahili
  - Kisumu   (KSM) — balanced mix
"""

import copy
import logging
import time
from typing import Dict, List, Optional, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

logger = logging.getLogger("sentikenya.federated.node")


class LocalDataset(Dataset):
    """Lightweight dataset for a node's local partition."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
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
        }


class EdgeNode:
    """
    Simulated federated learning edge node.

    Each node holds a local data partition and performs local training
    when triggered by the federation orchestrator.
    """

    def __init__(
        self,
        node_id: str,
        region: str,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        model_template: AutoModelForSequenceClassification,
        local_epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        device: torch.device = None,
    ):
        self.node_id = node_id
        self.region = region
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or torch.device("cpu")
        self.sample_count = len(texts)

        # Local dataset
        self.dataset = LocalDataset(texts, labels, tokenizer)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Local model (deep copy of global template)
        self.model = copy.deepcopy(model_template)
        self.model.to(self.device)

        logger.info(
            f"EdgeNode '{node_id}' ({region}): {self.sample_count} samples, "
            f"{local_epochs} local epochs"
        )

    def receive_global_model(self, global_state_dict: OrderedDict):
        """Update local model with the aggregated global weights."""
        self.model.load_state_dict(global_state_dict)

    def train_local(self) -> Dict:
        """
        Perform local training for E epochs.

        Returns:
            Dict with training stats (loss, accuracy).
        """
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        t0 = time.time()

        for epoch in range(self.local_epochs):
            for batch in self.loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(outputs.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs.logits, dim=-1)
                total_loss += loss.item()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        dt = time.time() - t0
        stats = {
            "node_id": self.node_id,
            "region": self.region,
            "local_epochs": self.local_epochs,
            "samples": self.sample_count,
            "train_loss": round(total_loss / max(total_samples / self.batch_size, 1), 4),
            "train_acc": round(total_correct / max(total_samples, 1), 4),
            "time_s": round(dt, 2),
        }
        logger.info(
            f"  {self.node_id} ({self.region}) trained: "
            f"loss={stats['train_loss']:.4f}, acc={stats['train_acc']:.3f}, {dt:.1f}s"
        )
        return stats

    def get_model_state(self) -> OrderedDict:
        """Return the current model state dict."""
        return copy.deepcopy(self.model.state_dict())
