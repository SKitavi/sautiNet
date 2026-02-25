"""
SentiKenya Federated Averaging (FedAvg)
=========================================
Implements the FedAvg algorithm (McMahan et al., 2017) for aggregating
model updates from distributed edge nodes.

FedAvg:
  1. Server sends global model to all nodes
  2. Each node trains locally on its partition for E local epochs
  3. Server collects updated model weights from all nodes
  4. Server averages the weights (weighted by node dataset size)
  5. Repeat for R rounds
"""

import copy
import logging
from typing import Dict, List, OrderedDict

import torch

logger = logging.getLogger("sentikenya.federated.fedavg")


def federated_average(
    node_state_dicts: List[OrderedDict],
    node_sample_counts: List[int],
) -> OrderedDict:
    """
    Federated Averaging: compute weighted average of model parameters.

    Args:
        node_state_dicts: List of state_dict from each node after local training.
        node_sample_counts: Number of training samples at each node (for weighting).

    Returns:
        Averaged state_dict for the global model.
    """
    if not node_state_dicts:
        raise ValueError("No node state dicts provided")

    total_samples = sum(node_sample_counts)
    weights = [n / total_samples for n in node_sample_counts]

    logger.info(
        f"FedAvg aggregation: {len(node_state_dicts)} nodes, "
        f"samples={node_sample_counts}, weights=[{', '.join(f'{w:.3f}' for w in weights)}]"
    )

    # Initialize averaged dict with zeros
    avg_state = copy.deepcopy(node_state_dicts[0])
    for key in avg_state:
        avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)

    # Weighted sum
    for state_dict, weight in zip(node_state_dicts, weights):
        for key in avg_state:
            avg_state[key] += state_dict[key].float() * weight

    return avg_state


def compute_model_divergence(
    global_state: OrderedDict,
    node_states: List[OrderedDict],
) -> Dict[str, float]:
    """
    Compute L2 divergence between global model and each node's model.

    Useful for monitoring how much nodes drift from the global model
    during local training — higher drift = more non-IID effects.
    """
    divergences = {}
    for i, node_state in enumerate(node_states):
        total_diff = 0.0
        total_norm = 0.0
        for key in global_state:
            diff = (global_state[key].float() - node_state[key].float()).norm().item()
            norm = global_state[key].float().norm().item()
            total_diff += diff ** 2
            total_norm += norm ** 2

        divergences[f"node_{i}"] = round((total_diff / max(total_norm, 1e-8)) ** 0.5, 6)

    return divergences
