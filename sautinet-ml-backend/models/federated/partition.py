"""
SentiKenya Dataset Partitioner
================================
Splits the training dataset into non-IID partitions for 3 regional nodes.

Non-IID strategy (realistic Kenyan regional language distributions):
  - Nairobi  → heavy Sheng, moderate English, some Swahili
  - Mombasa  → heavy Swahili, moderate English, little Sheng
  - Kisumu   → balanced mix, leaning Swahili

This simulates real-world conditions where each region's social media
has different language distributions.
"""

import json
import random
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger("sentikenya.federated.partition")

# Regional language weights (must sum to 1.0 per region)
REGION_LANG_WEIGHTS = {
    "nairobi": {"sh": 0.50, "en": 0.35, "sw": 0.15},
    "mombasa": {"sw": 0.55, "en": 0.30, "sh": 0.15},
    "kisumu":  {"sw": 0.40, "en": 0.35, "sh": 0.25},
}


def partition_dataset(
    dataset_path: str = "./data/training_dataset.json",
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Partition the dataset into 3 non-IID regional splits.

    Returns:
        {
            "nairobi": {"texts": [...], "labels": [...], "langs": [...]},
            "mombasa": {"texts": [...], "labels": [...], "langs": [...]},
            "kisumu":  {"texts": [...], "labels": [...], "langs": [...]},
        }
    """
    random.seed(seed)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    samples = data["data"]
    random.shuffle(samples)

    # Group samples by language
    by_lang: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        by_lang[s["lang"]].append(s)

    logger.info(f"Dataset: {len(samples)} total → " +
                ", ".join(f"{lang}: {len(v)}" for lang, v in by_lang.items()))

    regions = list(REGION_LANG_WEIGHTS.keys())
    partitions = {r: {"texts": [], "labels": [], "langs": []} for r in regions}

    # Distribute each language group proportionally to regional weights
    for lang, lang_samples in by_lang.items():
        random.shuffle(lang_samples)

        # Compute how many samples each region gets for this language
        weights = [REGION_LANG_WEIGHTS[r].get(lang, 0.1) for r in regions]
        total_w = sum(weights)
        fracs = [w / total_w for w in weights]

        idx = 0
        for i, region in enumerate(regions):
            if i == len(regions) - 1:
                # Last region gets the remainder
                chunk = lang_samples[idx:]
            else:
                n_samples = max(1, int(len(lang_samples) * fracs[i]))
                chunk = lang_samples[idx:idx + n_samples]
                idx += n_samples

            for s in chunk:
                partitions[region]["texts"].append(s["text"])
                partitions[region]["labels"].append(s["label"])
                partitions[region]["langs"].append(s["lang"])

    # Log partition stats
    for region, part in partitions.items():
        lang_dist = defaultdict(int)
        label_dist = defaultdict(int)
        for lang in part["langs"]:
            lang_dist[lang] += 1
        for lab in part["labels"]:
            label_dist[lab] += 1
        logger.info(
            f"  {region:>10}: {len(part['texts'])} samples | "
            f"langs={dict(lang_dist)} | labels={dict(label_dist)}"
        )

    return partitions


def create_global_test_set(
    dataset_path: str = "./data/training_dataset.json",
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict, List]:
    """
    Create a balanced global test set held out from all nodes.

    Returns:
        (test_data, remaining_samples)
    """
    random.seed(seed)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    samples = data["data"]
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - test_ratio))
    remaining = samples[:split_idx]
    test_samples = samples[split_idx:]

    test_data = {
        "texts": [s["text"] for s in test_samples],
        "labels": [s["label"] for s in test_samples],
        "langs": [s["lang"] for s in test_samples],
    }

    logger.info(f"Global test set: {len(test_samples)} samples held out")
    return test_data, remaining


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    partitions = partition_dataset()
    for region, p in partitions.items():
        print(f"{region}: {len(p['texts'])} samples")
