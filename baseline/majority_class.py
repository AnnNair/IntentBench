"""Majority-class baseline: predict the single most frequent training label
for every test example. The cheapest possible floor -- makes the fine-tuning
gain meaningful by comparison.

Usage: python -m baseline.majority_class --base-config configs/base.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter

from src.data import get_label_names, load_banking77
from src.metrics import compute_classification_metrics
from src.utils import load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--out", default="outputs/metrics/majority_class.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config)
    set_seed(config["seed"])

    train_raw, test_raw = load_banking77(config["dataset"]["hf_path"])
    labels = get_label_names(test_raw)

    majority_label_idx, _ = Counter(train_raw["label"]).most_common(1)[0]
    majority_label = labels[majority_label_idx]

    true_labels = [labels[i] for i in test_raw["label"]]
    predictions = [majority_label] * len(true_labels)

    metrics = compute_classification_metrics(true_labels, predictions, labels)
    save_json({"baseline": "majority_class", "majority_label": majority_label, **metrics}, args.out)
    print(f"Majority-class macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
