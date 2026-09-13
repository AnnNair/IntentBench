"""Load, profile, and split Banking77.

The test split is loaded once, frozen, and must not be touched again until
the single Day-3 evaluation run (see README.md). Splitting logic lives here
so training, baselines, and evaluation all see the exact same split.
"""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils import save_json, set_seed


def load_banking77(hf_path: str = "banking77"):
    """Return the raw (train, test) HF DatasetDict splits for Banking77."""
    from datasets import load_dataset

    ds = load_dataset(hf_path)
    return ds["train"], ds["test"]


def get_label_names(train_split) -> list[str]:
    return train_split.features["label"].names


def make_splits(train_split, val_fraction: float, seed: int):
    """Carve a stratified-ish validation set out of the official train split.

    Uses HF's own train_test_split (shuffled, seeded) rather than a custom
    stratifier -- Banking77 is large enough per class that a seeded random
    split is adequate for this scope.
    """
    set_seed(seed)
    split = train_split.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


def profile_dataset(dataset, label_names: list[str]) -> dict[str, Any]:
    """Compute label balance, text-length, and duplicate stats for one split."""
    texts = dataset["text"]
    labels = dataset["label"]

    label_counts = Counter(labels)
    lengths = [len(t.split()) for t in texts]
    duplicates = len(texts) - len(set(texts))

    return {
        "num_examples": len(texts),
        "num_classes": len(label_names),
        "label_counts": {label_names[i]: label_counts.get(i, 0) for i in range(len(label_names))},
        "min_class_count": min(label_counts.values()) if label_counts else 0,
        "max_class_count": max(label_counts.values()) if label_counts else 0,
        "text_length_words": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": statistics.mean(lengths) if lengths else 0.0,
            "median": statistics.median(lengths) if lengths else 0.0,
        },
        "duplicate_texts": duplicates,
    }


def build_data_profile(config: dict, out_path: str | Path) -> dict[str, Any]:
    train_raw, test_raw = load_banking77(config["dataset"]["hf_path"])
    label_names = get_label_names(train_raw)
    train, val = make_splits(train_raw, config["dataset"]["val_fraction"], config["seed"])

    profile = {
        "train": profile_dataset(train, label_names),
        "val": profile_dataset(val, label_names),
        "test": profile_dataset(test_raw, label_names),
    }
    save_json(profile, out_path)
    return profile
