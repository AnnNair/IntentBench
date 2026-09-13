"""Metric computation -- the deterministic, GPU-free core of the eval harness.

Pure functions over label lists so they're trivially unit-testable (see
tests/test_metrics.py) without loading a model.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    labels: Sequence[str],
) -> dict:
    """Compute accuracy, macro/weighted P/R/F1, per-class P/R, and invalid-output rate.

    y_pred entries that are None (unparseable model output) are treated as
    always-incorrect predictions -- they are counted against accuracy/F1 via
    a placeholder label that cannot match any true label, and separately
    reported as the invalid-output rate.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    n = len(y_true)
    invalid_count = sum(1 for p in y_pred if p is None or p not in labels)
    invalid_output_rate = invalid_count / n if n else 0.0

    sentinel = "__INVALID__"
    y_pred_clean = [p if p in labels else sentinel for p in y_pred]

    all_labels = list(labels)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred_clean)) / n if n else 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred_clean, labels=all_labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_clean, labels=all_labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_clean, labels=all_labels, average="weighted", zero_division=0
    )

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(all_labels)
    }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "invalid_output_rate": float(invalid_output_rate),
        "per_class": per_class,
    }


def compute_confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(labels))


def top_confusion_pairs(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], top_n: int = 20
) -> list[dict]:
    """Return the top-N most frequent (true != predicted) label pairs."""
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    pairs = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": true_label, "predicted": pred_label, "count": int(cm[i, j])})
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_n]


def compute_latency_percentiles(latencies_ms: Sequence[float]) -> dict:
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0}
    arr = np.array(latencies_ms)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}
