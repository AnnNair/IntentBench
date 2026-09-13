"""Confusion-matrix and metric plots feeding the README benchmark table."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_confusion_matrix(confusion: np.ndarray, labels: list[str], out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=4)
    ax.set_yticklabels(labels, fontsize=4)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix -- Banking77 test set")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
