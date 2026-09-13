"""Unified benchmark script -- one command, one JSON (build plan section 7).

Runs a model (base + optional adapter) over the frozen test split exactly
once, computing the full metric suite, latency percentiles, and peak GPU
memory, then saves benchmark.json plus a confusion matrix / error CSV for
error analysis.

Usage:
    python -m src.evaluate --base-config configs/base.yaml \
        --adapter-path outputs/best --run-id qlora_r16_lr2e-4_seed42
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.data import get_label_names, load_banking77
from src.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_latency_percentiles,
    top_confusion_pairs,
)
from src.model import count_trainable_parameters, generate, load_model_with_adapter
from src.prompts import build_prompt, parse_label
from src.utils import load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--adapter-path", default="outputs/best")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--num-warmup", type=int, default=3)
    return parser.parse_args()


def run_inference(model, tokenizer, texts: list[str], labels: list[str], max_new_tokens: int):
    """Run generation over every text, returning parsed labels and per-call latency."""
    import torch

    predictions: list[str | None] = []
    latencies_ms: list[float] = []

    for text in texts:
        prompt = build_prompt(text, labels)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        raw_output = generate(model, tokenizer, prompt, max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        predictions.append(parse_label(raw_output, labels))

    return predictions, latencies_ms


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config)
    set_seed(config["seed"])

    _, test_raw = load_banking77(config["dataset"]["hf_path"])
    labels = get_label_names(test_raw)

    model, tokenizer = load_model_with_adapter(config["model"]["base_model"], args.adapter_path)
    trainable_params, total_params = count_trainable_parameters(model)

    texts = test_raw["text"]
    true_labels = [labels[i] for i in test_raw["label"]]

    import torch

    # Warm-up runs so the first (slow, JIT/cache-cold) calls don't skew p50/p95.
    if args.num_warmup:
        run_inference(model, tokenizer, texts[: args.num_warmup], labels, config["model"]["max_new_tokens"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    predictions, latencies_ms = run_inference(
        model, tokenizer, texts, labels, config["model"]["max_new_tokens"]
    )

    peak_gpu_memory_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    metrics = compute_classification_metrics(true_labels, predictions, labels)
    latency = compute_latency_percentiles(latencies_ms)
    confusion = compute_confusion_matrix(true_labels, [p or "__INVALID__" for p in predictions], labels)
    top_pairs = top_confusion_pairs(true_labels, [p or "__INVALID__" for p in predictions], labels)

    benchmark = {
        "run_id": args.run_id,
        "model": config["model"]["base_model"],
        "dataset": config["dataset"]["name"],
        "split": "test",
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "invalid_output_rate": metrics["invalid_output_rate"],
        "latency_ms": latency,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "config": config,
    }
    save_json(benchmark, config["paths"]["benchmark_out"])
    save_json(metrics["per_class"], "outputs/metrics/per_class_metrics.json")
    save_json(top_pairs, "outputs/metrics/top_confusion_pairs.json")
    save_json(confusion.tolist(), "outputs/metrics/confusion_matrix.json")

    error_rows_path = Path("outputs/predictions/errors.csv")
    error_rows_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_rows_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "true_label", "predicted_label"])
        for text, true_label, pred in zip(texts, true_labels, predictions):
            if pred != true_label:
                writer.writerow([text, true_label, pred or "<invalid>"])

    print(f"Saved benchmark to {config['paths']['benchmark_out']}")


if __name__ == "__main__":
    main()
