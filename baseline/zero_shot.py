"""Zero-shot baseline: the full label list is given in the prompt, and only
the canonical label is parsed out of the response (build plan, Day 1).

Usage: python -m baseline.zero_shot --base-config configs/base.yaml
"""
from __future__ import annotations

import argparse

from src.data import get_label_names, load_banking77
from src.metrics import compute_classification_metrics
from src.model import generate, load_quantized_base_model
from src.prompts import build_prompt, parse_label
from src.utils import load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--out", default="outputs/metrics/zero_shot.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config)
    set_seed(config["seed"])

    _, test_raw = load_banking77(config["dataset"]["hf_path"])
    labels = get_label_names(test_raw)

    model, tokenizer = load_quantized_base_model(
        config["model"]["base_model"], config["model"]["compute_dtype"]
    )

    predictions = []
    for text in test_raw["text"]:
        prompt = build_prompt(text, labels)
        raw_output = generate(model, tokenizer, prompt, config["model"]["max_new_tokens"])
        predictions.append(parse_label(raw_output, labels))

    true_labels = [labels[i] for i in test_raw["label"]]
    metrics = compute_classification_metrics(true_labels, predictions, labels)
    save_json({"baseline": "zero_shot", **metrics}, args.out)
    print(f"Zero-shot macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
