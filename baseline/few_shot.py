"""Few-shot baseline with a fixed example set, reused across every future
comparison (build plan, Day 1) -- never resampled per run.

Usage: python -m baseline.few_shot --base-config configs/base.yaml
"""
from __future__ import annotations

import argparse
import random

from src.data import get_label_names, load_banking77
from src.metrics import compute_classification_metrics
from src.model import generate, load_quantized_base_model
from src.prompts import build_prompt, parse_label
from src.utils import load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--out", default="outputs/metrics/few_shot.json")
    return parser.parse_args()


def select_fixed_examples(train_raw, labels: list[str], config: dict) -> list[tuple[str, str]]:
    """Deterministically pick a fixed set of (text, label) examples.

    Uses a dedicated example_seed (independent of the run seed) so the fixed
    example set never changes even if the overall run seed does.
    """
    rng = random.Random(config["few_shot"]["example_seed"])
    indices = list(range(len(train_raw)))
    rng.shuffle(indices)
    chosen = indices[: config["few_shot"]["num_examples"]]
    return [(train_raw[i]["text"], labels[train_raw[i]["label"]]) for i in chosen]


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config)
    set_seed(config["seed"])

    train_raw, test_raw = load_banking77(config["dataset"]["hf_path"])
    labels = get_label_names(test_raw)
    examples = select_fixed_examples(train_raw, labels, config)

    model, tokenizer = load_quantized_base_model(
        config["model"]["base_model"], config["model"]["compute_dtype"]
    )

    predictions = []
    for text in test_raw["text"]:
        prompt = build_prompt(text, labels, examples=examples)
        raw_output = generate(model, tokenizer, prompt, config["model"]["max_new_tokens"])
        predictions.append(parse_label(raw_output, labels))

    true_labels = [labels[i] for i in test_raw["label"]]
    metrics = compute_classification_metrics(true_labels, predictions, labels)
    save_json({"baseline": "few_shot", "fixed_examples": examples, **metrics}, args.out)
    print(f"Few-shot macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
