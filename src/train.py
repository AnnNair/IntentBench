"""QLoRA training entrypoint.

Usage:
    python -m src.train --base-config configs/base.yaml --train-config configs/qlora.yaml

Reads every hyperparameter from configs/*.yaml (never hard-coded), fixes the
seed, logs to W&B, and saves the adapter + tokenizer + labels.json + a
snapshot of the exact config used -- so training and serving can never drift
apart on the label mapping (build plan section 8).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data import get_label_names, load_banking77, make_splits
from src.model import attach_lora, load_quantized_base_model
from src.prompts import build_training_example
from src.utils import load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--train-config", default="configs/qlora.yaml")
    return parser.parse_args()


def build_dataset(train_split, label_names: list[str], tokenizer):
    def to_text(example):
        return {"text": build_training_example(example["text"], label_names[example["label"]])}

    return train_split.map(to_text)


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.train_config)
    set_seed(config["seed"])

    wandb_run = None
    try:
        import wandb

        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity"),
            name=config["run_id"],
            config=config,
        )
    except ImportError:
        print("wandb not installed -- continuing without experiment tracking")

    train_raw, _ = load_banking77(config["dataset"]["hf_path"])
    label_names = get_label_names(train_raw)
    train_split, val_split = make_splits(train_raw, config["dataset"]["val_fraction"], config["seed"])

    model, tokenizer = load_quantized_base_model(
        config["model"]["base_model"], config["model"]["compute_dtype"]
    )
    model = attach_lora(model, config["lora"])

    train_dataset = build_dataset(train_split, label_names, tokenizer)
    val_dataset = build_dataset(val_split, label_names, tokenizer)

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        eval_strategy=config["training"]["eval_strategy"],
        save_total_limit=config["training"]["save_total_limit"],
        report_to=config["training"]["report_to"] if wandb_run else "none",
        seed=config["seed"],
    )

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=256)

    train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize, batched=True, remove_columns=val_dataset.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=collator,
    )

    start = time.time()
    trainer.train()
    training_minutes = (time.time() - start) / 60.0

    best_dir = Path(config["training"]["best_model_dir"])
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    save_json(label_names, best_dir / "labels.json")
    save_json({**config, "training_minutes": training_minutes}, best_dir / "run_config.json")

    if wandb_run:
        wandb_run.log({"training_minutes": training_minutes})
        wandb_run.finish()


if __name__ == "__main__":
    main()
