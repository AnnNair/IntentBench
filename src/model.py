"""Model loading: base model in 4-bit NF4, optional LoRA adapter attachment.

All heavy imports (torch/transformers/peft/bitsandbytes) are deferred inside
functions rather than at module scope. This keeps the module importable --
and therefore mockable in tests -- on machines without a GPU or those
packages installed (see tests/test_api.py).
"""
from __future__ import annotations

from typing import Any


def load_quantized_base_model(model_name: str, compute_dtype: str = "float16"):
    """Load the base instruction-tuned model in 4-bit NF4 for QLoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = getattr(torch, compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


def attach_lora(model, lora_config: dict[str, Any]):
    """Wrap a quantized base model with a LoRA adapter and verify trainable size.

    Raises if the trainable-parameter fraction looks like a full fine-tune
    (see build plan section 4.1: this is the cheapest bug to catch before
    spending GPU time, and the most expensive one to discover after).
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    target_modules = lora_config.get("target_modules", "all-linear")
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=target_modules,
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)

    trainable, total = count_trainable_parameters(model)
    fraction = trainable / total if total else 0.0
    model.print_trainable_parameters()
    if fraction > 0.01:
        raise RuntimeError(
            f"Trainable parameter fraction {fraction:.2%} looks like a full "
            "fine-tune, not a LoRA adapter. Check target_modules before "
            "spending GPU time on a training run."
        )
    return model


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_model_with_adapter(base_model_name: str, adapter_path: str):
    """Load the base model plus a trained LoRA adapter for inference/serving."""
    from peft import PeftModel

    model, tokenizer = load_quantized_base_model(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
