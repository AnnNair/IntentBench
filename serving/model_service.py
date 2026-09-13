"""ModelService: loads the model once at startup (adapter-at-load mode) and
serves predictions. Deliberately isolated from FastAPI wiring so tests can
substitute a mock implementation without importing torch/transformers/peft
(see build plan section 9.1 -- mock the model, don't load it per test).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from src.model import generate, load_model_with_adapter
from src.prompts import build_prompt, parse_label
from src.utils import load_json


class ModelNotReadyError(RuntimeError):
    pass


class ModelService:
    def __init__(self, base_model: str, adapter_path: str, max_new_tokens: int = 16):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._labels: list[str] | None = None

    def load(self) -> None:
        """Load model + adapter + label mapping. Fails fast if files are missing."""
        adapter_dir = Path(self.adapter_path)
        labels_path = adapter_dir / "labels.json"
        if not adapter_dir.exists() or not labels_path.exists():
            raise ModelNotReadyError(
                f"Adapter path '{self.adapter_path}' or its labels.json is missing. "
                "Run training (src/train.py) before starting the API."
            )
        self._labels = load_json(labels_path)
        self._model, self._tokenizer = load_model_with_adapter(self.base_model, self.adapter_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def classify(self, text: str) -> tuple[str, float]:
        if not self.is_loaded:
            raise ModelNotReadyError("Model is not loaded yet.")

        start = time.perf_counter()
        prompt = build_prompt(text, self._labels)
        raw_output = generate(self._model, self._tokenizer, prompt, self.max_new_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0

        label = parse_label(raw_output, self._labels)
        if label is None:
            raise ValueError(f"Model produced an unparseable output: {raw_output!r}")
        return label, latency_ms


def build_default_service() -> ModelService:
    return ModelService(
        base_model=os.environ.get("BASE_MODEL", "google/gemma-2-2b-it"),
        adapter_path=os.environ.get("MODEL_ADAPTER_PATH", "outputs/best"),
    )
