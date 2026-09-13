"""Pydantic request/response schemas for the inference API.

No `confidence` field: a generative model constrained to one of 77 labels
doesn't have a clean softmax-over-classes the way a sequence-classification
head does. See build plan section 9 -- omit rather than fabricate.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class ClassifyResponse(BaseModel):
    intent: str
    latency_ms: float
    model: str


class HealthResponse(BaseModel):
    status: str
    loaded: bool


class ModelInfoResponse(BaseModel):
    base_model: str
    quantization: str
    adapter_path: str
    dataset: str
