"""FastAPI inference service: GET /health, GET /model-info, POST /classify.

The model is loaded once at startup, never per-request. Startup fails fast
if the adapter path or model files are missing (build plan section 9).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from serving.model_service import ModelNotReadyError, ModelService, build_default_service
from serving.schemas import ClassifyRequest, ClassifyResponse, HealthResponse, ModelInfoResponse

_service: ModelService = build_default_service()


def get_service() -> ModelService:
    return _service


def set_service(service: ModelService) -> None:
    """Swap the module-level service instance -- used by tests to inject a mock."""
    global _service
    _service = service


@asynccontextmanager
async def lifespan(app: FastAPI):
    _service.load()
    yield


app = FastAPI(title="IntentBench Serving API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok" if _service.is_loaded else "not_ready", loaded=_service.is_loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        base_model=_service.base_model,
        quantization="4-bit NF4",
        adapter_path=_service.adapter_path,
        dataset="banking77",
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    try:
        intent, latency_ms = _service.classify(request.text)
    except ModelNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ClassifyResponse(intent=intent, latency_ms=latency_ms, model="intentbench-gemma-qlora")
