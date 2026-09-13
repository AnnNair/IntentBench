"""API tests against a mocked ModelService -- no GPU, no real model load
(build plan section 9.1: mock the model, don't load it per test)."""
import pytest
from fastapi.testclient import TestClient

from serving.app import app, set_service
from serving.model_service import ModelNotReadyError


class FakeModelService:
    """Stands in for ModelService: same interface, no torch/transformers."""

    def __init__(self, loaded: bool = True, fixed_intent: str = "card_not_working"):
        self.base_model = "google/gemma-2-2b-it"
        self.adapter_path = "outputs/best"
        self._loaded = loaded
        self._fixed_intent = fixed_intent

    def load(self) -> None:
        pass  # loading already simulated via __init__

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def classify(self, text: str):
        if not self._loaded:
            raise ModelNotReadyError("not loaded")
        if not text.strip():
            raise ValueError("empty text")
        return self._fixed_intent, 12.3


@pytest.fixture
def client():
    set_service(FakeModelService())
    with TestClient(app) as c:
        yield c


def test_health_reports_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "loaded": True}


def test_health_reports_not_ready_when_model_unloaded():
    set_service(FakeModelService(loaded=False))
    with TestClient(app) as c:
        response = c.get("/health")
    assert response.json()["loaded"] is False


def test_model_info_exposes_metadata(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["base_model"] == "google/gemma-2-2b-it"
    assert body["quantization"] == "4-bit NF4"


def test_classify_returns_intent_and_latency_no_confidence_field(client):
    response = client.post("/classify", json={"text": "Why was my card declined?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "card_not_working"
    assert isinstance(body["latency_ms"], float)
    assert "confidence" not in body


def test_classify_rejects_empty_text(client):
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 422


def test_classify_rejects_missing_field(client):
    response = client.post("/classify", json={})
    assert response.status_code == 422


def test_classify_returns_503_when_model_not_ready():
    set_service(FakeModelService(loaded=False))
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post("/classify", json={"text": "hello"})
    assert response.status_code == 503
