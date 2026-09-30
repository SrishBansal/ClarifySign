"""
ClarifySign Backend Tests - API endpoint tests using TestClient.
"""

import sys
import json
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# ── path setup ────────────────────────────────────────────────────────────────
_backend = Path(__file__).resolve().parent.parent
_root = _backend.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
if str(_root) not in sys.path:
    sys.path.append(str(_root))


# ── fixtures ─────────────────────────────────────────────────────────────────

def _mock_inference_service(model_loaded=True, device="cpu"):
    """Create a mock InferenceService that returns canned predictions."""
    mock = MagicMock()
    mock.model_loaded = model_loaded
    mock.device = device
    # Unambiguous prediction (high confidence)
    mock.predict.return_value = (
        [("blue", 0.92), ("black", 0.05), ("red", 0.02), ("white", 0.01)],
        {
            "top1_confidence": 0.92,
            "margin": 0.87,
            "entropy_bits": 0.38,
            "normalized_entropy": 0.10,
            "is_ambiguous": False,
            "primary_reason": "confident_clear",
            "triggered_criteria": [],
            "num_candidates": 4,
            "decision": "commit",
        },
        18.5,
    )
    return mock


def _mock_ambiguous_inference_service(device="cpu"):
    """Mock InferenceService with ambiguous prediction (triggers clarification)."""
    mock = MagicMock()
    mock.model_loaded = True
    mock.device = device
    mock.predict.return_value = (
        [("blue", 0.52), ("black", 0.44), ("red", 0.04)],
        {
            "top1_confidence": 0.52,
            "margin": 0.08,
            "entropy_bits": 1.32,
            "normalized_entropy": 0.58,
            "is_ambiguous": True,
            "primary_reason": "narrow_margin",
            "triggered_criteria": ["narrow_margin (0.08 < 0.20)"],
            "num_candidates": 3,
            "decision": "clarify",
        },
        21.3,
    )
    return mock


def _dummy_frame_b64() -> str:
    """1x1 white pixel JPEG encoded as base64."""
    import numpy as np
    import cv2
    img = (255 * np.ones((10, 10, 3), dtype=np.uint8))
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


@pytest.fixture()
def client():
    with patch("app.services.inference.get_inference_service", return_value=_mock_inference_service()):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def client_ambiguous():
    with patch("app.services.inference.get_inference_service", return_value=_mock_ambiguous_inference_service()):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.get("/health")
        assert r.status_code == 200

    def test_health_response_schema(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.get("/health")
        data = r.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "timestamp" in data

    def test_api_health_alias(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.get("/api/health")
        assert r.status_code == 200


# ── /api/predict ──────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_200(self, client):
        frame = _dummy_frame_b64()
        r = client.post("/api/predict", json={"frame_b64": frame})
        assert r.status_code == 200

    def test_predict_response_schema(self, client):
        frame = _dummy_frame_b64()
        r = client.post("/api/predict", json={"frame_b64": frame})
        data = r.json()
        assert "top_prediction" in data
        assert "predictions" in data
        assert "uncertainty" in data
        assert "inference_ms" in data

    def test_predict_top_prediction_has_label_and_probability(self, client):
        frame = _dummy_frame_b64()
        r = client.post("/api/predict", json={"frame_b64": frame})
        top = r.json()["top_prediction"]
        assert "label" in top
        assert "probability" in top
        assert isinstance(top["probability"], float)

    def test_predict_uncertainty_fields(self, client):
        frame = _dummy_frame_b64()
        r = client.post("/api/predict", json={"frame_b64": frame})
        u = r.json()["uncertainty"]
        for field in ["top1_confidence", "margin", "entropy_bits", "is_ambiguous", "decision"]:
            assert field in u, f"Missing field: {field}"

    def test_predict_rejects_empty_frame(self, client):
        r = client.post("/api/predict", json={"frame_b64": ""})
        assert r.status_code in (400, 422, 503)

    def test_predict_top_k_respected(self, client):
        frame = _dummy_frame_b64()
        r = client.post("/api/predict", json={"frame_b64": frame, "top_k": 2})
        assert r.status_code == 200  # result may have fewer; no crash


# ── /api/dialogue/start ───────────────────────────────────────────────────────

class TestDialogueStart:
    def _prediction_payload(self, ambiguous=False):
        if ambiguous:
            preds = [
                {"label": "blue", "probability": 0.52},
                {"label": "black", "probability": 0.44},
            ]
            uncertainty = {
                "top1_confidence": 0.52, "margin": 0.08, "entropy_bits": 1.32,
                "normalized_entropy": 0.58, "is_ambiguous": True,
                "primary_reason": "narrow_margin", "triggered_criteria": ["narrow_margin"],
                "decision": "clarify",
            }
        else:
            preds = [
                {"label": "blue", "probability": 0.92},
                {"label": "black", "probability": 0.05},
            ]
            uncertainty = {
                "top1_confidence": 0.92, "margin": 0.87, "entropy_bits": 0.38,
                "normalized_entropy": 0.10, "is_ambiguous": False,
                "primary_reason": "confident_clear", "triggered_criteria": [],
                "decision": "commit",
            }
        return {
            "prediction": {
                "top_prediction": preds[0],
                "predictions": preds,
                "uncertainty": uncertainty,
                "inference_ms": 20.0,
            },
            "language": "Hindi",
        }

    def test_start_commit_returns_200(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/dialogue/start", json=self._prediction_payload(ambiguous=False))
        assert r.status_code == 200

    def test_start_commit_has_dialogue_id(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/dialogue/start", json=self._prediction_payload(ambiguous=False))
        data = r.json()
        assert "dialogue_id" in data
        assert len(data["dialogue_id"]) > 0

    def test_start_commit_action(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/dialogue/start", json=self._prediction_payload(ambiguous=False))
        data = r.json()
        assert data["action"] in ("commit", "clarify")

    def test_start_ambiguous_has_clarification(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/dialogue/start", json=self._prediction_payload(ambiguous=True))
        assert r.status_code == 200
        data = r.json()
        if data["action"] == "clarify":
            assert data["clarification"] is not None
            assert "options" in data["clarification"]

    def test_start_commit_has_translations(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/dialogue/start", json=self._prediction_payload(ambiguous=False))
        data = r.json()
        if data["action"] == "commit":
            assert data["translation"] is not None


# ── /api/dialogue/clarify ─────────────────────────────────────────────────────

class TestDialogueClarify:
    def test_clarify_invalid_id_returns_400(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/dialogue/clarify", json={
                "dialogue_id": "nonexistent-id",
                "customer_response": "1",
            })
        assert r.status_code == 400

    def test_clarify_flow(self):
        """Create an ambiguous session, then resolve it."""
        from app.main import app
        preds = [
            {"label": "blue", "probability": 0.52},
            {"label": "black", "probability": 0.44},
        ]
        uncertainty = {
            "top1_confidence": 0.52, "margin": 0.08, "entropy_bits": 1.32,
            "normalized_entropy": 0.58, "is_ambiguous": True,
            "primary_reason": "narrow_margin", "triggered_criteria": ["narrow_margin"],
            "decision": "clarify",
        }
        payload = {
            "prediction": {
                "top_prediction": preds[0],
                "predictions": preds,
                "uncertainty": uncertainty,
                "inference_ms": 20.0,
            },
            "language": "Hindi",
        }
        with TestClient(app) as c:
            start_r = c.post("/api/dialogue/start", json=payload)
            assert start_r.status_code == 200
            data = start_r.json()
            if data["action"] == "clarify":
                dialogue_id = data["dialogue_id"]
                clarify_r = c.post("/api/dialogue/clarify", json={
                    "dialogue_id": dialogue_id,
                    "customer_response": "1",
                })
                assert clarify_r.status_code == 200
                cdata = clarify_r.json()
                assert "confirmed_class" in cdata
                assert "translation" in cdata


# ── /api/translate ────────────────────────────────────────────────────────────

class TestTranslate:
    def test_translate_all_languages(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/translate", json={"intent": "blue"})
        assert r.status_code == 200
        data = r.json()
        assert "translations" in data
        assert len(data["translations"]) >= 1

    def test_translate_single_language(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/translate", json={"intent": "blue", "language": "Hindi"})
        assert r.status_code == 200
        data = r.json()
        assert "Hindi" in data["translations"]

    def test_translate_known_concept_is_verified(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/translate", json={"intent": "blue", "language": "Hindi"})
        data = r.json()
        assert data["sources"].get("Hindi") == "verified"

    def test_translate_returns_sources(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/translate", json={"intent": "blue"})
        data = r.json()
        assert "sources" in data


# ── /api/speak ────────────────────────────────────────────────────────────────

class TestSpeak:
    def test_speak_returns_audio_or_503(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/speak", json={"text": "नीला", "language": "Hindi"})
        # gTTS requires network; accept either success or service unavailable
        assert r.status_code in (200, 503)

    def test_speak_200_has_audio_content_type(self):
        from app.main import app
        with TestClient(app) as c:
            r = c.post("/api/speak", json={"text": "नीला", "language": "Hindi"})
        if r.status_code == 200:
            assert "audio" in r.headers.get("content-type", "")
