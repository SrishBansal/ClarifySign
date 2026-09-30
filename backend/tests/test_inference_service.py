"""
ClarifySign Backend Tests - InferenceService unit tests.
"""

import sys
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

_backend = Path(__file__).resolve().parent.parent
_root = _backend.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
if str(_root) not in sys.path:
    sys.path.append(str(_root))

from app.utils.exceptions import ModelNotLoadedError, InvalidFrameError


def _make_frame_b64(width=20, height=20) -> str:
    import cv2
    img = (np.ones((height, width, 3), dtype=np.uint8) * 128)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


class TestInferenceService:

    def test_invalid_base64_raises(self):
        from app.services.inference import InferenceService
        svc = InferenceService.__new__(InferenceService)
        svc._initialized = True
        svc._model_loaded = True
        svc._recognizer = None
        svc._extractor = None
        svc._device = "cpu"
        with pytest.raises((InvalidFrameError, Exception)):
            svc._decode_frame("not-valid-base64!!!")

    def test_decode_valid_frame(self):
        from app.services.inference import InferenceService
        svc = InferenceService.__new__(InferenceService)
        svc._initialized = True
        frame = svc._decode_frame(_make_frame_b64())
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3

    def test_model_not_loaded_raises(self):
        from app.services.inference import InferenceService
        svc = InferenceService.__new__(InferenceService)
        svc._initialized = True
        svc._model_loaded = False
        svc._recognizer = None
        svc._extractor = None
        svc._device = "cpu"
        with pytest.raises(ModelNotLoadedError):
            svc.predict(_make_frame_b64())

    def test_singleton_pattern(self):
        from app.services.inference import InferenceService
        a = InferenceService()
        b = InferenceService()
        assert a is b
