"""
ClarifySign Backend - Inference Service
Singleton that loads the ISLBiLSTM model once and exposes predict().
"""

import sys
import time
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2

# ── ensure project root on path so core.* imports resolve ────────────────────
_backend_dir = Path(__file__).resolve().parent.parent.parent
_project_root = _backend_dir.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.config import MODEL_PATH, LABELS_PATH, HOLISTIC_TASK_PATH, DEVICE, TEMPERATURE
from app.utils.exceptions import ModelNotLoadedError, InvalidFrameError, InferenceError

logger = logging.getLogger(__name__)


def _resolve_device():
    """Auto-detect best available device."""
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class InferenceService:
    """Singleton inference service wrapping the core ISLRecognizer + HolisticExtractor."""

    _instance: "InferenceService | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._recognizer = None
        self._extractor = None
        self._device = _resolve_device()
        self._model_loaded = False
        self._load()

    def _load(self):
        try:
            from core.recognizer import ISLRecognizer
            self._recognizer = ISLRecognizer(
                model_path=MODEL_PATH,
                labels_path=LABELS_PATH,
                temperature=TEMPERATURE,
                device=self._device,
            )
            if self._recognizer.available:
                self._model_loaded = True
                logger.info(
                    "ISLBiLSTM model loaded on %s (%d classes)",
                    self._device,
                    len(self._recognizer.labels),
                )
            else:
                logger.warning("ISLRecognizer loaded but model not available (weights missing?)")
        except Exception as exc:
            logger.error("Failed to load ISLRecognizer: %s", exc)

        try:
            from core.landmarks import HolisticExtractor
            self._extractor = HolisticExtractor(model_path=HOLISTIC_TASK_PATH)
            logger.info("HolisticExtractor initialized")
        except Exception as exc:
            logger.warning("HolisticExtractor unavailable: %s", exc)

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def labels(self) -> List[str]:
        """Returns the list of class labels from the loaded recognizer."""
        if self._recognizer is not None:
            return list(self._recognizer.labels)
        return []

    def _decode_frame(self, frame_b64: str) -> np.ndarray:
        """Decode base64 image → BGR numpy array."""
        try:
            # Strip data-URI prefix if present
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]
            raw = base64.b64decode(frame_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                raise ValueError("cv2.imdecode returned None")
            return img
        except Exception as exc:
            raise InvalidFrameError(f"Cannot decode frame: {exc}") from exc

    def predict(
        self, frame_b64: str, top_k: int = 5
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        """
        Full pipeline: decode → landmark extraction → model inference → uncertainty.

        Returns:
            candidates: list of (label, probability) sorted descending
            uncertainty: dict with entropy, margin, is_ambiguous, etc.
        """
        if not self._model_loaded:
            raise ModelNotLoadedError(
                "ISLBiLSTM model is not loaded. Check MODEL_PATH and restart."
            )

        t0 = time.perf_counter()

        # 1. Decode frame
        frame = self._decode_frame(frame_b64)

        # 2. Extract landmarks
        try:
            if self._extractor is not None:
                feature_vec = self._extractor(frame)  # shape (225,)
            else:
                feature_vec = np.zeros(225, dtype=np.float32)
        except Exception as exc:
            raise InferenceError(f"Landmark extraction failed: {exc}") from exc

        # 3. Build sequence (replicate single frame to fill sequence)
        from app.config import SEQUENCE_LENGTH, FEATURE_DIM
        if feature_vec.shape[0] != FEATURE_DIM:
            feature_vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        sequence = np.tile(feature_vec, (SEQUENCE_LENGTH, 1))  # (T, D)

        # 4. Run model
        try:
            candidates = self._recognizer.predict(sequence, top_k=top_k)
        except Exception as exc:
            raise InferenceError(f"Model inference failed: {exc}") from exc

        # 5. Compute uncertainty
        from core.uncertainty import report as uncertainty_report
        uncertainty = uncertainty_report(candidates)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Predict: top=%s p=%.3f entropy=%.3f ms=%.1f",
            candidates[0][0] if candidates else "?",
            candidates[0][1] if candidates else 0.0,
            uncertainty.get("entropy_bits", 0.0),
            elapsed_ms,
        )

        return candidates, uncertainty, elapsed_ms


# Module-level singleton accessor
_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service
