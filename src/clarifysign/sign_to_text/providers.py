from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from clarifysign.data.schemas import Candidate, SignToTextRequest, SignToTextResponse
from clarifysign.sign_to_text.schemas import EncodedSignSequence, InputQualityMetrics, PoseExtraction, Prediction, PredictionCandidate


@dataclass(frozen=True, slots=True)
class MockSignToTextProvider:
    """CPU-only deterministic recognizer for tests and local wiring checks."""

    label: str = "mock_sign"

    def predict(self, request: SignToTextRequest) -> SignToTextResponse:
        if not request.pose_sequence:
            raise ValueError("mock recognizer requires a non-empty pose sequence")
        return SignToTextResponse(
            candidates=(Candidate(self.label, 0.90), Candidate("unknown", 0.10)),
            provider="mock-sign-to-text",
            model_version="mock-v1",
        )


@dataclass(frozen=True, slots=True)
class MockPoseExtractor:
    """Deterministic test extractor; it accepts already numeric synthetic frames."""

    def extract(self, frames: Sequence[Sequence[float]]) -> PoseExtraction:
        pose_frames = tuple(tuple(float(value) for value in frame) for frame in frames)
        valid = sum(bool(frame) for frame in pose_frames)
        count = len(pose_frames)
        ratio = valid / count if count else 0.0
        return PoseExtraction(pose_frames, InputQualityMetrics(count, valid, ratio, ratio), "mock-pose-v1")


@dataclass(frozen=True, slots=True)
class IdentityTemporalSignEncoder:
    def encode(self, extraction: PoseExtraction) -> EncodedSignSequence:
        if not extraction.pose_frames:
            raise ValueError("cannot encode an empty pose sequence")
        return EncodedSignSequence(extraction.pose_frames, "identity-temporal-v1")


@dataclass(frozen=True, slots=True)
class MockSignToTextModel:
    """CPU-only model baseline for contract tests; no model file is loaded."""

    label: str = "mock_sign"

    def predict(self, encoded: EncodedSignSequence, input_quality: PoseExtraction) -> Prediction:
        if not encoded.values:
            raise ValueError("mock model requires encoded pose values")
        started = perf_counter()
        candidates = (PredictionCandidate(self.label, 0.90), PredictionCandidate("unknown", 0.10))
        return Prediction(
            text=self.label,
            top_k_candidates=candidates,
            token_probabilities={self.label: 0.90, "unknown": 0.10},
            confidence=0.90,
            latency_ms=(perf_counter() - started) * 1000,
            model_version="mock-sign-to-text-v1",
            input_quality=input_quality.quality,
        )
