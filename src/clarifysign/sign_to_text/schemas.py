"""Typed values exchanged within the sign-to-text pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class InputQualityMetrics:
    frame_count: int
    valid_pose_frames: int
    valid_pose_ratio: float
    temporal_coverage: float

    def __post_init__(self) -> None:
        if self.frame_count < 0 or not 0 <= self.valid_pose_frames <= self.frame_count:
            raise ValueError("invalid frame counts")
        if not 0.0 <= self.valid_pose_ratio <= 1.0 or not 0.0 <= self.temporal_coverage <= 1.0:
            raise ValueError("quality metrics must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PoseExtraction:
    pose_frames: tuple[tuple[float, ...], ...]
    quality: InputQualityMetrics
    extractor_version: str = "unversioned"


@dataclass(frozen=True, slots=True)
class EncodedSignSequence:
    values: tuple[tuple[float, ...], ...]
    encoder_version: str = "unversioned"


@dataclass(frozen=True, slots=True)
class PredictionCandidate:
    text: str
    probability: float

    def __post_init__(self) -> None:
        if not self.text.strip() or not 0.0 <= self.probability <= 1.0:
            raise ValueError("candidate text and probability are invalid")


@dataclass(frozen=True, slots=True)
class Prediction:
    text: str
    top_k_candidates: tuple[PredictionCandidate, ...]
    token_probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: float = 0.0
    model_version: str = "unversioned"
    input_quality: InputQualityMetrics = field(default_factory=lambda: InputQualityMetrics(0, 0, 0.0, 0.0))

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.top_k_candidates:
            raise ValueError("prediction text and candidates are required")
        if not 0.0 <= self.confidence <= 1.0 or self.latency_ms < 0:
            raise ValueError("confidence or latency is invalid")
        if any(not 0.0 <= probability <= 1.0 for probability in self.token_probabilities.values()):
            raise ValueError("token probabilities must be in [0, 1]")
