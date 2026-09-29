"""Request/response schemas shared across provider boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    probability: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("candidate label cannot be empty")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("candidate probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SignToTextRequest:
    """Pose features supplied by a capture/preprocessing adapter, not raw video."""

    pose_sequence: Sequence[Sequence[float]]
    source_id: str = "local"


@dataclass(frozen=True, slots=True)
class SignToTextResponse:
    candidates: tuple[Candidate, ...]
    provider: str
    model_version: str = "unversioned"

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("a recognition response needs at least one candidate")


@dataclass(frozen=True, slots=True)
class SemanticIntent:
    intent: str
    slots: Mapping[str, str] = field(default_factory=dict)
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class TextRequest:
    text: str
    language: str = "en"


@dataclass(frozen=True, slots=True)
class AudioRequest:
    audio: bytes
    language_hint: str | None = None
