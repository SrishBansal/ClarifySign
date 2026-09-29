"""Typed, dependency-free configuration for the modular application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ClarificationSettings:
    confidence_threshold: float = 0.78
    margin_threshold: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    sign_to_text: str = "mock"
    asr: str = "mock"
    translation: str = "mock"
    sign_generation: str = "mock"
    avatar: str = "mock"
    tts: str = "mock"


@dataclass(frozen=True, slots=True)
class AppSettings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4])
    clarification: ClarificationSettings = field(default_factory=ClarificationSettings)
    providers: ProviderSettings = field(default_factory=ProviderSettings)
    default_language: str = "en"

    @property
    def config_directory(self) -> Path:
        return self.project_root / "configs"
