"""Pydantic contracts for calibrated clarification decisions."""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InputQualityMetrics(_Schema):
    valid_signal_ratio: float = Field(ge=0.0, le=1.0)
    temporal_coverage: float = Field(ge=0.0, le=1.0)
    frame_or_chunk_count: int = Field(ge=0)


class CandidateProbability(_Schema):
    text: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    semantic_group: str | None = None


class _CalibratedPrediction(_Schema):
    text: str = Field(min_length=1)
    top_k_candidates: tuple[CandidateProbability, ...] = Field(min_length=1)
    token_probabilities: Mapping[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    model_version: str = Field(min_length=1)
    input_quality: InputQualityMetrics

    @model_validator(mode="after")
    def candidate_matches_text(self) -> "_CalibratedPrediction":
        if self.top_k_candidates[0].text != self.text:
            raise ValueError("top candidate text must match prediction text")
        if any(not 0.0 <= value <= 1.0 for value in self.token_probabilities.values()):
            raise ValueError("token probabilities must be in [0, 1]")
        return self


class SignToTextPrediction(_CalibratedPrediction):
    source: Literal["sign"] = "sign"


class SpeechToTextPrediction(_CalibratedPrediction):
    source: Literal["speech"] = "speech"


CalibratedPrediction = SignToTextPrediction | SpeechToTextPrediction
ClarificationModality = Literal["speech", "text", "sign_animation", "yes_no_gesture", "multiple_choice_ui"]


class ClarificationOption(_Schema):
    option_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)


class ClarificationRequest(_Schema):
    session_id: str = Field(min_length=1)
    modality: ClarificationModality
    prompt: str = Field(min_length=1)
    options: tuple[ClarificationOption, ...] = Field(min_length=1)
    trigger_reasons: tuple[str, ...] = Field(min_length=1)
    expected_entropy_reduction: float = Field(ge=0.0)
    interaction_cost: float = Field(gt=0.0)
    expected_utility: float = Field(ge=0.0)


class ClarificationResult(_Schema):
    status: Literal["final", "clarification_required", "safe_fallback"]
    final_text: str | None = None
    request: ClarificationRequest | None = None
    reason: str

    @model_validator(mode="after")
    def result_shape(self) -> "ClarificationResult":
        if self.status == "clarification_required" and self.request is None:
            raise ValueError("clarification_required needs a request")
        if self.status != "clarification_required" and self.request is not None:
            raise ValueError("final/fallback must not contain a request")
        if self.status == "final" and not self.final_text:
            raise ValueError("final result needs final_text")
        return self
