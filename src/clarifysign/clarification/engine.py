"""Deterministic clarification policy driven only by calibrated probabilities."""

from __future__ import annotations

from collections.abc import Mapping
from math import log2
from pathlib import Path
import sqlite3
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from clarifysign.clarification.schemas import (
    CalibratedPrediction,
    ClarificationOption,
    ClarificationRequest,
    ClarificationResult,
)


class ClarificationPolicySettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    confidence_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    entropy_threshold_bits: float = Field(default=1.0, ge=0.0)
    minimum_signal_ratio: float = Field(default=0.65, ge=0.0, le=1.0)
    minimum_temporal_coverage: float = Field(default=0.60, ge=0.0, le=1.0)
    maximum_failures: int = Field(default=3, ge=1)
    interaction_costs: Mapping[str, float] = {
        "speech": 1.4,
        "text": 1.0,
        "sign_animation": 1.8,
        "yes_no_gesture": 0.6,
        "multiple_choice_ui": 1.1,
    }


class ControlledParaphraser(Protocol):
    """Optional controlled paraphraser; it must preserve all supplied option terms."""

    def paraphrase(self, template: str, required_terms: tuple[str, ...]) -> str: ...


class DeterministicTemplateRenderer:
    def __init__(self, paraphraser: ControlledParaphraser | None = None) -> None:
        self._paraphraser = paraphraser

    def render(self, options: tuple[ClarificationOption, ...]) -> str:
        terms = tuple(option.text for option in options)
        template = "Please confirm the intended meaning: " + " / ".join(terms) + "."
        if self._paraphraser is None:
            return template
        rendered = self._paraphraser.paraphrase(template, terms)
        return rendered if all(term in rendered for term in terms) else template


class SQLiteDialogueStore:
    """Minimal persistent confirmation history; callers choose the SQLite path."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS clarification_events "
                "(session_id TEXT NOT NULL, event_type TEXT NOT NULL, resolved_text TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)

    def failures(self, session_id: str) -> int:
        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM clarification_events "
                "WHERE session_id = ? AND event_type = 'requested' AND rowid > COALESCE("
                "(SELECT MAX(rowid) FROM clarification_events WHERE session_id = ? AND event_type = 'confirmed'), 0)",
                (session_id, session_id),
            ).fetchone()[0])

    def record_request(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO clarification_events VALUES (?, 'requested', NULL)", (session_id,))

    def record_confirmation(self, session_id: str, resolved_text: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO clarification_events VALUES (?, 'confirmed', ?)", (session_id, resolved_text))


def _entropy_bits(prediction: CalibratedPrediction) -> float:
    return -sum(candidate.probability * log2(candidate.probability) for candidate in prediction.top_k_candidates if candidate.probability > 0)


def _semantic_conflict(prediction: CalibratedPrediction) -> bool:
    groups = {candidate.semantic_group for candidate in prediction.top_k_candidates[:2] if candidate.semantic_group}
    return len(groups) > 1


class CalibratedClarificationEngine:
    def __init__(
        self,
        store: SQLiteDialogueStore,
        settings: ClarificationPolicySettings | None = None,
        renderer: DeterministicTemplateRenderer | None = None,
    ) -> None:
        self._store = store
        self._settings = settings or ClarificationPolicySettings()
        self._renderer = renderer or DeterministicTemplateRenderer()

    def review(self, prediction: CalibratedPrediction, session_id: str) -> ClarificationResult:
        if self._store.failures(session_id) >= self._settings.maximum_failures:
            return ClarificationResult(
                status="safe_fallback", final_text=None, request=None,
                reason="repeated clarification failures; please retry with a clearer input or a human assistant",
            )
        reasons = self._trigger_reasons(prediction)
        if not reasons:
            return ClarificationResult(status="final", final_text=prediction.text, request=None, reason="calibrated prediction accepted")
        options = tuple(
            ClarificationOption(option_id=f"candidate_{index}", text=candidate.text, probability=candidate.probability)
            for index, candidate in enumerate(prediction.top_k_candidates, start=1)
        )
        modality = self._select_modality(prediction, reasons, len(options))
        reduction = _entropy_bits(prediction)
        cost = self._settings.interaction_costs[modality]
        request = ClarificationRequest(
            session_id=session_id, modality=modality, prompt=self._renderer.render(options), options=options,
            trigger_reasons=tuple(reasons), expected_entropy_reduction=reduction,
            interaction_cost=cost, expected_utility=reduction / cost,
        )
        self._store.record_request(session_id)
        return ClarificationResult(status="clarification_required", final_text=None, request=request, reason="; ".join(reasons))

    def confirm(self, request: ClarificationRequest, option_id: str) -> ClarificationResult:
        option = next((item for item in request.options if item.option_id == option_id), None)
        if option is None:
            return ClarificationResult(status="safe_fallback", final_text=None, request=None, reason="unrecognized clarification response")
        self._store.record_confirmation(request.session_id, option.text)
        return ClarificationResult(status="final", final_text=option.text, request=None, reason="user confirmation recorded")

    def _trigger_reasons(self, prediction: CalibratedPrediction) -> list[str]:
        reasons: list[str] = []
        if prediction.confidence < self._settings.confidence_threshold:
            reasons.append("low calibrated confidence")
        if _entropy_bits(prediction) > self._settings.entropy_threshold_bits:
            reasons.append("high candidate entropy")
        if prediction.input_quality.valid_signal_ratio < self._settings.minimum_signal_ratio or prediction.input_quality.temporal_coverage < self._settings.minimum_temporal_coverage:
            reasons.append("poor input quality")
        if _semantic_conflict(prediction):
            reasons.append("semantically conflicting candidates")
        return reasons

    @staticmethod
    def _select_modality(prediction: CalibratedPrediction, reasons: list[str], option_count: int) -> str:
        if "poor input quality" in reasons:
            return "speech" if prediction.source == "sign" else "text"
        if prediction.source == "speech" and "high candidate entropy" in reasons:
            return "sign_animation"
        if prediction.source == "sign" and option_count == 2:
            return "yes_no_gesture"
        return "multiple_choice_ui"
