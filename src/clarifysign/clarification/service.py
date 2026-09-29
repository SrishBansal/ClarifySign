from __future__ import annotations

from dataclasses import dataclass

from clarifysign.config.settings import ClarificationSettings
from clarifysign.data.schemas import Candidate


@dataclass(frozen=True, slots=True)
class ClarificationDecision:
    resolved_label: str | None
    requires_clarification: bool
    options: tuple[Candidate, ...]
    reason: str


class ClarificationEngine:
    def __init__(self, settings: ClarificationSettings) -> None:
        self._settings = settings

    def decide(self, candidates: tuple[Candidate, ...]) -> ClarificationDecision:
        ranked = tuple(sorted(candidates, key=lambda item: item.probability, reverse=True))
        if not ranked:
            raise ValueError("candidates cannot be empty")
        margin = ranked[0].probability - (ranked[1].probability if len(ranked) > 1 else 0.0)
        ambiguous = (
            ranked[0].probability < self._settings.confidence_threshold
            or margin < self._settings.margin_threshold
        )
        if ambiguous:
            return ClarificationDecision(None, True, ranked[:2], "low confidence or small margin")
        return ClarificationDecision(ranked[0].label, False, ranked[:1], "calibrated confidence accepted")
