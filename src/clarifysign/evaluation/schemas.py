from __future__ import annotations

from dataclasses import dataclass

from clarifysign.data.schemas import SignToTextRequest


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    request: SignToTextRequest
    expected_label: str | None = None
