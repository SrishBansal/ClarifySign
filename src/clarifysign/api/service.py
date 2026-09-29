"""Application service composed entirely from typed provider interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from clarifysign.clarification.service import ClarificationDecision, ClarificationEngine
from clarifysign.data.schemas import SemanticIntent, SignToTextRequest
from clarifysign.dialogue.state import DialogueState
from clarifysign.preprocessing.interfaces import PosePreprocessor
from clarifysign.sign_to_text.interfaces import SignToTextProvider
from clarifysign.translation.interfaces import TranslationProvider


@dataclass(frozen=True, slots=True)
class VideoToTextResult:
    decision: ClarificationDecision
    text: str | None
    provider: str


class ClarifySignService:
    def __init__(
        self,
        preprocessor: PosePreprocessor,
        recognizer: SignToTextProvider,
        clarification: ClarificationEngine,
        translator: TranslationProvider,
        dialogue: DialogueState | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._recognizer = recognizer
        self._clarification = clarification
        self._translator = translator
        self._dialogue = dialogue or DialogueState()

    def sign_to_text(self, request: SignToTextRequest, target_language: str = "en") -> VideoToTextResult:
        prepared = self._preprocessor.transform(request.pose_sequence)
        response = self._recognizer.predict(SignToTextRequest(prepared, request.source_id))
        decision = self._clarification.decide(response.candidates)
        if decision.requires_clarification or decision.resolved_label is None:
            return VideoToTextResult(decision, None, response.provider)
        intent = SemanticIntent(decision.resolved_label, source=response.provider)
        self._dialogue.record(intent)
        return VideoToTextResult(decision, self._translator.translate(intent, target_language), response.provider)
