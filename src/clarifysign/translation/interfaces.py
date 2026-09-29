from __future__ import annotations

from typing import Protocol

from clarifysign.data.schemas import SemanticIntent


class TranslationProvider(Protocol):
    def translate(self, intent: SemanticIntent, target_language: str) -> str: ...
