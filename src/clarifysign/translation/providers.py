from __future__ import annotations

from dataclasses import dataclass

from clarifysign.data.schemas import SemanticIntent


@dataclass(frozen=True, slots=True)
class MockTranslationProvider:
    """A transparent placeholder; it deliberately does not contain phrase maps."""

    def translate(self, intent: SemanticIntent, target_language: str) -> str:
        return f"[{target_language}] {intent.intent}"
