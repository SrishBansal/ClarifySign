from __future__ import annotations

from dataclasses import dataclass, field

from clarifysign.data.schemas import SemanticIntent


@dataclass(slots=True)
class DialogueState:
    intents: list[SemanticIntent] = field(default_factory=list)

    def record(self, intent: SemanticIntent) -> None:
        self.intents.append(intent)
