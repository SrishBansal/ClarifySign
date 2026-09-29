"""Local-only compatibility and future-checkpoint adapters.

Neither adapter downloads a model. The legacy adapter is injectable for tests;
the transformer adapter is intentionally a placeholder, not an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol, Sequence

from clarifysign.sign_to_text.schemas import EncodedSignSequence, PoseExtraction, Prediction, PredictionCandidate


class LegacyIsolatedClassifier(Protocol):
    def predict(self, sequence: Sequence[Sequence[float]], top_k: int = 5) -> list[tuple[str, float]]: ...


@dataclass(slots=True)
class LegacyIsolatedSignClassifierAdapter:
    """Adapts an explicitly supplied local legacy classifier; never fetches weights."""

    classifier: LegacyIsolatedClassifier
    model_version: str = "legacy-local-bilstm"

    @classmethod
    def from_local_artifacts(cls, model_path: Path, labels_path: Path) -> "LegacyIsolatedSignClassifierAdapter":
        """Load only paths supplied by the caller; no network access is attempted."""
        if not model_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError("legacy model and labels must both exist locally")
        from core.recognizer import ISLRecognizer

        return cls(ISLRecognizer(model_path=model_path, labels_path=labels_path))

    def predict(self, encoded: EncodedSignSequence, input_quality: PoseExtraction) -> Prediction:
        started = perf_counter()
        candidates = tuple(
            PredictionCandidate(text=label, probability=float(probability))
            for label, probability in self.classifier.predict(encoded.values, top_k=5)
        )
        if not candidates:
            raise RuntimeError("legacy classifier returned no candidates")
        return Prediction(
            text=candidates[0].text,
            top_k_candidates=candidates,
            token_probabilities={candidate.text: candidate.probability for candidate in candidates},
            confidence=candidates[0].probability,
            latency_ms=(perf_counter() - started) * 1000,
            model_version=self.model_version,
            input_quality=input_quality.quality,
        )


@dataclass(frozen=True, slots=True)
class TransformerCheckpointAdapter:
    """Reserved for a reviewed fine-tuned local checkpoint implementation."""

    checkpoint_path: Path | None = None
    model_version: str = "transformer-placeholder"

    def predict(self, encoded: EncodedSignSequence, input_quality: PoseExtraction) -> Prediction:
        del encoded, input_quality
        if self.checkpoint_path is None:
            raise RuntimeError("No local fine-tuned checkpoint configured; automatic pretrained downloads are disabled.")
        raise NotImplementedError("Transformer checkpoint loading is not implemented; register a reviewed local adapter first.")
