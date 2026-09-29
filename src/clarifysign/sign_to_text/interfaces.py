from __future__ import annotations

from typing import Protocol, Sequence

from clarifysign.data.schemas import SignToTextRequest, SignToTextResponse
from clarifysign.sign_to_text.schemas import EncodedSignSequence, PoseExtraction, Prediction


class SignToTextProvider(Protocol):
    def predict(self, request: SignToTextRequest) -> SignToTextResponse: ...


class PoseExtractor(Protocol):
    """Extracts normalized pose frames from a capture adapter's frame representation."""

    def extract(self, frames: Sequence[Sequence[float]]) -> PoseExtraction: ...


class TemporalSignEncoder(Protocol):
    """Encodes a variable-length pose sequence for a sign-to-text model."""

    def encode(self, extraction: PoseExtraction) -> EncodedSignSequence: ...


class SignToTextModel(Protocol):
    """Predicts signed text from encoded pose; implementations may be local or remote."""

    def predict(self, encoded: EncodedSignSequence, input_quality: PoseExtraction) -> Prediction: ...
