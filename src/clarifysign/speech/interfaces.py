from __future__ import annotations

from typing import Protocol

from clarifysign.data.schemas import AudioRequest, TextRequest


class ASRProvider(Protocol):
    def transcribe(self, request: AudioRequest) -> str: ...


class TTSProvider(Protocol):
    def synthesize(self, request: TextRequest) -> bytes: ...
