from __future__ import annotations

from dataclasses import dataclass

from clarifysign.data.schemas import AudioRequest, TextRequest


@dataclass(frozen=True, slots=True)
class MockASRProvider:
    transcript: str = "mock transcript"

    def transcribe(self, request: AudioRequest) -> str:
        if not request.audio:
            raise ValueError("audio cannot be empty")
        return self.transcript


@dataclass(frozen=True, slots=True)
class MockTTSProvider:
    def synthesize(self, request: TextRequest) -> bytes:
        return f"mock-tts:{request.language}:{request.text}".encode("utf-8")
