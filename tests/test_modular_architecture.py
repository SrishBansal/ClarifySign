from app.backend.factory import build_local_service
from clarifysign.avatar.providers import MockAvatarProvider
from clarifysign.data.schemas import AudioRequest, SemanticIntent, SignToTextRequest, TextRequest
from clarifysign.sign_generation.providers import MockSignGenerationProvider
from clarifysign.speech.providers import MockASRProvider, MockTTSProvider


def test_local_service_uses_mock_providers_without_external_resources() -> None:
    result = build_local_service().sign_to_text(SignToTextRequest(((0.0, 1.0),)), "hi")
    assert result.decision.requires_clarification is False
    assert result.text == "[hi] mock_sign"
    assert result.provider == "mock-sign-to-text"


def test_reverse_direction_contract_is_provider_based_not_video_lookup() -> None:
    intent = SemanticIntent("request_item", {"item": "example"}, source="mock-asr")
    motion_plan = MockSignGenerationProvider().generate(intent)
    assert MockAvatarProvider().render(motion_plan).startswith("mock-avatar:")
    assert MockASRProvider().transcribe(AudioRequest(b"audio")) == "mock transcript"
    assert MockTTSProvider().synthesize(TextRequest("hello", "en")).startswith(b"mock-tts:")
