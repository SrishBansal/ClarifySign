from pathlib import Path

import pytest

from clarifysign.sign_to_text.adapters import LegacyIsolatedSignClassifierAdapter, TransformerCheckpointAdapter
from clarifysign.sign_to_text.providers import IdentityTemporalSignEncoder, MockPoseExtractor, MockSignToTextModel


class FakeLegacyClassifier:
    def predict(self, sequence: object, top_k: int = 5) -> list[tuple[str, float]]:
        assert top_k == 5
        assert sequence
        return [("legacy-sign", 0.8), ("other", 0.2)]


def test_mock_pose_encoder_and_model_emit_complete_prediction() -> None:
    extraction = MockPoseExtractor().extract(((0.0, 1.0), (2.0, 3.0)))
    prediction = MockSignToTextModel().predict(IdentityTemporalSignEncoder().encode(extraction), extraction)
    assert prediction.text == "mock_sign"
    assert prediction.top_k_candidates[0].probability == 0.90
    assert prediction.token_probabilities["mock_sign"] == 0.90
    assert prediction.confidence == 0.90
    assert prediction.latency_ms >= 0
    assert prediction.model_version == "mock-sign-to-text-v1"
    assert prediction.input_quality.valid_pose_ratio == 1.0


def test_legacy_adapter_accepts_injected_local_classifier_without_model_download() -> None:
    extraction = MockPoseExtractor().extract(((1.0,),))
    prediction = LegacyIsolatedSignClassifierAdapter(FakeLegacyClassifier()).predict(
        IdentityTemporalSignEncoder().encode(extraction), extraction
    )
    assert prediction.text == "legacy-sign"
    assert prediction.model_version == "legacy-local-bilstm"


def test_transformer_placeholder_never_downloads_or_infers() -> None:
    extraction = MockPoseExtractor().extract(((1.0,),))
    encoded = IdentityTemporalSignEncoder().encode(extraction)
    with pytest.raises(RuntimeError, match="automatic pretrained downloads are disabled"):
        TransformerCheckpointAdapter().predict(encoded, extraction)
    with pytest.raises(NotImplementedError, match="not implemented"):
        TransformerCheckpointAdapter(Path("local-only-checkpoint")).predict(encoded, extraction)
