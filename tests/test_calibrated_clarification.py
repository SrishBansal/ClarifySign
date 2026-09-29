from pathlib import Path

from clarifysign.clarification.engine import CalibratedClarificationEngine, ClarificationPolicySettings, SQLiteDialogueStore
from clarifysign.clarification.schemas import CandidateProbability, InputQualityMetrics, SignToTextPrediction, SpeechToTextPrediction


def prediction(
    candidates: tuple[CandidateProbability, ...],
    confidence: float,
    quality: InputQualityMetrics | None = None,
    source: str = "sign",
) -> SignToTextPrediction | SpeechToTextPrediction:
    values = {
        "text": candidates[0].text,
        "top_k_candidates": candidates,
        "token_probabilities": {candidate.text: candidate.probability for candidate in candidates},
        "confidence": confidence,
        "latency_ms": 1.0,
        "model_version": "calibrated-test-v1",
        "input_quality": quality or InputQualityMetrics(valid_signal_ratio=1.0, temporal_coverage=1.0, frame_or_chunk_count=12),
    }
    return SpeechToTextPrediction(**values) if source == "speech" else SignToTextPrediction(**values)


def engine(tmp_path: Path) -> CalibratedClarificationEngine:
    return CalibratedClarificationEngine(SQLiteDialogueStore(tmp_path / "dialogue.sqlite"))


def test_high_confidence_prediction_returns_final_text_without_clarification(tmp_path: Path) -> None:
    result = engine(tmp_path).review(prediction((CandidateProbability(text="book", probability=0.95),), 0.95), "s1")
    assert result.status == "final"
    assert result.final_text == "book"


def test_ambiguous_prediction_generates_structured_options_and_utility(tmp_path: Path) -> None:
    result = engine(tmp_path).review(
        prediction((CandidateProbability(text="blue", probability=0.50, semantic_group="colour"), CandidateProbability(text="black", probability=0.45, semantic_group="colour")), 0.50),
        "s2",
    )
    assert result.status == "clarification_required"
    assert result.request is not None
    assert result.request.modality == "yes_no_gesture"
    assert [option.text for option in result.request.options] == ["blue", "black"]
    assert result.request.expected_utility == result.request.expected_entropy_reduction / result.request.interaction_cost
    confirmed = engine(tmp_path).confirm(result.request, "candidate_1")
    assert confirmed.status == "final"
    assert confirmed.final_text == "blue"


def test_low_video_quality_triggers_spoken_clarification(tmp_path: Path) -> None:
    result = engine(tmp_path).review(
        prediction(
            (CandidateProbability(text="shirt", probability=0.90), CandidateProbability(text="shoes", probability=0.05)), 0.90,
            InputQualityMetrics(valid_signal_ratio=0.20, temporal_coverage=0.30, frame_or_chunk_count=3),
        ),
        "s3",
    )
    assert result.status == "clarification_required"
    assert result.request is not None
    assert result.request.modality == "speech"
    assert "poor input quality" in result.request.trigger_reasons


def test_semantic_conflict_and_speech_entropy_choose_appropriate_modalities(tmp_path: Path) -> None:
    conflict = engine(tmp_path).review(
        prediction((CandidateProbability(text="red", probability=0.80, semantic_group="colour"), CandidateProbability(text="pen", probability=0.10, semantic_group="object")), 0.80), "s4"
    )
    assert conflict.request is not None and "semantically conflicting candidates" in conflict.request.trigger_reasons
    speech = engine(tmp_path).review(
        prediction((CandidateProbability(text="one", probability=0.34), CandidateProbability(text="two", probability=0.33), CandidateProbability(text="three", probability=0.32)), 0.34, source="speech"), "s5"
    )
    assert speech.request is not None and speech.request.modality == "sign_animation"


def test_repeated_failures_return_safe_fallback_and_confirmation_resets_counter(tmp_path: Path) -> None:
    store = SQLiteDialogueStore(tmp_path / "dialogue.sqlite")
    active = CalibratedClarificationEngine(store, ClarificationPolicySettings(maximum_failures=2))
    ambiguous = prediction((CandidateProbability(text="a", probability=0.50), CandidateProbability(text="b", probability=0.45)), 0.50)
    first = active.review(ambiguous, "s6")
    assert first.request is not None
    active.confirm(first.request, "candidate_1")
    assert store.failures("s6") == 0
    active.review(ambiguous, "s6")
    active.review(ambiguous, "s6")
    fallback = active.review(ambiguous, "s6")
    assert fallback.status == "safe_fallback"
    assert fallback.final_text is None
