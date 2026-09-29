"""
End-to-End Pipeline Smoke Test for ClarifySign
Verifies complete flow:
Landmark sequence -> Model forward pass -> Uncertainty metrics ->
Clarification planner -> Dialogue state resolution -> Multilingual output (10 languages) -> Speech HTML.
"""

import numpy as np
import torch
import pytest
from config import LANGUAGES, FEATURE_DIM, SEQUENCE_LENGTH
from core.landmarks import resample_sequence
from core.recognizer import ISLBiLSTM
from core.uncertainty import report
from core.clarification import Planner
from core.dialogue import DialogueState
from core.translation import MultilingualTranslator
from core.speech import get_browser_speech_html


def test_language_routing_10_languages():
    """Verifies that all 10 official Indian languages return valid translations."""
    translator = MultilingualTranslator()
    assert len(LANGUAGES) == 10

    test_concept = "blue"
    for lang_name, lang_code in LANGUAGES.items():
        translated, backend = translator.translate_semantic(test_concept, lang_name)
        assert translated is not None
        assert len(translated) > 0
        assert lang_name in LANGUAGES
        assert backend in ["verified_multilingual_semantic", f"neural_{translator.model_name}", "semantic_fallback"]


def test_isl_bilstm_architecture_forward():
    """Verifies PyTorch BiLSTM neural network forward pass dimensions."""
    model = ISLBiLSTM(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=5
    )
    model.eval()

    dummy_input = torch.randn(2, SEQUENCE_LENGTH, FEATURE_DIM)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 5)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0, 1.0]), atol=1e-5)


def test_end_to_end_smoke_flow():
    """Complete integration smoke test through the whole stack."""
    # 1. Simulate landmark sequence
    raw_seq = np.random.randn(30, FEATURE_DIM).astype(np.float32)
    seq = resample_sequence(raw_seq, target_length=SEQUENCE_LENGTH)
    assert seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM)

    # 2. Simulate candidate distribution
    candidates = [("blue", 0.49), ("black", 0.39), ("green", 0.12)]

    # 3. Uncertainty Report
    unc_report = report(candidates)
    assert unc_report["is_ambiguous"] is True
    assert unc_report["entropy_bits"] > 1.0

    # 4. Clarification Planning
    dialogue = DialogueState()
    planner = Planner()
    decision = planner.decide(candidates, context=dialogue.context)

    assert decision["action"] == "clarify"
    question = decision["question"]
    assert question is not None
    assert len(question.options) >= 2

    # 5. Dialogue Resolution
    dialogue.set_pending(question)
    assert dialogue.pending is True
    resolved = dialogue.resolve("1")
    assert resolved == "blue"
    assert dialogue.pending is False
    dialogue.add("customer", "Option 1 (Blue)", semantic=resolved, language="Hindi")

    # 6. Multilingual Translation into all 10 Indian Languages
    translator = MultilingualTranslator()
    for lang in LANGUAGES:
        out_text, backend = translator.translate_semantic(resolved, lang)
        assert len(out_text) > 0

    # 7. Speech output HTML generation
    speech_html = get_browser_speech_html("क्या आपके पास यह नीले रंग में है?", language="Hindi")
    assert "SpeechSynthesisUtterance" in speech_html
    assert "hi-IN" in speech_html
