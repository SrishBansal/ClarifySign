"""
Tests for core/dialogue.py
Verifies turn logging, context decay, domain tracking, and clarification resolution.
"""

from core.dialogue import DialogueState, Turn


def test_dialogue_turn_logging():
    d = DialogueState()
    d.add("customer", "hello", semantic="hello", language="English")
    assert len(d.turns) == 1
    assert d.turns[0].speaker == "customer"
    assert d.turns[0].semantic == "hello"
    assert "hello" in d.context


def test_dialogue_context_decay():
    d = DialogueState(decay_factor=0.8)
    d.add("customer", "shirt", semantic="shirt")
    val_init = d.context["shirt"]

    # Add second turn
    d.add("customer", "blue", semantic="blue")
    assert d.context["shirt"] < val_init
    assert "blue" in d.context


def test_dialogue_resolve_numeric_option():
    d = DialogueState()
    options = ["1. Blue", "2. Black", "3. Neither"]
    candidate_keys = ["blue", "black", "other"]

    class MockQuestion:
        def __init__(self, opts, keys):
            self.options = opts
            self.candidate_keys = keys

    d.set_pending(MockQuestion(options, candidate_keys))
    assert d.pending

    resolved = d.resolve("1")
    assert resolved == "blue"
    assert not d.pending
    assert "blue" in d.context


def test_dialogue_resolve_text_option():
    d = DialogueState()
    options = ["1. Small", "2. Big"]
    candidate_keys = ["small", "big"]

    class MockQuestion:
        def __init__(self, opts, keys):
            self.options = opts
            self.candidate_keys = keys

    d.set_pending(MockQuestion(options, candidate_keys))
    resolved = d.resolve("Small")
    assert resolved == "small"
    assert not d.pending


def test_dialogue_reset():
    d = DialogueState()
    d.add("customer", "tshirt", semantic="tshirt")
    assert len(d.turns) > 0
    d.reset()
    assert len(d.turns) == 0
    assert len(d.context) == 0
    assert not d.pending
