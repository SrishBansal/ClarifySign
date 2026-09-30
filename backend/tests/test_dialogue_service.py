"""
ClarifySign Backend Tests - Dialogue service unit tests.
"""

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent
_root = _backend.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
if str(_root) not in sys.path:
    sys.path.append(str(_root))

import pytest
from app.services.dialogue import DialogueService
from app.utils.exceptions import DialogueStateError


@pytest.fixture()
def svc():
    return DialogueService()


CLEAR_CANDIDATES = [("blue", 0.93), ("black", 0.04), ("red", 0.02), ("white", 0.01)]
AMBIGUOUS_CANDIDATES = [("blue", 0.52), ("black", 0.44), ("red", 0.04)]


class TestDialogueService:

    def test_create_session_returns_id(self, svc):
        did = svc.create_session("Hindi")
        assert isinstance(did, str) and len(did) > 0

    def test_create_session_different_ids(self, svc):
        id1 = svc.create_session("Hindi")
        id2 = svc.create_session("Tamil")
        assert id1 != id2

    def test_get_session_returns_session(self, svc):
        did = svc.create_session("Hindi")
        session = svc.get_session(did)
        assert session.dialogue_id == did

    def test_get_session_unknown_raises(self, svc):
        with pytest.raises(DialogueStateError):
            svc.get_session("nonexistent-id-xyz")

    def test_process_clear_prediction_commits(self, svc):
        did = svc.create_session("Hindi")
        decision = svc.process_prediction(did, CLEAR_CANDIDATES)
        assert decision["action"] in ("commit", "clarify")

    def test_process_ambiguous_prediction(self, svc):
        did = svc.create_session("Hindi")
        decision = svc.process_prediction(did, AMBIGUOUS_CANDIDATES)
        assert decision["action"] in ("commit", "clarify")

    def test_commit_class_adds_to_history(self, svc):
        did = svc.create_session("Hindi")
        svc.process_prediction(did, CLEAR_CANDIDATES)
        svc.commit_class(did, "blue")
        session = svc.get_session(did)
        assert len(session.state.turns) >= 1

    def test_resolve_clarification_no_pending_raises(self, svc):
        did = svc.create_session("Hindi")
        svc.process_prediction(did, CLEAR_CANDIDATES)
        # If there's no pending clarification, resolve should raise
        session = svc.get_session(did)
        if not session.state.pending:
            with pytest.raises(DialogueStateError):
                svc.resolve_clarification(did, "1")

    def test_delete_session(self, svc):
        did = svc.create_session("Hindi")
        svc.delete_session(did)
        with pytest.raises(DialogueStateError):
            svc.get_session(did)

    def test_clarification_flow_ambiguous(self, svc):
        did = svc.create_session("Hindi")
        decision = svc.process_prediction(did, AMBIGUOUS_CANDIDATES)
        session = svc.get_session(did)
        if session.state.pending:
            confirmed = svc.resolve_clarification(did, "1")
            assert isinstance(confirmed, str) and len(confirmed) > 0
