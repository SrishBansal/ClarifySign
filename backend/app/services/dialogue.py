"""
ClarifySign Backend - Dialogue Service
In-memory session store + EIG-based clarification planner wrapping core modules.
"""

import sys
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Optional

_project_root = Path(__file__).resolve().parent.parent.parent.parent
for _p in [str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.config import (
    CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD, ENTROPY_THRESHOLD,
    UTILITY_THRESHOLD, ALPHA_CONTEXT, BETA_ANSWERABILITY, LAMBDA_COST,
    RHO0, DIALOGUE_TIMEOUT_SECONDS,
)
from app.utils.exceptions import DialogueStateError

logger = logging.getLogger(__name__)


class SessionEntry:
    """Container for a live dialogue session."""

    def __init__(self, dialogue_id: str, language: str):
        from core.dialogue import DialogueState
        from core.clarification import Planner

        self.dialogue_id = dialogue_id
        self.language = language
        self.state = DialogueState()
        self.planner = Planner(
            threshold=UTILITY_THRESHOLD,
            alpha=ALPHA_CONTEXT,
            beta=BETA_ANSWERABILITY,
            lambda_cost=LAMBDA_COST,
            conf_threshold=CONFIDENCE_THRESHOLD,
            margin_threshold=MARGIN_THRESHOLD,
            entropy_threshold=ENTROPY_THRESHOLD,
            rho0=RHO0,
        )
        self.candidates = []   # last prediction candidates
        self.created_at = time.time()
        self.last_activity = time.time()

    def touch(self):
        self.last_activity = time.time()

    @property
    def expired(self) -> bool:
        return (time.time() - self.last_activity) > DIALOGUE_TIMEOUT_SECONDS


class DialogueService:
    """
    Manages dialogue sessions:
    - create_session(): returns dialogue_id
    - process_prediction(): runs clarification planner and returns decision dict
    - resolve_clarification(): applies customer answer, returns confirmed class
    """

    def __init__(self):
        self._sessions: Dict[str, SessionEntry] = {}

    # ─── Session lifecycle ────────────────────────────────────────────────────

    def create_session(self, language: str) -> str:
        self._gc()
        dialogue_id = str(uuid.uuid4())
        self._sessions[dialogue_id] = SessionEntry(dialogue_id, language)
        logger.info("New dialogue session %s (lang=%s)", dialogue_id, language)
        return dialogue_id

    def get_session(self, dialogue_id: str) -> SessionEntry:
        self._gc()
        session = self._sessions.get(dialogue_id)
        if session is None:
            raise DialogueStateError(f"Dialogue session '{dialogue_id}' not found or expired.")
        session.touch()
        return session

    def delete_session(self, dialogue_id: str):
        self._sessions.pop(dialogue_id, None)

    def _gc(self):
        """Remove expired sessions."""
        expired_ids = [did for did, s in self._sessions.items() if s.expired]
        for did in expired_ids:
            logger.debug("GC: removing expired session %s", did)
            del self._sessions[did]

    # ─── Core logic ──────────────────────────────────────────────────────────

    def process_prediction(
        self, dialogue_id: str, candidates: list
    ) -> dict:
        """
        Run the EIG planner on the given candidates within session context.

        Returns a dict matching core.clarification.Planner.decide() output:
          { action, candidate, confidence, reason, question, all_questions, triggers }
        """
        session = self.get_session(dialogue_id)
        session.candidates = candidates
        context = dict(session.state.context)
        decision = session.planner.decide(candidates, context=context)
        logger.info(
            "Session %s decision=%s candidate=%s confidence=%.3f",
            dialogue_id,
            decision.get("action"),
            decision.get("candidate"),
            decision.get("confidence", 0.0),
        )
        # If clarifying, store pending question on state
        if decision.get("action") == "clarify" and decision.get("question"):
            session.state.set_pending(decision["question"])
        return decision

    def resolve_clarification(self, dialogue_id: str, customer_response: str) -> str:
        """
        Apply customer's answer to the pending clarification.
        Returns the confirmed class string.
        """
        session = self.get_session(dialogue_id)
        if not session.state.pending:
            raise DialogueStateError(
                f"Session {dialogue_id} has no pending clarification to resolve."
            )
        confirmed = session.state.resolve(customer_response)
        session.state.add(
            speaker="customer",
            raw=customer_response,
            semantic=confirmed,
            language=session.language,
            was_clarified=True,
        )
        logger.info("Session %s resolved -> %s", dialogue_id, confirmed)
        return confirmed

    def commit_class(self, dialogue_id: str, class_name: str):
        """Record a committed (non-clarified) class in dialogue history."""
        session = self.get_session(dialogue_id)
        session.state.add(
            speaker="customer",
            raw=class_name,
            semantic=class_name,
            language=session.language,
            was_clarified=False,
        )


# Module-level singleton
_service: Optional[DialogueService] = None


def get_dialogue_service() -> DialogueService:
    global _service
    if _service is None:
        _service = DialogueService()
    return _service
