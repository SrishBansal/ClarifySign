"""
ClarifySign Backend - /api/dialogue/* endpoints
"""

import logging
from fastapi import APIRouter
from app.models.schemas import (
    DialogueStartRequest, DialogueStartResponse,
    DialogueClarifyRequest, DialogueClarifyResponse,
    ClarificationQuestion,
)
from app.services.dialogue import get_dialogue_service
from app.services.translation import get_translation_service
from app.utils.exceptions import DialogueStateError

router = APIRouter(prefix="/dialogue")
logger = logging.getLogger(__name__)


@router.post("/start", response_model=DialogueStartResponse)
async def dialogue_start(request: DialogueStartRequest):
    """
    Begin a new dialogue session given a prediction.

    - If the prediction is unambiguous → immediately return committed class + translations.
    - If ambiguous → return clarification question options.
    """
    dlg_svc = get_dialogue_service()
    trans_svc = get_translation_service()

    # Create fresh session
    dialogue_id = dlg_svc.create_session(request.language)

    # Build candidates list from request predictions
    candidates = [
        (item.label, item.probability)
        for item in request.prediction.predictions
    ]

    # Run EIG planner
    decision = dlg_svc.process_prediction(dialogue_id, candidates)
    action = decision.get("action", "commit")
    top_class = decision.get("candidate", candidates[0][0] if candidates else "unknown")

    if action == "commit":
        # Translate immediately
        translations, _ = trans_svc.translate_all(top_class)
        dlg_svc.commit_class(dialogue_id, top_class)
        return DialogueStartResponse(
            dialogue_id=dialogue_id,
            action="commit",
            confirmed_class=top_class,
            translation=translations,
            clarification=None,
            message=f"Recognized: {top_class}",
        )

    # action == "clarify"
    question = decision.get("question")
    clarification = None
    if question:
        clarification = ClarificationQuestion(
            text=question.text,
            options=question.options,
            candidate_keys=question.candidate_keys,
            question_type=question.question_type,
            information_gain=question.information_gain,
            utility=question.utility,
        )

    return DialogueStartResponse(
        dialogue_id=dialogue_id,
        action="clarify",
        confirmed_class=None,
        translation=None,
        clarification=clarification,
        message=question.text if question else "Please clarify your intent.",
    )


@router.post("/clarify", response_model=DialogueClarifyResponse)
async def dialogue_clarify(request: DialogueClarifyRequest):
    """
    Apply customer's clarification answer to a pending session.
    Returns the confirmed class and translations.
    """
    dlg_svc = get_dialogue_service()
    trans_svc = get_translation_service()

    session = dlg_svc.get_session(request.dialogue_id)
    confirmed = dlg_svc.resolve_clarification(request.dialogue_id, request.customer_response)

    translations, _ = trans_svc.translate_all(confirmed)

    return DialogueClarifyResponse(
        dialogue_id=request.dialogue_id,
        confirmed_class=confirmed,
        translation=translations,
        was_clarified=True,
        message=f"Confirmed: {confirmed}",
    )
