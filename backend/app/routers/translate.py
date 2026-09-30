"""
ClarifySign Backend - /api/translate and /api/speak endpoints
"""

import logging
from fastapi import APIRouter
from fastapi.responses import Response
from app.models.schemas import TranslateRequest, TranslateResponse, SpeakRequest
from app.services.translation import get_translation_service
from app.services.speech import get_speech_service
from app.utils.exceptions import InvalidLanguageError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    Translate a resolved intent string into Indian languages.
    If language is specified, returns only that language.
    Otherwise returns all 10 supported languages.
    """
    svc = get_translation_service()

    if request.language:
        text, source = svc.translate_single(request.intent, request.language)
        return TranslateResponse(
            intent=request.intent,
            translations={request.language: text},
            sources={request.language: source},
        )

    translations, sources = svc.translate_all(request.intent)
    return TranslateResponse(
        intent=request.intent,
        translations=translations,
        sources=sources,
    )


@router.post("/speak")
async def speak(request: SpeakRequest):
    """
    Generate MP3 audio from text using gTTS.
    Returns audio/mpeg bytes directly for browser playback.
    """
    svc = get_speech_service()
    try:
        from core.speech import synthesize_audio_bytes
        audio_bytes = synthesize_audio_bytes(request.text, request.language)
    except Exception as exc:
        logger.error("Speech synthesis error: %s", exc)
        audio_bytes = None

    if not audio_bytes:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Audio synthesis failed")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )
