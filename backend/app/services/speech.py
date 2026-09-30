"""
ClarifySign Backend - Speech Synthesis Service
Generates MP3 audio via gTTS and serves it as a static file.
Cleans up files older than 1 hour periodically.
"""

import sys
import time
import uuid
import logging
import threading
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.config import AUDIO_OUTPUT_DIR

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL = 600   # seconds between cleanup runs
_MAX_FILE_AGE = 3600      # 1 hour


class SpeechService:
    """Generates MP3 audio using gTTS and writes to AUDIO_OUTPUT_DIR."""

    _instance: "SpeechService | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._last_cleanup = time.time()
        return cls._instance

    # ─── Public API ──────────────────────────────────────────────────────────

    def synthesize(self, text: str, language: str = "Hindi") -> Optional[str]:
        """
        Synthesize speech for text in the given language.

        Returns:
            Relative URL path: /audio/<filename>.mp3
            or None if synthesis fails.
        """
        self._maybe_cleanup()

        try:
            from core.speech import synthesize_audio_bytes, SPEECH_LANG_CODES
            audio_bytes = synthesize_audio_bytes(text, language)
        except Exception as exc:
            logger.error("gTTS synthesis failed: %s", exc)
            return None

        if not audio_bytes:
            return None

        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = AUDIO_OUTPUT_DIR / filename
        try:
            filepath.write_bytes(audio_bytes)
        except Exception as exc:
            logger.error("Failed to write audio file: %s", exc)
            return None

        logger.debug("Audio written: %s (%d bytes)", filename, len(audio_bytes))
        return f"/audio/{filename}"

    # ─── Cleanup ─────────────────────────────────────────────────────────────

    def _maybe_cleanup(self):
        now = time.time()
        if now - self._last_cleanup < _CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        threading.Thread(target=self._cleanup_old_files, daemon=True).start()

    def _cleanup_old_files(self):
        cutoff = time.time() - _MAX_FILE_AGE
        for f in AUDIO_OUTPUT_DIR.glob("*.mp3"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    logger.debug("Deleted old audio: %s", f.name)
            except Exception:
                pass


def get_speech_service() -> SpeechService:
    return SpeechService()
