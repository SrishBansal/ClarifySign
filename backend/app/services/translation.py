"""
ClarifySign Backend - Translation Service
Wraps core.translation.MultilingualTranslator with caching and source tracking.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Thin service layer over core.translation.MultilingualTranslator.
    Lazily initialises the translator on first use to avoid blocking startup.
    """

    _instance: "TranslationService | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._translator = None
            cls._instance._initialized = False
        return cls._instance

    def _ensure_init(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            from core.translation import MultilingualTranslator
            self._translator = MultilingualTranslator()
            logger.info("MultilingualTranslator initialized")
        except Exception as exc:
            logger.error("TranslationService init failed: %s", exc)
            self._translator = None

    def translate_all(self, intent: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Translate intent into all 10 supported Indian languages.

        Returns:
            translations: {language_name -> translated_text}
            sources:      {language_name -> "verified" | "nllb" | "fallback"}
        """
        self._ensure_init()

        from config import LANGUAGES  # uses project-root config
        from core.translation import VERIFIED_SEMANTIC_DATABASE

        intent_key = intent.strip().lower()
        translations: Dict[str, str] = {}
        sources: Dict[str, str] = {}

        for lang_name in LANGUAGES.keys():
            # Check verified semantic database first
            if intent_key in VERIFIED_SEMANTIC_DATABASE:
                db_entry = VERIFIED_SEMANTIC_DATABASE[intent_key]
                if lang_name in db_entry:
                    translations[lang_name] = db_entry[lang_name]
                    sources[lang_name] = "verified"
                    continue

            # Try neural translator
            if self._translator is not None:
                try:
                    result = self._translator.translate(intent, lang_name)
                    translations[lang_name] = result.get("text", intent)
                    sources[lang_name] = result.get("source", "nllb")
                    continue
                except Exception as exc:
                    logger.warning("Neural translation failed (%s→%s): %s", intent, lang_name, exc)

            # Last resort: return the intent string itself
            translations[lang_name] = intent
            sources[lang_name] = "fallback"

        return translations, sources

    def translate_single(self, intent: str, language: str) -> Tuple[str, str]:
        """Translate intent into a single language. Returns (text, source)."""
        self._ensure_init()

        from core.translation import VERIFIED_SEMANTIC_DATABASE

        intent_key = intent.strip().lower()

        if intent_key in VERIFIED_SEMANTIC_DATABASE:
            db_entry = VERIFIED_SEMANTIC_DATABASE[intent_key]
            if language in db_entry:
                return db_entry[language], "verified"

        if self._translator is not None:
            try:
                result = self._translator.translate(intent, language)
                return result.get("text", intent), result.get("source", "nllb")
            except Exception as exc:
                logger.warning("Single translation failed: %s", exc)

        return intent, "fallback"


def get_translation_service() -> TranslationService:
    return TranslationService()
