"""Local-development composition root. Production providers are injected here later."""

from clarifysign.api.service import ClarifySignService
from clarifysign.clarification.service import ClarificationEngine
from clarifysign.config.settings import AppSettings
from clarifysign.dialogue.state import DialogueState
from clarifysign.preprocessing.providers import IdentityPosePreprocessor
from clarifysign.sign_to_text.providers import MockSignToTextProvider
from clarifysign.translation.providers import MockTranslationProvider


def build_local_service(settings: AppSettings | None = None) -> ClarifySignService:
    active_settings = settings or AppSettings()
    return ClarifySignService(
        preprocessor=IdentityPosePreprocessor(),
        recognizer=MockSignToTextProvider(),
        clarification=ClarificationEngine(active_settings.clarification),
        translator=MockTranslationProvider(),
        dialogue=DialogueState(),
    )
