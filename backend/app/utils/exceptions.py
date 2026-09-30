"""
ClarifySign Backend - Custom Exceptions
"""


class ClarifySignError(Exception):
    """Base exception for ClarifySign."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"


class ModelNotLoadedError(ClarifySignError):
    status_code = 503
    error_code = "MODEL_NOT_LOADED"


class InvalidFrameError(ClarifySignError):
    status_code = 400
    error_code = "INVALID_FRAME"


class InvalidLanguageError(ClarifySignError):
    status_code = 400
    error_code = "INVALID_LANGUAGE"


class DialogueStateError(ClarifySignError):
    status_code = 400
    error_code = "DIALOGUE_STATE_ERROR"


class InferenceError(ClarifySignError):
    status_code = 500
    error_code = "INFERENCE_ERROR"
