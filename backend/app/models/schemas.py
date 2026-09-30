"""
ClarifySign Backend - Pydantic Models / Schemas
All request/response shapes for the API.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ─── Predict ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    frame_b64: str = Field(..., description="Base-64 encoded JPEG/PNG frame")
    top_k: int = Field(5, ge=1, le=17)


class PredictionItem(BaseModel):
    label: str
    probability: float


class UncertaintyMetrics(BaseModel):
    top1_confidence: float
    margin: float
    entropy_bits: float
    normalized_entropy: float
    is_ambiguous: bool
    primary_reason: str
    triggered_criteria: List[str]
    decision: str  # "commit" | "clarify"


class PredictResponse(BaseModel):
    top_prediction: PredictionItem
    predictions: List[PredictionItem]
    uncertainty: UncertaintyMetrics
    inference_ms: float


# ─── Dialogue / Start ─────────────────────────────────────────────────────────

class DialogueStartRequest(BaseModel):
    prediction: PredictResponse
    language: str = Field("Hindi", description="Display language name, e.g. 'Hindi'")


class ClarificationQuestion(BaseModel):
    text: str
    options: List[str]
    candidate_keys: List[str]
    question_type: str
    information_gain: float
    utility: float


class DialogueStartResponse(BaseModel):
    dialogue_id: str
    action: str  # "commit" | "clarify"
    confirmed_class: Optional[str] = None
    translation: Optional[Dict[str, str]] = None  # lang -> text
    clarification: Optional[ClarificationQuestion] = None
    message: str


# ─── Dialogue / Clarify ───────────────────────────────────────────────────────

class DialogueClarifyRequest(BaseModel):
    dialogue_id: str
    customer_response: str  # option text or numeric index ("1", "2", ...)


class DialogueClarifyResponse(BaseModel):
    dialogue_id: str
    confirmed_class: str
    translation: Dict[str, str]
    was_clarified: bool
    message: str


# ─── Translate ────────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    intent: str
    language: Optional[str] = None  # None → return all 10 languages


class TranslateResponse(BaseModel):
    intent: str
    translations: Dict[str, str]   # language_name -> translated_text
    sources: Dict[str, str]        # language_name -> "verified" | "nllb"


# ─── Speech ──────────────────────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    text: str
    language: str = "Hindi"


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# ─── Error ───────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
