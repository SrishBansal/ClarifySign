"""
ClarifySign Backend - /api/predict endpoint
"""

import logging
from fastapi import APIRouter
from app.models.schemas import PredictRequest, PredictResponse, PredictionItem, UncertaintyMetrics
from app.services.inference import get_inference_service
from app.utils.exceptions import ModelNotLoadedError, InvalidFrameError, InferenceError

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run ISLBiLSTM inference on a base64-encoded video frame.

    Returns top-K predictions with calibrated probabilities and
    real-time uncertainty metrics (entropy, margin, ambiguity flag).
    """
    svc = get_inference_service()
    candidates, uncertainty, elapsed_ms = svc.predict(
        frame_b64=request.frame_b64,
        top_k=request.top_k,
    )

    predictions = [PredictionItem(label=lbl, probability=prob) for lbl, prob in candidates]
    top = predictions[0] if predictions else PredictionItem(label="unknown", probability=0.0)

    return PredictResponse(
        top_prediction=top,
        predictions=predictions,
        uncertainty=UncertaintyMetrics(**uncertainty),
        inference_ms=elapsed_ms,
    )
