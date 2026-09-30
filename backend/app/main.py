"""
ClarifySign Backend - FastAPI Application Entry Point
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── ensure project root on path ───────────────────────────────────────────────
_app_dir = Path(__file__).resolve().parent
_backend_dir = _app_dir.parent
_project_root = _backend_dir.parent

# Remove _app_dir if present so 'config' doesn't collide with app.config
if str(_app_dir) in sys.path:
    sys.path.remove(str(_app_dir))

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_backend_dir) not in sys.path:
    sys.path.insert(1, str(_backend_dir))

from app.config import CORS_ORIGINS, DEBUG, AUDIO_OUTPUT_DIR
from app.utils.logging import configure_logging, get_logger
from app.utils.exceptions import ClarifySignError
from app.models.schemas import ErrorResponse, HealthResponse
from app.routers import predict, dialogue, translate
from app.services.inference import get_inference_service

configure_logging(debug=DEBUG)
logger = get_logger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ClarifySign API",
    description=(
        "Real-time Indian Sign Language recognition with EIG-based clarification dialogue. "
        "The distinguishing contribution is the Expected Information Gain clarification-decision "
        "layer (core/clarification.py + core/uncertainty.py + core/dialogue.py)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (audio outputs) ──────────────────────────────────────────────
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_OUTPUT_DIR)), name="audio")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router, prefix="/api", tags=["inference"])
app.include_router(dialogue.router, prefix="/api", tags=["dialogue"])
app.include_router(translate.router, prefix="/api", tags=["translation"])

# ── Exception handler ─────────────────────────────────────────────────────────
@app.exception_handler(ClarifySignError)
async def clarifysign_exception_handler(request: Request, exc: ClarifySignError):
    logger.error("%s: %s", type(exc).__name__, exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc), "error_code": exc.error_code},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
    )

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["health"])
@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health():
    svc = get_inference_service()
    return HealthResponse(
        status="healthy",
        model_loaded=svc.model_loaded,
        device=svc.device,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("ClarifySign API starting up…")
    # Eagerly initialise inference service so first request is fast
    svc = get_inference_service()
    logger.info(
        "Startup complete. model_loaded=%s device=%s",
        svc.model_loaded,
        svc.device,
    )
