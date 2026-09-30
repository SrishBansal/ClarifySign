"""
ClarifySign Backend - Configuration Module
Loads settings from environment variables and sets sensible defaults.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent  # repo root

# Core modules live in repo root (not duplicated into backend)
CORE_DIR = PROJECT_ROOT
MODELS_DIR = os.environ.get("MODELS_DIR", str(PROJECT_ROOT / "models"))

MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(MODELS_DIR) / "isl_bilstm.pt"))
LABELS_PATH = os.environ.get("LABELS_PATH", str(Path(MODELS_DIR) / "labels.json"))
HOLISTIC_TASK_PATH = os.environ.get(
    "HOLISTIC_TASK_PATH", str(Path(MODELS_DIR) / "holistic_landmarker.task")
)
AUDIO_OUTPUT_DIR = BACKEND_DIR / "outputs" / "audio"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
CORS_ORIGINS: List[str] = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"
).split(",")

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = os.environ.get("DEVICE", "auto")  # "auto" | "cpu" | "cuda" | "mps"

# ─── Thresholds (same as core/config.py defaults) ────────────────────────────
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.78"))
MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "0.20"))
ENTROPY_THRESHOLD = float(os.environ.get("ENTROPY_THRESHOLD", "1.25"))
UTILITY_THRESHOLD = float(os.environ.get("UTILITY_THRESHOLD", "0.05"))

# ─── EIG Weights ─────────────────────────────────────────────────────────────
ALPHA_CONTEXT = float(os.environ.get("ALPHA_CONTEXT", "0.30"))
BETA_ANSWERABILITY = float(os.environ.get("BETA_ANSWERABILITY", "0.15"))
LAMBDA_COST = float(os.environ.get("LAMBDA_COST", "0.18"))
RHO0 = float(os.environ.get("RHO0", "0.95"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))

# ─── Dialogue ────────────────────────────────────────────────────────────────
DIALOGUE_TIMEOUT_SECONDS = int(os.environ.get("DIALOGUE_TIMEOUT_SECONDS", "300"))

# ─── Feature dimensions & Architecture (matches core/config.py) ──────────────
FEATURE_DIM = 225
SEQUENCE_LENGTH = 48
FPS_TARGET = 25

LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
DENSE_HIDDEN_DIM = 128
DROPOUT_RATE = 0.30
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# 10 Official Indian Languages + English
LANGUAGES = {
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva",
    "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Odia": "ory_Orya",
}

# Shopkeeper Domain Sign Classes (curated from AI4Bharat INCLUDE dataset)
SHOPKEEPER_CLASSES = [
    "bank",
    "biglarge",
    "black",
    "blue",
    "cellphone",
    "good",
    "hello",
    "hot",
    "new",
    "pen",
    "red",
    "shoes",
    "smalllittle",
    "storeorshop",
    "thankyou",
    "tshirt",
    "white",
]

