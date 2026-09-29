"""
ClarifySign - Configuration Module
Defines system-wide constants, paths, supported languages, model dimensions,
and information-gain clarification hyperparameters.
"""

from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
DOCS_DIR = ROOT_DIR / "docs"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Default Model Paths
MODEL_PATH = MODELS_DIR / "isl_bilstm.pt"
LABELS_PATH = MODELS_DIR / "labels.json"
PREPROCESSING_CONFIG_PATH = MODELS_DIR / "preprocessing_config.json"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"
HOLISTIC_TASK_PATH = MODELS_DIR / "holistic_landmarker.task"

# MediaPipe & Landmark Feature Dimensions
# Left hand: 21 landmarks * 3 (x, y, z) = 63
# Right hand: 21 landmarks * 3 (x, y, z) = 63
# Pose: 33 landmarks * 3 (x, y, z) = 99
# Total feature dimension = 63 + 63 + 99 = 225
FEATURE_DIM = 225
SEQUENCE_LENGTH = 48
FPS_TARGET = 25

# 10 Official Indian Languages + English (Flores-200 / IndicTrans2 codes)
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
    "Odia": "ory_Orya"
}

# Recognition Model Architecture Hyperparameters
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
DENSE_HIDDEN_DIM = 128
DROPOUT_RATE = 0.30
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# Uncertainty & Clarification Hyperparameters
# Ambiguity Trigger Conditions:
# 1. Top-1 probability < CONFIDENCE_THRESHOLD
# 2. Margin (p1 - p2) < MARGIN_THRESHOLD
# 3. Shannon Entropy H(Y|X) > ENTROPY_THRESHOLD (bits)
CONFIDENCE_THRESHOLD = 0.78
MARGIN_THRESHOLD = 0.20
ENTROPY_THRESHOLD = 1.25  # bits (for log2)

# Clarification Utility:
# U(q) = IG(q) + alpha * Context(q) + beta * Answerability(q) - lambda * Cost(q)
UTILITY_THRESHOLD = 0.05
ALPHA_CONTEXT = 0.30        # alpha: contextual relevance weight
BETA_ANSWERABILITY = 0.15   # beta: cognitive answerability weight
LAMBDA_COST = 0.18          # lambda: interaction cost penalty

# Answer reliability assumed by the planner's noisy answer model.
# THIS IS A MODELLING ASSUMPTION — it is NOT measured from real users.
# rho_n = max(0.5, RHO0 - 0.03*(n-2))  where n is the number of options.
# Exposed in the Research Inspector so it is visibly disclosed, not hidden.
RHO0 = 0.95

# Temperature scaling parameter (T=1.0 means no scaling).
# Fit on validation split ONLY via training/evaluate_recognizer.py.
# Set > 1 to soften overconfident outputs; ECE before/after reported in docs/DATASET.md.
TEMPERATURE = 1.0

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
    "white"
]
